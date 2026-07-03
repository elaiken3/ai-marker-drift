"""
Pull PubMed abstracts + publication dates for a date range and search query.

Uses NCBI E-utilities (esearch -> efetch), which are free and don't require
an API key, but are rate-limited to 3 req/sec without a key and 10 req/sec
with one (set NCBI_API_KEY env var to use a key).

NCBI only lets you retrieve the first ~9,999 records of any query (retstart is
capped), so the query date range is recursively bisected until each slice is
under the cap. The window filters on `--datetype` (default 'edat', the Entrez
index date, which is spread evenly day-to-day; 'pdat' piles year-only records
onto Jan 1, making single days overflow the cap). Each record's `date` field is
still its actual publication date from the XML regardless of --datetype. Work is
checkpointed per calendar month into a `{out}.parts/` directory (one shard per
completed month) and concatenated into `--out` at the end -- so an interrupted
pull resumes where it left off instead of restarting.

Output: one JSON object per line (jsonl) with fields:
    {"id": pmid, "date": "YYYY-MM-DD" or "YYYY-MM", "text": abstract_text}

Usage:
    python src/ingest/fetch_pubmed.py \
        --query "cancer AND treatment" \
        --start 2020-01-01 --end 2026-06-01 \
        --out data/raw/pubmed_cancer.jsonl
"""
import argparse
import datetime
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from tqdm import tqdm

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BATCH_SIZE = 200  # pmids per efetch call
# NCBI rejects retstart > 9998 for both esearch and efetch, so a query date
# range must be sliced until each slice yields at most this many records
# (kept a hair under 10k so the last BATCH_SIZE page stays within the limit).
MAX_SLICE = 9990


def get_api_key() -> str | None:
    return os.environ.get("NCBI_API_KEY")


def rate_limit_sleep():
    time.sleep(0.11 if get_api_key() else 0.35)


# Transport-level failures that are worth retrying (NCBI truncating a response
# mid-stream, dropped connections, timeouts). These are RequestException
# subclasses but NOT HTTPError, so a bare `except HTTPError` misses them.
TRANSIENT_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def http_get(endpoint: str, params: dict, timeout: int, retries: int = 4) -> requests.Response:
    """GET an E-utilities endpoint, retrying transient failures with backoff.

    Retries on dropped/truncated connections and on 5xx/429 responses. Does NOT
    retry other 4xx -- in particular the deterministic 400 that NCBI returns for
    retstart past its cap, which the bisection logic in fetch_date_range relies
    on failing fast.
    """
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(f"{EUTILS_BASE}/{endpoint}", params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status is not None and status != 429 and status < 500:
                raise  # deterministic client error (e.g. 400 cap) -- don't retry
            last_err = e
        except TRANSIENT_ERRORS as e:
            last_err = e
        time.sleep(2 ** attempt)
    raise RuntimeError(f"E-utilities {endpoint} failed after {retries} attempts: {last_err}")


def eutils_get_json(endpoint: str, params: dict, retries: int = 3) -> dict:
    """GET an E-utilities endpoint and parse JSON tolerantly.

    NCBI occasionally emits JSON containing raw (unescaped) control characters
    -- most often in the echoed query translation. requests' .json() parses in
    strict mode and rejects those with 'Invalid control character', so parse
    with strict=False instead. Transient dirty/truncated bodies are retried
    with backoff.
    """
    last_err = None
    for attempt in range(retries):
        r = http_get(endpoint, params, timeout=30)
        try:
            return json.loads(r.text, strict=False)
        except json.JSONDecodeError as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(
        f"E-utilities {endpoint} returned unparseable JSON after {retries} attempts: {last_err}"
    )


def esearch_history(query: str, start: str, end: str, datetype: str = "edat") -> tuple[int, str, str]:
    """Post the query to NCBI's history server and return (total, WebEnv, query_key).

    We deliberately do NOT page esearch to collect PMIDs: esearch can only reach
    the first 9,999 records of a query (retstart > 9998 is rejected), so any query
    with more results than that cannot be fully listed this way. Instead we store
    the result set on the history server and efetch straight from it (see
    efetch_from_history), which has no such ceiling.

    `datetype` is the date field the mindate/maxdate window filters on. Default
    'edat' (Entrez index date) is roughly uniform day-to-day; 'pdat' (publication
    date) piles year-only records onto Jan 1, producing single-day buckets far
    above the retrieval cap that cannot be sliced smaller.
    """
    params = {
        "db": "pubmed",
        "term": query,
        "datetype": datetype,
        "mindate": start,
        "maxdate": end,
        "retmode": "json",
        "retmax": 0,
        "usehistory": "y",
    }
    if get_api_key():
        params["api_key"] = get_api_key()

    data = eutils_get_json("esearch.fcgi", params)["esearchresult"]
    if "count" not in data or "webenv" not in data:
        problem = {k: data[k] for k in ("ERROR", "WarningList", "errorlist") if k in data}
        raise RuntimeError(f"esearch did not return a usable history handle. NCBI response: {problem or data}")
    return int(data["count"]), data["webenv"], data["querykey"]


def parse_pubdate(article_elem) -> str | None:
    """Extract a YYYY-MM-DD or YYYY-MM string from PubmedArticle XML."""
    pubdate = article_elem.find(".//Journal/JournalIssue/PubDate")
    if pubdate is None:
        return None
    year = pubdate.findtext("Year")
    month = pubdate.findtext("Month")
    day = pubdate.findtext("Day")
    if not year:
        medline_date = pubdate.findtext("MedlineDate")
        return medline_date[:4] if medline_date else None

    month_map = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
        "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
    }
    if month:
        month = month_map.get(month, month.zfill(2) if month.isdigit() else "01")
    else:
        return year
    if day:
        return f"{year}-{month}-{day.zfill(2)}"
    return f"{year}-{month}"


def parse_efetch_xml(content: bytes) -> list[dict]:
    """Parse an efetch PubMed XML response into {id, date, text} records."""
    root = ET.fromstring(content)
    records = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID")
        date = parse_pubdate(article)
        abstract_parts = article.findall(".//Abstract/AbstractText")
        text = " ".join(a.text for a in abstract_parts if a.text)
        if text and date:
            records.append({"id": pmid, "date": date, "text": text})
    return records


# Characters legal in XML 1.0 are tab/newline/CR and the printable ranges; a stray
# control char or bad byte in a PubMed abstract makes the whole response
# not-well-formed. Strip anything outside the legal set before reparsing.
_ILLEGAL_XML = re.compile(
    "[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]"
)


def sanitize_xml(content: bytes) -> bytes:
    """Drop illegal XML characters/bad bytes so a reparse can succeed. Returns
    bytes (not str) so the XML encoding declaration still parses."""
    text = content.decode("utf-8", errors="replace")
    return _ILLEGAL_XML.sub("", text).encode("utf-8")


def efetch_from_history(webenv: str, query_key: str, retstart: int, retmax: int,
                        retries: int = 4) -> list[dict]:
    """Fetch one page of records directly from the history server.

    Unlike esearch, efetch from history has no 9,999-record limit, so this pages
    through arbitrarily large result sets via retstart/retmax.

    NCBI occasionally returns a 200 whose body is not well-formed XML -- either
    truncated mid-document or carrying an illegal character. Recover rather than
    crash a multi-hour pull: try the raw body, then a sanitized copy (fixes bad
    chars), then re-fetch (fixes truncation). If a batch stays malformed after
    `retries`, skip it with a loud warning (bounded, logged loss of <= retmax
    records) so the pull can finish.
    """
    params = {
        "db": "pubmed",
        "WebEnv": webenv,
        "query_key": query_key,
        "retstart": retstart,
        "retmax": retmax,
        "retmode": "xml",
        "rettype": "abstract",
    }
    if get_api_key():
        params["api_key"] = get_api_key()

    last_err = None
    for attempt in range(retries):
        content = http_get("efetch.fcgi", params, timeout=120).content
        try:
            return parse_efetch_xml(content)
        except ET.ParseError as e:
            last_err = e
        try:
            return parse_efetch_xml(sanitize_xml(content))
        except ET.ParseError as e:
            last_err = e  # likely truncated; re-fetch below
        time.sleep(2 ** attempt)

    print(
        f"  WARNING: efetch batch at retstart={retstart} stayed malformed after "
        f"{retries} attempts ({last_err}); skipping up to {retmax} records."
    )
    return []


def _fetch_small_slice(webenv: str, query_key: str, count: int, out_handle) -> int:
    """Page a single slice known to be within the retstart cap; return #written."""
    n = 0
    cap = min(count, MAX_SLICE)
    for retstart in range(0, cap, BATCH_SIZE):
        records = efetch_from_history(webenv, query_key, retstart, BATCH_SIZE)
        for rec in records:
            out_handle.write(json.dumps(rec) + "\n")
            n += 1
        rate_limit_sleep()
    return n


def fetch_date_range(query: str, day0: datetime.date, day1: datetime.date,
                     out_handle, datetype: str = "edat") -> int:
    """Recursively fetch [day0, day1] (inclusive), bisecting the date range until
    each slice is under NCBI's retrieval cap. Writes records to out_handle;
    returns the number written."""
    start_s, end_s = day0.isoformat(), day1.isoformat()
    count, webenv, query_key = esearch_history(query, start_s, end_s, datetype)
    if count == 0:
        return 0
    if count <= MAX_SLICE:
        return _fetch_small_slice(webenv, query_key, count, out_handle)
    if day0 == day1:
        # Cannot split a single day further; take what the cap allows and warn.
        print(
            f"  WARNING: {start_s} has {count} records, above NCBI's ~10,000 "
            f"retrieval cap; only the records reachable under the cap will be "
            f"fetched for this day (the rest are unavoidably dropped)."
        )
        return _fetch_small_slice(webenv, query_key, MAX_SLICE, out_handle)
    mid = day0 + datetime.timedelta(days=(day1 - day0).days // 2)
    return (
        fetch_date_range(query, day0, mid, out_handle, datetype)
        + fetch_date_range(query, mid + datetime.timedelta(days=1), day1, out_handle, datetype)
    )


def month_windows(start: datetime.date, end: datetime.date):
    """Yield (label 'YYYY-MM', win_start, win_end) calendar-month windows covering
    [start, end], with the first and last clamped to start/end."""
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        first = datetime.date(y, m, 1)
        # last day of month m: day before the first of the next month
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        last = datetime.date(ny, nm, 1) - datetime.timedelta(days=1)
        yield f"{y:04d}-{m:02d}", max(first, start), min(last, end)
        y, m = ny, nm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="PubMed search term, e.g. 'cancer AND treatment'")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument(
        "--datetype", default="edat", choices=["edat", "pdat", "mdat", "crdt"],
        help="date field the --start/--end window filters on. Default 'edat' "
             "(Entrez index date) is roughly uniform; 'pdat' (publication date) "
             "piles year-only records onto Jan 1, overflowing the retrieval cap.",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime.date.fromisoformat(args.start)
    end = datetime.date.fromisoformat(args.end)
    if start > end:
        raise SystemExit(f"--start ({args.start}) is after --end ({args.end})")

    # Resumable per-month shards. A shard file's existence means that month is
    # complete; a meta file guards against reusing shards for a different query.
    parts_dir = Path(str(out_path) + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    meta_path = parts_dir / "_meta.json"
    meta = {"query": args.query, "start": args.start, "end": args.end, "datetype": args.datetype}
    if meta_path.exists():
        prev = json.loads(meta_path.read_text())
        if prev != meta:
            raise SystemExit(
                f"{parts_dir} holds shards for a different pull ({prev}). "
                f"Delete it or choose a new --out before running this query."
            )
    else:
        meta_path.write_text(json.dumps(meta))

    windows = list(month_windows(start, end))
    for label, win_start, win_end in tqdm(windows, desc="months"):
        shard = parts_dir / f"{label}.jsonl"
        if shard.exists():
            continue  # already fetched (resume)
        tmp = parts_dir / f"{label}.jsonl.tmp"
        with tmp.open("w") as f:
            fetch_date_range(args.query, win_start, win_end, f, args.datetype)
        os.replace(tmp, shard)  # atomic: partial months never become shards

    # Concatenate month shards (date order) into the single output file.
    n_written = 0
    with out_path.open("w") as out_f:
        for label, _, _ in windows:
            shard = parts_dir / f"{label}.jsonl"
            with shard.open() as sf:
                for line in sf:
                    out_f.write(line)
                    n_written += 1

    print(f"Wrote {n_written} records to {out_path}")


if __name__ == "__main__":
    main()
