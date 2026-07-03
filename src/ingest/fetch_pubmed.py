"""
Pull PubMed abstracts + publication dates for a date range and search query.

Uses NCBI E-utilities (esearch -> efetch), which are free and don't require
an API key, but are rate-limited to 3 req/sec without a key and 10 req/sec
with one (set NCBI_API_KEY env var to use a key).

Output: one JSON object per line (jsonl) with fields:
    {"id": pmid, "date": "YYYY-MM-DD" or "YYYY-MM", "text": abstract_text}

Usage:
    python src/ingest/fetch_pubmed.py \
        --query "cancer AND treatment" \
        --start 2020-01-01 --end 2026-06-01 \
        --out data/raw/pubmed_cancer.jsonl
"""
import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from tqdm import tqdm

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BATCH_SIZE = 200  # pmids per efetch call


def get_api_key() -> str | None:
    return os.environ.get("NCBI_API_KEY")


def rate_limit_sleep():
    time.sleep(0.11 if get_api_key() else 0.35)


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
        r = requests.get(f"{EUTILS_BASE}/{endpoint}", params=params, timeout=30)
        r.raise_for_status()
        try:
            return json.loads(r.text, strict=False)
        except json.JSONDecodeError as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(
        f"E-utilities {endpoint} returned unparseable JSON after {retries} attempts: {last_err}"
    )


def esearch_all_ids(query: str, start: str, end: str) -> list[str]:
    """Use esearch with history (usehistory=y) to page through all matching PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "datetype": "pdat",
        "mindate": start,
        "maxdate": end,
        "retmode": "json",
        "retmax": 0,
        "usehistory": "y",
    }
    if get_api_key():
        params["api_key"] = get_api_key()

    data = eutils_get_json("esearch.fcgi", params)["esearchresult"]
    total = int(data["count"])
    webenv = data["webenv"]
    query_key = data["querykey"]

    ids = []
    retmax = 10000
    for retstart in tqdm(range(0, total, retmax), desc="esearch paging"):
        params = {
            "db": "pubmed",
            "query_key": query_key,
            "WebEnv": webenv,
            "retstart": retstart,
            "retmax": retmax,
            "retmode": "json",
        }
        if get_api_key():
            params["api_key"] = get_api_key()
        ids.extend(eutils_get_json("esearch.fcgi", params)["esearchresult"]["idlist"])
        rate_limit_sleep()

    return ids


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


def efetch_batch(pmids: list[str]) -> list[dict]:
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    if get_api_key():
        params["api_key"] = get_api_key()

    r = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    root = ET.fromstring(r.content)

    records = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID")
        date = parse_pubdate(article)
        abstract_parts = article.findall(".//Abstract/AbstractText")
        text = " ".join(a.text for a in abstract_parts if a.text)
        if text and date:
            records.append({"id": pmid, "date": date, "text": text})
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="PubMed search term, e.g. 'cancer AND treatment'")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", required=True, help="output jsonl path")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ids = esearch_all_ids(args.query, args.start, args.end)
    print(f"Found {len(ids)} PMIDs for query.")

    with out_path.open("w") as f:
        for i in tqdm(range(0, len(ids), BATCH_SIZE), desc="efetch batches"):
            batch = ids[i:i + BATCH_SIZE]
            try:
                records = efetch_batch(batch)
            except requests.HTTPError as e:
                print(f"batch {i} failed: {e}, retrying once after backoff")
                time.sleep(2)
                records = efetch_batch(batch)
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            rate_limit_sleep()

    print(f"Wrote records to {out_path}")


if __name__ == "__main__":
    main()
