"""
Pull yearly frequency data from the Google Books Ngram public JSON endpoint.
No auth required. Only covers up through the corpus's last indexed year
(~2022 as of the 2020 corpus release) -- use this as a long-run PRE-PERIOD
baseline for your marker list, not as evidence about post-ChatGPT drift.

Note: the endpoint is picky about punctuation tokens. Em dash and semicolon
are NOT reliably queryable as standalone ngram tokens (Google's tokenizer
mostly strips punctuation), so this script focuses on the lexical/phrase
markers. Treat em-dash trend validation as a job for your own corpora
(PubMed/NYT/arXiv), not Ngram.

Usage:
    python src/ingest/fetch_ngram.py --terms delve,underscore,crucial \
        --start 1950 --end 2022 --out data/raw/ngram_baseline.csv
"""
import argparse
import csv
from pathlib import Path

import requests

NGRAM_URL = "https://books.google.com/ngrams/json"


def fetch_term(term: str, start: int, end: int, corpus: str = "en-2019") -> list[dict]:
    params = {
        "content": term,
        "year_start": start,
        "year_end": end,
        "corpus": corpus,
        "smoothing": 0,
    }
    r = requests.get(NGRAM_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data:
        print(f"warning: no data returned for '{term}' (may need multi-word phrasing tweak)")
        return []

    series = data[0]
    years = list(range(start, end + 1))
    values = series.get("timeseries", [])
    return [{"term": term, "year": y, "freq": v} for y, v in zip(years, values)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", required=True, help="comma-separated list of terms/phrases")
    ap.add_argument("--start", type=int, default=1950)
    ap.add_argument("--end", type=int, default=2022)
    ap.add_argument("--corpus", default="en-2019")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for term in args.terms.split(","):
        term = term.strip()
        rows.extend(fetch_term(term, args.start, args.end, args.corpus))

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["term", "year", "freq"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
