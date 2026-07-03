"""
Pull Guardian article bodies + publication dates for a date range and query.

This is the "journalism" leg of the theory (less-edited-than-PubMed prose,
different topic mix). It uses the Guardian Open Platform Content API, which is
free but requires an API key -- register at
https://open-platform.theguardian.com/access/ and set GUARDIAN_API_KEY.
The free "developer" tier is roughly 1 request/second and 500 requests/day,
so a wide date range may need to run across multiple days or a higher tier.

Output matches the shared ingest contract -- one JSON object per line (jsonl):
    {"id": guardian_id, "date": "YYYY-MM-DD", "text": article_body_text}
so it drops straight into src/analysis/extract_features.py with no changes.

Usage:
    python src/ingest/fetch_guardian.py \
        --query "climate" \
        --start 2020-01-01 --end 2026-06-01 \
        --out data/raw/guardian_climate.jsonl
"""
import argparse
import json
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

SEARCH_URL = "https://content.guardianapis.com/search"
PAGE_SIZE = 200  # Guardian max page-size for content search
RATE_LIMIT_SECONDS = 1.1  # free tier ~1 req/sec; stay just under


def get_api_key() -> str:
    key = os.environ.get("GUARDIAN_API_KEY")
    if not key:
        raise SystemExit(
            "GUARDIAN_API_KEY not set. Get a free key at "
            "https://open-platform.theguardian.com/access/ and export it."
        )
    return key


def parse_item(item: dict) -> dict | None:
    """Map a Guardian search result to the {id, date, text} contract.

    Pure function (no network) so it can be unit-tested offline. Returns None
    for items missing a usable body or date so callers can skip them.
    """
    body = (item.get("fields") or {}).get("bodyText", "")
    published = item.get("webPublicationDate", "")
    if not body or not published:
        return None
    return {
        "id": item.get("id"),
        "date": published[:10],  # ISO 8601 'YYYY-MM-DDThh:mm:ssZ' -> 'YYYY-MM-DD'
        "text": body,
    }


def fetch_page(query: str, start: str, end: str, page: int, api_key: str) -> dict:
    params = {
        "q": query,
        "from-date": start,
        "to-date": end,
        "page": page,
        "page-size": PAGE_SIZE,
        "order-by": "oldest",
        "show-fields": "bodyText",
        "api-key": api_key,
    }
    r = requests.get(SEARCH_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json()["response"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="Guardian search term, e.g. 'climate'")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", required=True, help="output jsonl path")
    args = ap.parse_args()

    api_key = get_api_key()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first = fetch_page(args.query, args.start, args.end, page=1, api_key=api_key)
    total_pages = first.get("pages", 1)
    print(f"Found {first.get('total', 0)} articles across {total_pages} pages.")

    n_written = 0
    with out_path.open("w") as f:
        for page in tqdm(range(1, total_pages + 1), desc="guardian pages"):
            if page == 1:
                resp = first
            else:
                try:
                    resp = fetch_page(args.query, args.start, args.end, page, api_key)
                except requests.HTTPError as e:
                    print(f"page {page} failed: {e}, retrying once after backoff")
                    time.sleep(2)
                    resp = fetch_page(args.query, args.start, args.end, page, api_key)
            for item in resp.get("results", []):
                rec = parse_item(item)
                if rec:
                    f.write(json.dumps(rec) + "\n")
                    n_written += 1
            time.sleep(RATE_LIMIT_SECONDS)

    print(f"Wrote {n_written} records to {out_path}")


if __name__ == "__main__":
    main()
