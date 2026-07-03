"""
Read a jsonl corpus (output of an ingest script; one doc per line with
{"id", "date", "text"}), compute per-1,000-word frequency for each marker
in config/markers.yaml, and aggregate to a monthly time series.

Usage:
    python src/analysis/extract_features.py \
        --in data/raw/pubmed_cancer.jsonl \
        --out data/processed/pubmed_cancer_monthly.csv \
        --markers config/markers.yaml
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

PUNCT_PATTERNS = {
    "em_dash": re.compile(r"\u2014|--"),  # allow "--" as a common ASCII substitute
    "semicolon": re.compile(r";"),
}


def load_markers(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def compile_phrase_patterns(cfg: dict) -> dict:
    """Word-boundary regex per phrase, so 'boast' doesn't match inside
    other words. Multi-word phrases get boundaries at both ends."""
    patterns = {}
    for phrase in list(cfg.get("phrases", [])) + list(cfg.get("controls", [])):
        escaped = re.escape(phrase.lower())
        patterns[phrase] = re.compile(rf"\b{escaped}\b")
    return patterns


def count_markers(text: str, cfg: dict, phrase_patterns: dict) -> dict:
    lower = text.lower()
    counts = {}
    for name in cfg.get("punct", []):
        pattern = PUNCT_PATTERNS.get(name)
        if pattern:
            counts[name] = len(pattern.findall(text))
    for phrase, pattern in phrase_patterns.items():
        counts[phrase] = len(pattern.findall(lower))
    return counts


def month_key(date_str: str) -> str:
    # dates come in as YYYY, YYYY-MM, or YYYY-MM-DD -- normalize to YYYY-MM
    parts = date_str.split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return f"{parts[0]}-01"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", dest="outfile", required=True)
    ap.add_argument("--markers", default="config/markers.yaml")
    args = ap.parse_args()

    cfg = load_markers(args.markers)
    marker_names = (
        list(cfg.get("punct", []))
        + list(cfg.get("phrases", []))
        + list(cfg.get("controls", []))
    )
    phrase_patterns = compile_phrase_patterns(cfg)

    # monthly accumulators
    monthly_marker_counts = defaultdict(lambda: defaultdict(int))
    monthly_word_counts = defaultdict(int)
    monthly_doc_counts = defaultdict(int)

    n_docs = 0
    with open(args.infile) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("text", "")
            date = rec.get("date")
            if not text or not date:
                continue

            wc = word_count(text)
            if wc == 0:
                continue

            mk = month_key(date)
            counts = count_markers(text, cfg, phrase_patterns)
            for name, c in counts.items():
                monthly_marker_counts[mk][name] += c
            monthly_word_counts[mk] += wc
            monthly_doc_counts[mk] += 1
            n_docs += 1

    rows = []
    for mk in sorted(monthly_word_counts.keys()):
        row = {
            "month": mk,
            "n_docs": monthly_doc_counts[mk],
            "n_words": monthly_word_counts[mk],
        }
        for name in marker_names:
            raw = monthly_marker_counts[mk].get(name, 0)
            row[f"{name}_per_1k_words"] = (raw / monthly_word_counts[mk]) * 1000
            row[f"{name}_raw"] = raw
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Processed {n_docs} docs across {len(rows)} months -> {out_path}")


if __name__ == "__main__":
    main()
