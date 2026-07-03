"""
Read a jsonl corpus (output of an ingest script; one doc per line with
{"id", "date", "text"}), compute per-1,000-word frequency for each marker
in config/markers.yaml, and aggregate to a monthly time series.

As well as the monthly mean-based frequency ({marker}_per_1k_words), this
emits monthly quantiles of the *per-document* frequency distribution
({marker}_pQQ_per_1k_words, e.g. em_dash_p90_per_1k_words). Those tail
columns exist for hypothesis H2 (avoidance): if writers start trimming a
marker once it's a recognized "AI tell", the right tail of the per-doc
distribution shrinks even when the monthly mean barely moves. Because the
quantile columns follow the same {marker}_per_1k_words naming convention,
they drop straight into regression_discontinuity.py via --marker em_dash_p90.

Usage:
    python src/analysis/extract_features.py \
        --in data/raw/pubmed_cancer.jsonl \
        --out data/processed/pubmed_cancer_monthly.csv \
        --markers config/markers.yaml \
        --quantiles 0.5,0.75,0.9,0.95 \
        --per-doc-out data/processed/pubmed_cancer_perdoc.csv   # optional
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
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


def month_key(date_str: str) -> str | None:
    # dates come in as YYYY, YYYY-MM, or YYYY-MM-DD. Year-only dates have no
    # month, so they can't be placed on a monthly axis -- return None so the
    # caller drops them instead of silently piling them into January (PubMed
    # emits ~20k year-only records per year; binning them to Jan created a huge
    # spurious annual spike that broke every marker AND every control).
    parts = date_str.split("-")
    if len(parts) >= 2 and parts[1]:
        return f"{parts[0]}-{parts[1]}"
    return None


def quantile_suffix(q: float) -> str:
    """0.9 -> 'p90', 0.5 -> 'p50', 0.975 -> 'p98' (rounded to integer pct)."""
    return f"p{int(round(q * 100))}"


def parse_quantiles(spec: str) -> list[float]:
    qs = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        q = float(tok)
        if not 0.0 <= q <= 1.0:
            raise SystemExit(f"quantile {q} out of range [0, 1]")
        qs.append(q)
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", dest="outfile", required=True)
    ap.add_argument("--markers", default="config/markers.yaml")
    ap.add_argument(
        "--quantiles",
        default="0.5,0.75,0.9,0.95",
        help="comma-separated per-doc frequency quantiles to emit per month "
             "(as {marker}_pQQ_per_1k_words columns); pass '' to skip",
    )
    ap.add_argument(
        "--per-doc-out",
        dest="perdoc_out",
        default=None,
        help="optional path to also dump a long per-document table "
             "(id, month, {marker}_per_1k_words...) for ad-hoc analysis",
    )
    ap.add_argument(
        "--min-docs",
        type=int,
        default=0,
        help="drop months with fewer than this many documents (removes sparse "
             "edge months that add noise to the regression); default 0 = keep all",
    )
    args = ap.parse_args()

    cfg = load_markers(args.markers)
    marker_names = (
        list(cfg.get("punct", []))
        + list(cfg.get("phrases", []))
        + list(cfg.get("controls", []))
    )
    phrase_patterns = compile_phrase_patterns(cfg)
    quantiles = parse_quantiles(args.quantiles)

    # monthly accumulators
    monthly_marker_counts = defaultdict(lambda: defaultdict(int))
    monthly_word_counts = defaultdict(int)
    monthly_doc_counts = defaultdict(int)
    # per-doc per-1k frequencies, kept per month for quantile computation
    monthly_perdoc_freq = defaultdict(lambda: defaultdict(list))
    perdoc_rows = []  # only populated when --per-doc-out is set

    n_docs = 0
    n_year_only = 0  # dropped: no month, can't be placed on a monthly axis
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

            mk = month_key(date)
            if mk is None:
                n_year_only += 1
                continue

            wc = word_count(text)
            if wc == 0:
                continue

            counts = count_markers(text, cfg, phrase_patterns)
            perdoc_row = {"id": rec.get("id"), "month": mk}
            for name, c in counts.items():
                monthly_marker_counts[mk][name] += c
                doc_freq = (c / wc) * 1000
                monthly_perdoc_freq[mk][name].append(doc_freq)
                if args.perdoc_out:
                    perdoc_row[f"{name}_per_1k_words"] = doc_freq
            if args.perdoc_out:
                perdoc_rows.append(perdoc_row)
            monthly_word_counts[mk] += wc
            monthly_doc_counts[mk] += 1
            n_docs += 1

    rows = []
    n_sparse_months = 0
    for mk in sorted(monthly_word_counts.keys()):
        if monthly_doc_counts[mk] < args.min_docs:
            n_sparse_months += 1
            continue
        row = {
            "month": mk,
            "n_docs": monthly_doc_counts[mk],
            "n_words": monthly_word_counts[mk],
        }
        for name in marker_names:
            raw = monthly_marker_counts[mk].get(name, 0)
            row[f"{name}_per_1k_words"] = (raw / monthly_word_counts[mk]) * 1000
            row[f"{name}_raw"] = raw
            doc_freqs = monthly_perdoc_freq[mk].get(name, [])
            for q in quantiles:
                col = f"{name}_{quantile_suffix(q)}_per_1k_words"
                row[col] = float(np.quantile(doc_freqs, q)) if doc_freqs else 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Processed {n_docs} docs across {len(rows)} months -> {out_path}")
    if n_year_only:
        print(f"Dropped {n_year_only} year-only records (no month; can't bin to a monthly axis)")
    if n_sparse_months:
        print(f"Dropped {n_sparse_months} month(s) with < {args.min_docs} docs (--min-docs)")

    if args.perdoc_out:
        perdoc_path = Path(args.perdoc_out)
        perdoc_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(perdoc_rows).to_csv(perdoc_path, index=False)
        print(f"Wrote {len(perdoc_rows)} per-doc rows -> {perdoc_path}")


if __name__ == "__main__":
    main()
