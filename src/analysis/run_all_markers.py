"""
Run the structural-break test across ALL markers in the monthly frequency
table at once, apply Benjamini-Hochberg false-discovery-rate correction,
and print a ranked summary table.

Why this exists: testing ~20+ markers at p < .05 individually guarantees
false positives by chance (~1 expected per 20 tests). BH correction controls
the false discovery rate across the whole marker set. Negative-control
markers (from config controls: list) are labeled in the output -- if a
control survives BH correction, treat the whole run as suspect (corpus
drift, not an AI effect).

Usage:
    python src/analysis/run_all_markers.py \
        --in data/processed/pubmed_cancer_monthly.csv \
        --breakpoint 2022-11-30 \
        --markers config/markers.yaml \
        --out data/processed/break_test_summary.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml
from statsmodels.stats.multitest import multipletests

from regression_discontinuity import build_design, run_chow_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--breakpoint", required=True, help="YYYY-MM-DD")
    ap.add_argument("--markers", default="config/markers.yaml")
    ap.add_argument("--out", default=None, help="optional CSV path for the summary table")
    ap.add_argument("--alpha", type=float, default=0.05, help="FDR level for BH correction")
    args = ap.parse_args()

    with open(args.markers) as f:
        cfg = yaml.safe_load(f)
    controls = set(cfg.get("controls", []))
    all_markers = (
        list(cfg.get("punct", []))
        + list(cfg.get("phrases", []))
        + list(cfg.get("controls", []))
    )

    df = pd.read_csv(args.infile)
    breakpoint_date = pd.Timestamp(args.breakpoint)

    results = []
    for marker in all_markers:
        y_col = f"{marker}_per_1k_words"
        if y_col not in df.columns:
            print(f"skipping '{marker}' -- column {y_col} not in input")
            continue
        if df[y_col].nunique() <= 1:
            print(f"skipping '{marker}' -- series is constant (likely never occurs in corpus)")
            continue
        design = build_design(df, breakpoint_date)
        _, full, robust_f, robust_p, _, _ = run_chow_test(design, y_col)
        results.append({
            "marker": marker,
            "is_control": marker in controls,
            "level_jump": full.params.get("is_post", float("nan")),
            "slope_change": full.params.get("t_since_break", float("nan")),
            "robust_F": robust_f,
            "p_raw": robust_p,
        })

    res = pd.DataFrame(results)
    if res.empty:
        raise SystemExit("No markers matched columns in the input file.")

    reject, p_adj, _, _ = multipletests(res["p_raw"], alpha=args.alpha, method="fdr_bh")
    res["p_bh"] = p_adj
    res["significant_after_bh"] = reject
    res = res.sort_values("p_bh").reset_index(drop=True)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(f"\nBreakpoint: {args.breakpoint} | BH FDR alpha = {args.alpha}")
    print(res.to_string(index=False))

    sig_controls = res[(res["is_control"]) & (res["significant_after_bh"])]
    if not sig_controls.empty:
        print(
            "\nWARNING: negative control(s) significant after BH correction: "
            f"{', '.join(sig_controls['marker'])}. "
            "This suggests corpus-wide drift rather than an AI-specific effect. "
            "Do not interpret marker results as AI-driven without addressing this."
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out_path, index=False)
        print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
