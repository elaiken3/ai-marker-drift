"""
Automated structural-break detection on a marker's monthly frequency series.

Where regression_discontinuity.py *tests a break at a date you specify*, this
script *finds* breakpoints without being told where to look, then reports how
close the nearest detected break is to a reference date (default: the ChatGPT
release, 2022-11-30). Finding a break near Nov 2022 without pre-specifying the
date is stronger evidence than confirming a hypothesized one -- it removes the
"you only found it because you looked there" objection.

Uses the `ruptures` library (Bai-Perron-style changepoint detection):
  - --n-breaks N   : exact number of breaks, via ruptures.Dynp (dynamic prog).
  - --penalty P    : data-driven number of breaks, via ruptures.Pelt.
Default is a single break (--n-breaks 1). The default cost model is "l2"
(mean/level shift), which matches the level jump the segmented regression tests;
"rbf" is available for more general distributional change.

Usage:
    python src/analysis/detect_breaks.py \
        --in data/processed/pubmed_cancer_monthly.csv \
        --marker em_dash \
        --n-breaks 1 \
        --reference 2022-11-30
"""
import argparse

import numpy as np
import pandas as pd
import ruptures as rpt


def months_between(a: pd.Timestamp, b: pd.Timestamp) -> float:
    """Signed distance in (approx) months from a to b."""
    return (b - a).days / 30.44


def detect(signal: np.ndarray, model: str, n_breaks: int | None, penalty: float | None):
    """Return sorted internal breakpoint indices (excluding the trailing n)."""
    sig = signal.reshape(-1, 1)
    # jump=1 so a break can land on ANY month -- ruptures defaults jump=5, which
    # only considers breakpoints at multiples of 5 and would snap a true break to
    # the nearest grid point. min_size=2 keeps each segment fittable.
    if penalty is not None:
        algo = rpt.Pelt(model=model, min_size=2, jump=1).fit(sig)
        bkps = algo.predict(pen=penalty)
    else:
        algo = rpt.Dynp(model=model, min_size=2, jump=1).fit(sig)
        bkps = algo.predict(n_bkps=n_breaks)
    # ruptures returns end-indices including len(signal); drop that sentinel.
    return [b for b in bkps if b < len(signal)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--marker", required=True,
                    help="marker name, e.g. em_dash (matches *_per_1k_words column)")
    ap.add_argument("--reference", default="2022-11-30",
                    help="reference date (YYYY-MM-DD) to measure detected breaks against")
    ap.add_argument("--model", default="l2", choices=["l1", "l2", "rbf"],
                    help="ruptures cost model (l2 = level shift)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--n-breaks", dest="n_breaks", type=int, default=None,
                      help="exact number of breakpoints to find (ruptures.Dynp)")
    mode.add_argument("--penalty", type=float, default=None,
                      help="penalty for data-driven break count (ruptures.Pelt)")
    args = ap.parse_args()

    if args.n_breaks is None and args.penalty is None:
        args.n_breaks = 1  # default: assume a single break, let the data place it

    df = pd.read_csv(args.infile)
    y_col = f"{args.marker}_per_1k_words"
    if y_col not in df.columns:
        raise SystemExit(f"'{y_col}' not found. Available columns: {list(df.columns)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["month"] + "-01")
    df = df.sort_values("date").reset_index(drop=True)
    signal = df[y_col].to_numpy(dtype=float)

    if len(signal) < 4:
        raise SystemExit(f"Only {len(signal)} months of data -- too few to detect a break.")

    bkp_indices = detect(signal, args.model, args.n_breaks, args.penalty)
    reference = pd.Timestamp(args.reference)

    print(f"Series: {y_col} | {len(signal)} months "
          f"({df['date'].min():%Y-%m} to {df['date'].max():%Y-%m})")
    if args.penalty is not None:
        print(f"Method: Pelt (model={args.model}, penalty={args.penalty})")
    else:
        print(f"Method: Dynp (model={args.model}, n_breaks={args.n_breaks})")

    if not bkp_indices:
        print("No breakpoints detected.")
        return

    print(f"\nReference date: {reference:%Y-%m-%d}")
    print("Detected breakpoints (the change begins at this month):")
    nearest = None
    for idx in bkp_indices:
        bkp_date = df["date"].iloc[idx]
        dist = months_between(reference, bkp_date)
        print(f"  {bkp_date:%Y-%m}  ({dist:+.1f} months from reference)")
        if nearest is None or abs(dist) < abs(nearest[1]):
            nearest = (bkp_date, dist)

    print(f"\nNearest break to {reference:%Y-%m-%d}: "
          f"{nearest[0]:%Y-%m} ({nearest[1]:+.1f} months away)")


if __name__ == "__main__":
    main()
