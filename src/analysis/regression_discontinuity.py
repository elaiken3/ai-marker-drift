"""
Test whether a marker's monthly frequency shows a statistically significant
change in trend (slope) or level (jump) at a given breakpoint -- e.g.
Nov 30 2022 (ChatGPT release) or some later "AI-detection-anxiety" date you want
to test as a second breakpoint.

Method: segmented (piecewise-linear) OLS regression of
    freq ~ t + is_post + t_since_break*is_post
against a single-line OLS of freq ~ t, compared via an F-test (Chow test).
A significant F-test means allowing the trend to bend at the breakpoint
explains meaningfully more variance than a single straight line.

This deliberately does NOT use a naive pre/post t-test on the two means --
monthly publication counts are autocorrelated, so a t-test on raw means
overstates significance. Segmented regression + F-test is the standard
approach in interrupted-time-series designs.

Usage:
    python src/analysis/regression_discontinuity.py \
        --in data/processed/pubmed_cancer_monthly.csv \
        --marker em_dash \
        --breakpoint 2022-11-30 \
        --plot data/processed/em_dash_break.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.api import anova_lm


def build_design(df: pd.DataFrame, breakpoint_date: pd.Timestamp):
    df = df.copy()
    df["date"] = pd.to_datetime(df["month"] + "-01")
    df = df.sort_values("date").reset_index(drop=True)
    df["t"] = (df["date"] - df["date"].min()).dt.days / 30.44  # months since start
    df["is_post"] = (df["date"] >= breakpoint_date).astype(int)
    df["t_since_break"] = np.where(
        df["is_post"] == 1,
        (df["date"] - breakpoint_date).dt.days / 30.44,
        0.0,
    )
    return df


def run_chow_test(df: pd.DataFrame, y_col: str, hac_lags: int = 6):
    """Segmented regression with a structural-break test.

    Coefficient inference uses Newey-West (HAC) standard errors because
    monthly frequencies are autocorrelated -- plain OLS errors would
    overstate significance. The break test itself is a Wald test on the
    break terms (is_post, t_since_break) computed under the HAC covariance,
    which is the autocorrelation-robust analogue of the classic Chow F-test.
    The plain-OLS anova F is also reported for reference; expect it to be
    more optimistic (smaller p) than the robust Wald test.
    """
    y = df[y_col]

    X_restricted = sm.add_constant(df[["t"]])
    restricted = sm.OLS(y, X_restricted).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags}
    )

    X_full = sm.add_constant(df[["t", "is_post", "t_since_break"]])
    full = sm.OLS(y, X_full).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags}
    )

    # Robust (HAC) Wald test: are the break terms jointly zero?
    wald = full.wald_test("(is_post = 0), (t_since_break = 0)", scalar=True)
    robust_f = float(wald.statistic)
    robust_p = float(wald.pvalue)

    # Classic (non-robust) Chow F for reference
    restricted_ols = sm.OLS(y, X_restricted).fit()
    full_ols = sm.OLS(y, X_full).fit()
    anova_result = anova_lm(restricted_ols, full_ols)
    naive_f = anova_result["F"].iloc[1]
    naive_p = anova_result["Pr(>F)"].iloc[1]

    return restricted, full, robust_f, robust_p, naive_f, naive_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--marker", required=True, help="marker name, e.g. em_dash (matches *_per_1k_words column)")
    ap.add_argument("--breakpoint", required=True, help="YYYY-MM-DD, e.g. 2022-11-30")
    ap.add_argument("--plot", default=None, help="optional path to save a PNG plot")
    ap.add_argument(
        "--placebos",
        default="2019-11-30,2021-11-30",
        help="comma-separated fake breakpoint dates for placebo tests; pass '' to skip",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.infile)
    y_col = f"{args.marker}_per_1k_words"
    if y_col not in df.columns:
        raise SystemExit(f"'{y_col}' not found. Available columns: {list(df.columns)}")

    breakpoint_date = pd.Timestamp(args.breakpoint)
    df_design = build_design(df, breakpoint_date)

    restricted, full, robust_f, robust_p, naive_f, naive_p = run_chow_test(df_design, y_col)

    print("=== Full model (level jump + slope change at breakpoint, HAC errors) ===")
    print(full.summary().tables[1])
    print(f"\nRobust (HAC/Newey-West) break test: F = {robust_f:.3f}, p = {robust_p:.4f}")
    print(f"Naive Chow F (no autocorrelation correction, for reference): F = {naive_f:.3f}, p = {naive_p:.4f}")
    if robust_p < 0.05:
        print("-> Structural break at the specified date is significant under HAC errors (p < .05).")
    else:
        print("-> No significant break under HAC errors at this date/marker/corpus.")

    # Placebo tests: same test at dates where no AI-driven break should exist.
    # Crucially, placebos are run on PRE-BREAK DATA ONLY -- if the real break
    # were left in the placebo's post-period, the segmented model would
    # partially fit it and the placebo would fire spuriously. If a placebo is
    # still significant on pre-break data alone, the series has unrelated
    # structure and the main result shouldn't be trusted at face value.
    if args.placebos:
        print("\n=== Placebo breakpoint tests (pre-break data only) ===")
        pre_df = df[pd.to_datetime(df["month"] + "-01") < breakpoint_date]
        for placebo in args.placebos.split(","):
            placebo = placebo.strip()
            placebo_date = pd.Timestamp(placebo)
            pdf = build_design(pre_df, placebo_date)
            n_pre = (pdf["is_post"] == 0).sum()
            n_post = (pdf["is_post"] == 1).sum()
            if n_pre < 6 or n_post < 6:
                print(f"  {placebo}: skipped (only {n_pre} pre / {n_post} post months -- too few for a stable fit)")
                continue
            *_, p_rf, p_rp, _, _ = run_chow_test(pdf, y_col)
            flag = "  <-- WARNING: placebo significant" if p_rp < 0.05 else ""
            print(f"  {placebo}: F = {p_rf:.3f}, p = {p_rp:.4f}{flag}")

    if args.plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(df_design["date"], df_design[y_col], s=12, alpha=0.5, label="observed")
        ax.plot(df_design["date"], full.fittedvalues, color="red", label="segmented fit")
        ax.axvline(breakpoint_date, color="gray", linestyle="--", label="breakpoint")
        ax.set_ylabel(f"{args.marker} per 1,000 words")
        ax.set_title(f"{args.marker}: monthly frequency with structural break test")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
