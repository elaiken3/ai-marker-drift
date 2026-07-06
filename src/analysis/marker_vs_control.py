"""
Drift-adjusted marker test (difference-in-differences vs negative controls).

run_all_markers.py asks "did this marker break at the date?" -- but when the
whole corpus drifts, neutral control words break too, so a bare "significant"
break is not evidence of an AI-specific effect (with ~10^5 docs/month every
series is statistically significant anyway). This script asks the sharper
question: *did a marker break MORE than the neutral controls did?*

Method:
- For each series, fit the same segmented model as regression_discontinuity.py
  and take the level jump at the break (the `is_post` coefficient).
- Standardize it by that series' own pre-break standard deviation, giving a
  scale-free effect size (jump measured in pre-break SDs) that is comparable
  across markers with very different base rates.
- The negative controls' standardized effects form a "drift band" -- the size
  of break a word with no AI association shows at this date, i.e. the corpus
  drift baseline.
- Each marker is then scored against that band, both parametrically
  (z = SDs above the control-band mean) and non-parametrically (how many of the
  controls its |effect| beats). A marker only counts as a candidate AI signal
  if it stands clearly OUTSIDE the control band -- otherwise it is just riding
  the same corpus drift the controls show.

Sign is preserved: a positive standout is consistent with H1 (adoption), a
negative one with H2 (avoidance).

Usage:
    python src/analysis/marker_vs_control.py \
        --in data/processed/pubmed_monthly.csv \
        --breakpoint 2022-11-30 \
        --out data/processed/marker_vs_control.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from regression_discontinuity import build_design, run_chow_test


def series_effect(design: pd.DataFrame, y_col: str):
    """Return (level_jump, pre_break_mean, pre_break_sd) for one series."""
    _, full, *_ = run_chow_test(design, y_col)
    level_jump = float(full.params.get("is_post", float("nan")))
    pre = design.loc[design["is_post"] == 0, y_col]
    return level_jump, float(pre.mean()), float(pre.std(ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--breakpoint", required=True, help="YYYY-MM-DD")
    ap.add_argument("--markers", default="config/markers.yaml")
    ap.add_argument("--out", default=None, help="optional CSV path for the table")
    ap.add_argument("--z-threshold", type=float, default=2.0,
                    help="|z| above the control band to call a marker a standout")
    ap.add_argument("--min-count", type=int, default=1000,
                    help="minimum total occurrences for a series to be analyzed; "
                         "rarer markers/controls are set aside as unreliable "
                         "(standardizing a near-zero series inflates tiny blips)")
    args = ap.parse_args()

    with open(args.markers) as f:
        cfg = yaml.safe_load(f)
    control_names = list(cfg.get("controls", []))
    marker_names = list(cfg.get("punct", [])) + list(cfg.get("phrases", []))
    control_set = set(control_names)

    df = pd.read_csv(args.infile)
    design = build_design(df, pd.Timestamp(args.breakpoint))

    total_words = df["n_words"].to_numpy() if "n_words" in df.columns else None
    rows = []
    for name in marker_names + control_names:
        y_col = f"{name}_per_1k_words"
        if y_col not in df.columns or df[y_col].nunique() <= 1:
            continue
        level_jump, pre_mean, pre_sd = series_effect(design, y_col)
        if not pre_sd or pre_sd <= 0 or np.isnan(pre_sd):
            continue  # can't standardize a flat/degenerate pre-period
        raw_col = f"{name}_raw"
        if raw_col in df.columns:
            total_count = int(df[raw_col].sum())
        elif total_words is not None:
            total_count = int((df[y_col].to_numpy() * total_words / 1000).sum())
        else:
            total_count = None
        rows.append({
            "marker": name,
            "is_control": name in control_set,
            "total_count": total_count,
            "level_jump": level_jump,
            "pre_mean": pre_mean,
            "rel_jump_pct": 100 * level_jump / pre_mean if pre_mean else float("nan"),
            "std_effect": level_jump / pre_sd,
        })

    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("No usable series found in the input.")

    # Set aside series too rare to analyze -- standardizing a near-zero series
    # turns a handful of occurrences into a huge "effect" (e.g. a word appearing
    # ~1x/month). These are reported separately, never flagged as signal.
    res["low_count"] = res["total_count"].notna() & (res["total_count"] < args.min_count)
    reliable = ~res["low_count"]

    ctrl_mask = res["is_control"] & reliable
    control_eff = res.loc[ctrl_mask, "std_effect"].to_numpy()
    if len(control_eff) < 2:
        raise SystemExit(
            f"Only {len(control_eff)} control(s) with >= {args.min_count} occurrences; "
            "need >=2 to estimate a drift band. Lower --min-count or add controls."
        )
    band_mean = float(control_eff.mean())
    band_sd = float(control_eff.std(ddof=1))
    max_ctrl_abs = float(np.abs(control_eff).max())
    reliable_ctrls = res.loc[ctrl_mask]
    strongest_ctrl = reliable_ctrls.iloc[int(np.argmax(np.abs(control_eff)))]["marker"]

    res["z_vs_controls"] = (res["std_effect"] - band_mean) / band_sd
    res["beats_controls"] = res["std_effect"].abs().apply(
        lambda e: int((np.abs(control_eff) < e).sum())
    )
    res["direction"] = np.where(res["std_effect"] >= 0, "up (H1)", "down (H2)")
    # A marker is a candidate signal only if it clears the band both ways:
    # far from the control mean AND larger than every individual control --
    # and only if it has enough occurrences to be reliable.
    res["exceeds_drift"] = (
        (~res["is_control"])
        & reliable
        & (res["z_vs_controls"].abs() >= args.z_threshold)
        & (res["std_effect"].abs() > max_ctrl_abs)
    )

    def by_abs_effect(frame):
        return frame.reindex(frame["std_effect"].abs().sort_values(ascending=False).index)

    markers = by_abs_effect(res[(~res["is_control"]) & reliable])
    controls = by_abs_effect(res[res["is_control"] & reliable])
    excluded = by_abs_effect(res[res["low_count"]])

    cols = ["marker", "total_count", "level_jump", "pre_mean", "rel_jump_pct",
            "std_effect", "z_vs_controls", "beats_controls", "direction", "exceeds_drift"]
    n_ctrl = len(control_eff)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    print(f"\nBreakpoint: {args.breakpoint} | {n_ctrl} controls in drift band "
          f"| min-count = {args.min_count}")
    print(f"Control drift band (standardized effect): mean={band_mean:.3f}, "
          f"sd={band_sd:.3f}; strongest control = {strongest_ctrl} (|effect|={max_ctrl_abs:.3f})")
    print("\n=== MARKERS (sorted by |standardized effect|) ===")
    print(markers[cols].to_string(index=False))
    print("\n=== CONTROLS (the drift baseline) ===")
    print(controls[cols[:-1]].to_string(index=False))
    if not excluded.empty:
        print(f"\n=== EXCLUDED: fewer than {args.min_count} occurrences (too rare to "
              f"analyze; standardized effect is unreliable) ===")
        print(excluded[["marker", "is_control", "total_count", "pre_mean"]].to_string(index=False))

    standouts = markers[markers["exceeds_drift"]]
    print()
    if standouts.empty:
        print("VERDICT: no marker breaks clearly beyond the control drift band. "
              "The marker shifts are not distinguishable from corpus-wide drift "
              "at this breakpoint/corpus -- not evidence of an AI-specific effect.")
    else:
        print(f"VERDICT: {len(standouts)} marker(s) break beyond the control drift "
              f"band (|z| >= {args.z_threshold} and larger than every control):")
        for _, r in standouts.iterrows():
            print(f"  {r['marker']}: {r['direction']}, std_effect={r['std_effect']:.3f}, "
                  f"z={r['z_vs_controls']:.2f}, beats {r['beats_controls']}/{n_ctrl} controls")
        print("These are candidate AI-related shifts; still confounded by any "
              "drift that is correlated with the markers but not the controls.")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out_path, index=False)
        print(f"\nSaved table to {out_path}")


if __name__ == "__main__":
    main()
