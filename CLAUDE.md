# CLAUDE.md — AI Marker Drift project

## What this is
Testing whether markers associated with LLM-generated text (em dash, "delve,"
"underscore," "crucial," etc.) show a statistically significant shift in
frequency around Nov 2022 (ChatGPT release), in either direction:
- increase (LLM-assisted writing entering the corpus), or
- decrease (human writers avoiding the marker once it's recognized as an AI tell).

## Status: pipeline scaffolded and smoke-tested, not yet run on real data

Verified working end-to-end on synthetic data:
- `src/analysis/extract_features.py` — tokenizes a jsonl corpus, computes
  per-1,000-word marker frequency (word-boundary regex matching for phrases),
  aggregates to monthly CSV. Confirmed correct on synthetic data with a
  known injected break.
- `src/analysis/regression_discontinuity.py` — segmented regression with
  Newey-West (HAC) standard errors + robust Wald break test, plus placebo
  breakpoint tests run on pre-break data only. Verified: detects the injected
  break, correctly skips placebos outside the data range, and correctly
  returns non-significant (p=0.41) for a placebo date on pre-break data.
  Note: placebos MUST stay restricted to pre-break data — an earlier version
  ran them on the full series and they fired spuriously because the real
  break sat inside the placebo's post-period. Don't undo that fix.
- `src/analysis/run_all_markers.py` — batch runner: all markers + controls,
  Benjamini-Hochberg FDR correction across the set, warns loudly if a
  negative control survives correction (= corpus drift, not AI). Skips
  constant/all-zero series (degenerate Wald otherwise). Verified on
  synthetic data: injected markers significant post-BH, control not.

Not yet run against real data:
- `src/ingest/fetch_pubmed.py` — written against the documented NCBI
  E-utilities API but never actually hit the live endpoint (this sandbox's
  network allowlist doesn't include eutils.ncbi.nlm.nih.gov). Run this from
  your own machine or a Claude Code session with open network access.
  Sanity-check the XML parsing (`parse_pubdate`, abstract extraction) against
  a couple of real records before trusting a full pull — PubMed XML has
  enough structural variation (missing months, MedlineDate-only dates,
  multi-part abstracts) that there are likely edge cases the current parser
  doesn't hit.
- `src/ingest/fetch_ngram.py` — same story, untested against the live Google
  Ngram endpoint. Also: confirmed via research that Ngram's public corpus
  only runs through ~2022, so it's only useful as a pre-ChatGPT baseline,
  not for measuring the actual break.

## Immediate next steps (in priority order)
1. Run `fetch_pubmed.py` on a small query/date range first (e.g. one month)
   to confirm the XML parsing holds up, before committing to a big pull.
2. Wire up a second corpus for a non-academic comparison point — NYT or
   Guardian API is the natural pick for "journalism," since that's the other
   domain in the original theory. Same `{id, date, text}` jsonl contract,
   drops straight into `extract_features.py` with no changes needed there.
3. Decide on the actual query/topic scope for PubMed — "cancer AND treatment"
   in the README is a placeholder. Pick a topic-stable field (topic drift
   over years is one of the confounds noted below) with high enough monthly
   volume for the regression to have power — a few thousand abstracts/month
   minimum, more if the effect size is likely small.
4. Once real data is in, re-run `regression_discontinuity.py` per marker and
   look at both:
   - the Nov 2022 breakpoint (H1: adoption effect)
   - a later 2023–2024 breakpoint you'll need to pick, tied to when em-dash
     "AI tell" articles started circulating widely (H2: avoidance effect) —
     worth a quick search to pin an actual date before hardcoding one.

## Known confounds, not yet controlled for
- Style guide edition changes (APA 7 vs prior, journal-specific updates)
- Shifts in which journals/outlets are in-sample month to month
- Topic drift (some topics naturally use more parenthetical asides)
- PubMed skews toward edited/peer-reviewed prose — expect a *smaller* effect
  there than in less-edited web text; don't over-generalize from PubMed alone

## Improvements identified but NOT yet implemented
- Bai-Perron automated break detection (`ruptures` library) — finding a break
  near Nov 2022 without specifying the date is stronger evidence than testing
  a hypothesized date
- Per-document distribution analysis — H2 (avoidance) predicts the right tail
  of per-doc em-dash counts shrinking even if the mean barely moves; current
  monthly aggregation discards that. Needs extract_features to also emit
  per-doc counts (or quantiles per month).
- Checkpoint/resume in fetch_pubmed.py for long pulls
- NYT/Guardian ingest scripts (journalism leg of the theory)

## Design decisions already made, don't relitigate without reason
- Frequency normalized per 1,000 words, not raw counts (controls doc length)
- Segmented OLS + Chow F-test chosen over a naive pre/post t-test, because
  monthly counts are autocorrelated and a t-test on means overstates
  significance
- Em dash matching includes literal "--" as a fallback since some corpora
  (older plain-text dumps especially) use ASCII double-hyphen instead of U+2014
- Marker list lives in `config/markers.yaml`, not hardcoded, so it's a config
  edit rather than a code change to add/remove markers
- HAC (Newey-West, maxlags=6) errors for all inference; the naive Chow F is
  printed only for reference and should never be the headline number
- BH FDR correction across the full marker set in batch mode; single-marker
  p-values from regression_discontinuity.py are exploratory only
- Negative controls live in `config/markers.yaml` under `controls:` and run
  through the identical pipeline as real markers by design

## Don't re-do
- Don't rebuild the Google Books Ngram script to query em dash/semicolon
  directly — already confirmed Ngram's tokenizer strips most punctuation,
  it won't return usable data for those two markers.
