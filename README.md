# AI Marker Drift

Does usage of certain punctuation/lexical markers (em dash, "delve," "underscore," etc.)
shift measurably around the release of mainstream LLM writing tools (ChatGPT, Nov 30 2022)?

Two competing hypotheses this pipeline is built to test:

- **H1 (adoption effect):** markers associated with LLM output *increase* in a corpus
  after Nov 2022, because more of the corpus is now LLM-assisted or LLM-written.
- **H2 (avoidance effect):** markers *decrease* after some detection threshold, because
  human writers self-censor once a marker becomes a recognized "AI tell."

These aren't mutually exclusive — a marker can spike short-term (H1) and then decline
once "AI-detection anxiety" sets in publicly (H2), which is itself a testable
second breakpoint.

## Pipeline

```
src/ingest/          -> pulls raw text + publication dates from a corpus source
src/analysis/         -> tokenizes, computes marker frequency per 1,000 words,
                          aggregates to monthly time series, runs structural-break stats
config/markers.yaml   -> the marker list (edit this to add/remove markers)
data/raw/              -> untouched source documents (jsonl, one doc per line)
data/processed/        -> monthly frequency tables (parquet/csv)
```

## Corpora wired up so far

1. **PubMed/PMC abstracts** (`src/ingest/fetch_pubmed.py`) — via NCBI E-utilities.
   Free, no key required (10 req/sec with a free API key, 3/sec without).
   Good for the "academic writing" leg of the theory.
2. **Google Books Ngram** (`src/ingest/fetch_ngram.py`) — free JSON endpoint, no auth.
   Only goes through ~2022 in the public corpus, so it's a *pre-period baseline*,
   not useful for the post-ChatGPT window. Use it to sanity-check your marker
   list against a century of prior trend before you trust anything post-2022.
3. **Guardian articles** (`src/ingest/fetch_guardian.py`) — the journalism leg
   of the theory, via the Guardian Open Platform Content API. Free but needs an
   API key (`GUARDIAN_API_KEY`); free tier is ~1 req/s, 500 req/day. Same
   `{id, date, text}` jsonl contract, so it drops straight into the analysis step.

Not yet wired up (see CLAUDE.md for notes): NYT API, arXiv bulk dumps,
Common Crawl. Each needs its own ingest script following the same
`(text, pub_date, source_id)` contract so they drop into the same analysis step.

## Quickstart

```bash
pip install -r requirements.txt
python src/ingest/fetch_pubmed.py --query "cancer AND treatment" \
    --start 2020-01-01 --end 2026-06-01 --out data/raw/pubmed_cancer.jsonl
python src/analysis/extract_features.py --in data/raw/pubmed_cancer.jsonl \
    --out data/processed/pubmed_cancer_monthly.csv
python src/analysis/regression_discontinuity.py \
    --in data/processed/pubmed_cancer_monthly.csv --marker em_dash --breakpoint 2022-11-30
# optional: let the data place the break instead of hypothesizing a date
python src/analysis/detect_breaks.py \
    --in data/processed/pubmed_cancer_monthly.csv --marker em_dash --n-breaks 1
```

`extract_features.py` also emits per-month quantiles of the *per-document*
frequency distribution (`{marker}_pQQ_per_1k_words`), for testing the H2
avoidance hypothesis on the right tail — e.g. `--marker em_dash_p90` feeds the
90th-percentile series straight into `regression_discontinuity.py`.

See `CLAUDE.md` for current state and what's left to build.
