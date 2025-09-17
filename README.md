# News Sensitivity — News analysis pipeline

## Overview

- Purpose: This repository contains a single-file ETL pipeline (`main.py`) that extracts news articles, optionally scrapes full article text from Reuters, enriches articles with AI (DeepSeek), and saves raw and enriched outputs for analysis.
- Data sources:
  - NewsAPI (https://newsapi.org) — primary source for headlines and article metadata (title, description, url, publishedAt, source).
  - Aggregator pages (Yahoo, Memeorandum, Yahoo Finance, etc.) — sometimes NewsAPI returns syndicated articles on aggregator domains; the pipeline attempts to detect original Reuters links on these pages.
  - Reuters (https://reuters.com) — where possible the pipeline scrapes full Reuters article text for higher-quality analysis.

## DeepSeek Enhancements

- The `deepseek_enrichment` module is used to perform AI-powered enrichment of articles. Enhancements include:
  - Sentiment analysis and confidence scores
  - Emotional tone detection
  - Primary and secondary topic/category classification
  - Keyword extraction
  - Bias level and bias direction estimation
  - Objectivity scoring and AI reasoning summaries

- These enrichments are performed in batches and appended as new columns in the enriched output CSV. The enrichment module is expected to expose `enrich_dataframe(df, deepseek_key, batch_size)` and `save_enriched_data()` functions.

## Before/After Examples (AI Value)

- The pipeline writes Markdown examples to `examples/before_after_examples_recent.md` and a CSV sample at `examples/sample_enriched_articles_recent.csv`.
- The examples show the original (raw) data fields (title, description, source, data quality score) alongside DeepSeek outputs (sentiment, keywords, bias, categories and AI reasoning). This demonstrates how AI adds structured context and reasoning that is not present in raw metadata.

## Installation

1. Clone the repository and change into the project folder.

2. Create and activate a Python virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

- Provide your NewsAPI key in one of three ways (priority order):
  1. Edit `main.py` and set the `NEWSAPI_KEY` variable near the top of the file.
  2. Pass `--newsapi-key YOUR_KEY` on the command line.
  3. Export an environment variable: `export NEWSAPI_KEY="YOUR_KEY"`.

- Run the full pipeline:

```bash
python3 main.py
```

- Optional flags:
  - `--deepseek-key KEY` — use a custom DeepSeek API key (defaults to included demo key).
  - `--max-articles N` — limit how many articles to extract (default: 30).
  - `--max-scrapes N` — limit Reuters scraping attempts (default: 15).
  - `--categories cat1 cat2 ...` — categories for NewsAPI headlines (default: `technology business`).
  - `--search-terms term1 term2 ...` — search terms for NewsAPI "everything" endpoint (default: `Reuters`).
  - `--test-connection` — test the NewsAPI connection and exit.

## Outputs

- `data/raw/pipeline_raw_data_recent.csv` — cleaned raw articles saved by the pipeline.
- `data/enriched/pipeline_enriched_data_recent.csv` — enriched articles with DeepSeek outputs.
- `data/enriched/analysis_summary.txt` — summary report of pipeline run.
- `examples/before_after_examples_recent.md` — Markdown before/after AI example file.
- `examples/sample_enriched_articles_recent.csv` — top sample of enriched articles.

## Notes & Troubleshooting

- Reuters may block automated scraping (401/403). The pipeline includes a proxy fallback and improved request headers, but scraping success can vary. If you need high coverage consider using an official Reuters feed or a paid scraping/proxy service.
- Keep API keys secret: do not commit your `NEWSAPI_KEY` or `--deepseek-key` into public version control.

## License

- This repository does not include a license by default. Add one if you intend to publish.
