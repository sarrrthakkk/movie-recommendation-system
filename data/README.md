# Dataset and Metadata Retrieval Guide

This directory manages datasets and metadata for the hybrid movie recommendation and LLM reranking pipeline.

---

## Redistribution & Data Policy

Source datasets and external metadata caches are excluded from version control:
- **MovieLens 100K:** GroupLens restricts redistribution of MovieLens datasets without express permission.
- **TMDB API Content:** The Movie Database (TMDB) terms of use prohibit redistributing TMDB content (plot overviews, credits, cast, and crew) as a standalone file or dataset.
- **Local Response Cache:** `artifacts/cache/llm_cache.sqlite` is generated during LLM reranking and explanation runs, containing cached prompts with user history and plot snippets, and is therefore also excluded.

Only lightweight aggregate split summaries ([split_summary.json](split_summary.json)) and catalog mappings ([genre_names.json](genre_names.json)) are tracked in this repository.

---

## How to Retrieve MovieLens 100K

### 1. Automated Download
From the repository root, run:
```bash
python scripts/download_data.py
```
This fetches the official GroupLens archive and extracts `u.data`, `u.item`, and supporting files directly into `data/raw/ml-100k/`.

### 2. Manual Download Fallback
1. Download `ml-100k.zip` from [GroupLens MovieLens 100K](https://grouplens.org/datasets/movielens/100k/).
2. Extract the archive into `data/raw/ml-100k/` (or `./ml-100k/`).

### 3. Build Processed Chronological Splits
Run the data preparation script to generate chronological train, validation, and test splits along with precomputed genre vectors:
```bash
python scripts/prepare_data.py --data-dir data/raw/ml-100k
```
This populates `data/processed/` with `movies.csv`, `ratings.csv`, `train.csv`, `validation.csv`, `test.csv`, `genres.npy`, `movie_ids.npy`, `genre_names.json`, and `split_summary.json`.

---

## How to Retrieve TMDB Metadata (Plots, Directors, and Cast)

`scripts/fetch_tmdb_metadata.py` queries The Movie Database API to enrich each MovieLens movie with plot overviews, director names, and top billing cast members.

### Step-by-Step API Key Setup
1. **Create an account:** Register a free account at [The Movie Database (TMDB)](https://www.themoviedb.org) and confirm your verification email (unverified accounts will return HTTP 401 Unauthorized errors).
2. **Request an API key:** Log in using a desktop browser, click your user avatar -> **Settings**, select **API** from the left navigation sidebar, and click **Create** under the **Developer (non-commercial)** option.
3. **Accept terms:** Complete the brief application form explaining your educational/research use case.
4. **Configure `.env`:** Copy your v3 **API Key** (a 32-character hexadecimal string) into your local `.env` file:
   ```dotenv
   TMDB_API_KEY=<your_tmdb_api_key>
   ```
   *(Never commit `.env` to version control).*
5. **Fetch and cache metadata:** Run the metadata fetcher:
   ```bash
   python scripts/fetch_tmdb_metadata.py
   ```
   This generates `data/processed/tmdb_metadata.json` locally for all 1,682 MovieLens titles (~1,670 with full plot summaries).

---

## Citations and Attribution

- **MovieLens 100K Citation:**
  > F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

- **TMDB Attribution:**
  > *This product uses the TMDB API but is not endorsed or certified by TMDB.*
  > For more information, visit [The Movie Database (TMDB)](https://www.themoviedb.org).
