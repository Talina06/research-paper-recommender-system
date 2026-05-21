# ArXivists - Research Paper Citation Recommender

**CMPE 256 Recommender Systems, San Jose State University**
Sheetal Sattiraju, Talina Shrotriya, Manjula Ganesh

[GitHub](https://github.com/Talina06/research-paper-recommender-system) | [Report](cmpe_256_report.md) | [License](LICENSE.md)

---

## What this project does

Given a research paper abstract or arXiv ID, the system recommends papers you are likely to cite.
It combines TF-IDF content similarity with graph-based and neural collaborative filtering
(SVD, PageRank, GMF, NeuMF) and exposes recommendations through a Gradio web UI with a
category distribution chart.

The dataset covers up to 100,000 papers per arXiv domain (8 domains, 2015–2025) with citation
edges fetched from Semantic Scholar.

---

## Running the Gradio UI (`app.py`)

### Step 1 — Run the notebook in Colab and export weights

Open `final_notebook_may20.ipynb` in Google Colab and run all cells through **Section 8 (Export)**.
The export cell saves all model artifacts to your Google Drive folder `cmpe_256_project_files/`.

> **You do not need to download pre-trained weights from anywhere.** The notebook trains and saves
> all models itself. The export cell at the end saves everything to Drive.

### Step 2 — Download files from Drive

Go to `cmpe_256_project_files/` in Google Drive and download:

| File | Required? | Unlocks |
|---|---|---|
| `papers.csv` | **required** | app won't start without this |
| `tfidf_vectorizer.pkl` | **required** | app won't start without this |
| `tfidf_matrix.npz` | **required** | app won't start without this |
| `paper_embeddings.npy` | optional | SVD model + Hybrid TF-IDF+SVD model |
| `pagerank_vector.npy` | optional | PageRank model |
| `gmf_model.pt` | optional | GMF model |
| `neumf_model.pt` | optional | NeuMF model |

Models not found on disk are silently skipped — they simply won't appear in the dropdown.
Start with just the three required files and add more as needed.

### Step 3 — Install dependencies and run

```bash
git clone https://github.com/Talina06/research-paper-recommender-system.git
cd research-paper-recommender-system

pip install gradio scikit-learn pandas numpy scipy matplotlib torch
```

```bash
# Point to the folder where you downloaded the Drive files:
python app.py --data-dir /path/to/downloaded/files

# Or if you put the files in the same directory as app.py:
python app.py
```

Open **http://127.0.0.1:7860** in your browser.

### Run the UI directly in Colab

Add a cell at the end of the notebook:

```python
!pip install -q gradio
%run /content/app.py
```

Gradio will print a public share link valid for 72 hours.

---

## Models in the UI

| Model | Input | Signal | Artifact needed |
|---|---|---|---|
| TF-IDF | abstract text | vocabulary / topic overlap | `tfidf_vectorizer.pkl` + `tfidf_matrix.npz` |
| SVD | arXiv ID | citation co-occurrence (latent factors) | `paper_embeddings.npy` |
| PageRank | categories | global citation importance | `pagerank_vector.npy` |
| Hybrid TF-IDF+SVD | abstract text | blended content + graph signal | both of the above |
| GMF | arXiv ID | neural collaborative filtering | `gmf_model.pt` |
| NeuMF | arXiv ID | neural MF (GMF branch + MLP branch) | `neumf_model.pt` |

---

## Running the full pipeline in Colab

1. Open `final_notebook_may20.ipynb` in Google Colab.
2. Add your Semantic Scholar API key to Colab Secrets (key icon in left sidebar) as `SEMANTIC_API_KEY`.
   - Free tier: 1 req/s (slow but works).
   - [Tier 2](https://www.semanticscholar.org/product/api): 10 req/s (set `S2_RPS=9.0`, `S2_WORKERS=5` in the config cell).
3. **Runtime > Run all.**

Pipeline stages:

| Section | What runs | Notes |
|---|---|---|
| 1 - arXiv fetch | OAI-PMH concurrent fetch across 8 domains | Resumes from per-(set,year) checkpoints |
| 2 - S2 citations | Batch citation fetch from Semantic Scholar | Resumes from ID-level checkpoint |
| 3 - Data merge | In-sample filter + integer index | No cold-start filter at dataset level |
| 4 - EDA | Charts and dataset stats | |
| 5 - Eval framework | Train/test split + warm-paper edges | `_get_train_edges()` per model |
| 6 - TF-IDF | Fit vectorizer + evaluate | |
| 7 - Advanced models | SVD, PageRank, PPR-MC, GMF, EG, NeuMF, Hybrid | |
| 8 - Export | Save all artifacts to Drive | Run this before downloading |

All stages write atomic checkpoints to Drive. Re-running resumes automatically from the last
completed point — no data is lost on Colab disconnect.

### Skip the fetch (dataset already in Drive)

Section 3 checks for `papers.csv` and `citations.csv`. If found and complete, it sets
`SKIP_PIPELINE = True` and jumps straight to EDA — no re-fetching.

---

## Cold-start filtering

Cold-start papers (fewer than `MIN_DEGREE=2` total citation connections) are kept in the full
dataset but excluded from CF model training. This lets content-based models (TF-IDF) use all
papers while CF models (SVD, GMF, NeuMF, PageRank, PPR-MC) train only on papers with enough
citation signal.

To change this per model, edit `COLD_START_PER_MODEL` in the eval framework cell (Section 5):

```python
COLD_START_PER_MODEL: dict[str, bool] = {
    "SVD":            True,   # True = train on warm papers only
    "GMF":            True,
    "TF-IDF":         False,  # False = train on all papers
    ...
}
```

---

## Config

All knobs live in the config cell (Section 0) of `final_notebook_may20.ipynb`:

| Variable | Current value | Effect |
|---|---|---|
| `PAPERS_PER_SET` | 100,000 | Papers to collect per arXiv domain |
| `MIN_YEAR` / `MAX_YEAR` | 2015 / 2025 | Year range |
| `OAI_SETS` | 8 domains | cs, math, physics, stat, eess, q-bio, q-fin, econ |
| `S2_BATCH_SIZE` | 500 | S2 API hard limit — do not raise above 500 |
| `S2_RPS` / `S2_WORKERS` | 9.0 / 5 | Requires Tier 2 S2 key; set to 0.9/1 for free tier |
| `MIN_DEGREE` | 2 | Min citation degree to consider a paper "warm" |
| `EVAL_SAMPLE` | 1,000 | Query papers used for evaluation |

---

## Repo layout

```
research-paper-recommender-system/
├── final_notebook_may20.ipynb     # main notebook: pipeline + EDA + models + eval
├── app.py                         # Gradio web UI
├── cmpe_256_report.md             # written project report
├── README.md                      # this file
├── quick_wins.md                  # model improvement ideas
├── LICENSE.md
└── v1/                            # archived exploratory notebooks
```

**Google Drive folder:** `cmpe_256_project_files/`

```
cmpe_256_project_files/
├── papers.csv                     # paper metadata (required)
├── citations.csv                  # in-sample citation edges
├── citation_pop.csv               # citation popularity scores
├── tfidf_vectorizer.pkl           # fitted TF-IDF vectorizer (required)
├── tfidf_matrix.npz               # pre-built TF-IDF matrix (required)
├── paper_embeddings.npy           # 128-dim SVD embeddings (optional)
├── pagerank_vector.npy            # PageRank scores per paper_idx (optional)
├── gmf_model.pt                   # GMF state dict (optional)
├── neumf_model.pt                 # NeuMF state dict (optional)
├── model_comparison.png           # compare_models() output chart
│
│   -- checkpoint files (safe to delete after pipeline finishes) --
├── _ckpt_oai_{set}_{year}.csv     # arXiv per-(set,year) checkpoints
├── _ckpt_oai_{set}_{year}.exhausted   # sentinel: stream fully consumed
├── _ckpt_s2_pairs.csv             # S2 citation pairs checkpoint
├── _ckpt_s2_fields.json           # S2 fields-of-study checkpoint
└── _ckpt_s2_done.json             # S2 completed paper IDs
```

---

## Evaluation framework

All models share the same 80/20 edge split (seed=256) and 1,000 fixed query papers (seed=42).
Metrics: Precision@K, Recall@K, NDCG@K, Hit Rate@K, MRR at K ∈ {5, 10, 20}.

```python
# Per-paper interface:
evaluate_model("My Model", score_fn=lambda q_idx: scores_array)

# Batch interface (faster for matrix models):
evaluate_model("My Model", batch_score_fn=lambda indices: score_matrix, batch_size=100)

compare_models()  # side-by-side table + bar chart
```

---

## References

- arXiv OAI-PMH API: https://info.arxiv.org/help/oa/index.html
- Semantic Scholar API: https://www.semanticscholar.org/product/api
