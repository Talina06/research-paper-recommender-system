# ArXivists - Research Paper Citation Recommender

**CMPE 256 Recommender Systems, San Jose State University**
Sheetal Sattiraju, Talina Shrotriya, Manjula Ganesh

[GitHub](https://github.com/Talina06/research-paper-recommender-system) | [Report](cmpe_256_report.md) | [License](LICENSE.md)

---

## What this project does

Given a research paper abstract, the system recommends papers you are likely to cite.
It combines TF-IDF content similarity with graph-based and neural collaborative filtering
and exposes all nine trained models through a Gradio web UI with a category distribution chart.

**Dataset:** up to 100,000 papers per arXiv domain (8 domains, 2015–2025) with citation
edges fetched from Semantic Scholar — 486,916 total papers, 171,289 warm papers (degree ≥ 2),
951,597 in-sample citation edges.

---

## Running the Gradio UI

### Step 1 — Run the notebook in Colab and export weights

Open `notebook.ipynb` in Google Colab and run all cells through **Section 8 (Export)**.
The export cell saves all model artifacts to `cmpe_256_project_files/` in your Google Drive.

> No pre-trained weights exist elsewhere. The notebook trains and saves everything itself.

### Step 2 — Download artifacts from Drive

```bash
python download_data.py                   # all available files → ./data/
python download_data.py --required-only   # TF-IDF only (~550 MB, fastest)
python download_data.py --out-dir ~/data  # custom directory
```

What gets downloaded:

| File | Required? | Enables |
|---|---|---|
| `papers.csv` | yes | app won't start without this |
| `citations.csv` | yes | app won't start without this; needed for PPR-MC |
| `tfidf_vectorizer.pkl` | yes | app won't start without this |
| `tfidf_matrix.npz` | yes | app won't start without this |
| `paper_embeddings.npy` | optional | SVD model |
| `pagerank_vector.npy` | optional | PageRank model |
| `hybrid_embeddings.npy` | optional | Hybrid TF-IDF+SVD model |
| `gmf_model.pt` | optional | GMF model |
| `neumf_model.pt` | optional | NeuMF model |

Models not found on disk are silently skipped — they won't appear in the dropdown.

### Step 3 — Install and run

```bash
pip install gradio scikit-learn pandas numpy scipy matplotlib torch
python app.py --data-dir ./data
# open http://127.0.0.1:7860
```

### Run directly in Colab

```python
!pip install -q gradio
%run /content/app.py
```

Gradio prints a public share link valid for 72 hours.

---

## Models in the UI

All nine notebook models are available. Each accepts an abstract text query. The UI uses
a **TF-IDF anchor step** to bridge text queries into graph-based models: the abstract is
matched to the closest paper in the dataset by cosine similarity, and that paper's index
is used as the query node for the graph model.

| Model | Signal | Artifact |
|---|---|---|
| TF-IDF | Vocabulary / topic overlap | `tfidf_matrix.npz` |
| SVD | Citation co-occurrence (latent factors) | `paper_embeddings.npy` |
| PageRank | Global citation importance | `pagerank_vector.npy` |
| PPR-MC | Personalized PageRank from TF-IDF anchor | `citations.csv` |
| TF-IDF + PPR-MC | Blended (50% TF-IDF + 50% PPR-MC) | `citations.csv` |
| Epsilon-Greedy | TF-IDF candidate pool + bandit reranking | `tfidf_matrix.npz` |
| GMF | Neural collaborative filtering | `gmf_model.pt` |
| NeuMF | Neural MF (GMF branch + MLP branch) | `neumf_model.pt` |
| Hybrid TF-IDF+SVD | Content + citation latent blend | `hybrid_embeddings.npy` |

---

## Running the full pipeline in Colab

1. Open `notebook.ipynb` in Colab.
2. Add your Semantic Scholar API key to Colab Secrets as `SEMANTIC_API_KEY`.
   - Free tier (Tier 1): 1 req/s — set `S2_RPS=0.9`, `S2_WORKERS=1`
   - [Tier 2](https://www.semanticscholar.org/product/api): 10 req/s — set `S2_RPS=9.0`, `S2_WORKERS=5`
3. **Runtime > Run all.**

| Section | What runs | Notes |
|---|---|---|
| 1 — arXiv fetch | OAI-PMH concurrent fetch, 8 domains | Resumes from per-(set, year) checkpoints |
| 2 — S2 citations | Batch citation fetch | Resumes from ID-level checkpoint |
| 3 — Data merge | In-sample filter + integer index | Cold-start filter applied per-model at train time |
| 4 — EDA | Charts, stats, cold-start analysis | 486k total / 171k warm papers |
| 5 — Eval framework | 80/20 split + warm-edge computation | `_get_train_edges()` per model |
| 6 — TF-IDF | Fit vectorizer + evaluate | NDCG@10 = 0.1021 |
| 7 — Advanced models | 8 models evaluated | Best: TF-IDF+PPR-MC NDCG@10 = 0.1501 |
| 8 — Export | Save all artifacts to Drive | Run before downloading |

Checkpoints are written atomically to Drive. Re-running resumes automatically.

### Skip the pipeline (dataset already in Drive)

Section 3 checks for `papers.csv` and `citations.csv`. If found and valid, it sets
`SKIP_PIPELINE = True` and jumps to EDA — no re-fetching.

---

## Model results

From the latest notebook run (1,000 query papers, 80/20 split, seed=256):

| Model | NDCG@10 | HR@10 | Recall@10 |
|---|---|---|---|
| **TF-IDF + PPR-MC** | **0.1501** | **0.3080** | **0.2088** |
| PPR-MC | 0.1141 | 0.2690 | 0.1632 |
| TF-IDF | 0.1021 | 0.2250 | 0.1486 |
| Epsilon-Greedy | 0.0511 | 0.0750 | 0.0488 |
| Hybrid TF-IDF+SVD | 0.0393 | 0.0980 | 0.0592 |
| PageRank | 0.0089 | 0.0210 | 0.0102 |
| NeuMF | 0.0069 | 0.0230 | 0.0093 |
| SVD | 0.0048 | 0.0210 | 0.0068 |
| GMF | 0.0038 | 0.0170 | 0.0042 |

---

## Cold-start filtering

Cold-start papers (fewer than `MIN_DEGREE=2` total citation connections) are kept in the
full dataset but excluded from CF model training. Content-based models (TF-IDF,
Epsilon-Greedy) use all 486k papers; CF models train only on the 171k warm papers.

To change which models apply cold-start, edit `COLD_START_PER_MODEL` in the eval
framework cell (Section 5):

```python
COLD_START_PER_MODEL: dict[str, bool] = {
    "SVD":            True,   # True = train on warm papers only
    "GMF":            True,
    "TF-IDF":         False,  # False = use all papers
    ...
}
```

---

## Config

All knobs are in the config cell (Section 0) of `notebook.ipynb`:

| Variable | Value | Effect |
|---|---|---|
| `PAPERS_PER_SET` | 100,000 | Papers to collect per arXiv domain |
| `MIN_YEAR` / `MAX_YEAR` | 2015 / 2025 | Year range |
| `OAI_SETS` | 8 domains | cs, math, physics, stat, eess, q-bio, q-fin, econ |
| `S2_BATCH_SIZE` | 500 | S2 API hard limit — do not raise above 500 |
| `S2_RPS` / `S2_WORKERS` | 9.0 / 5 | Requires Tier 2 S2 key; use 0.9/1 for free tier |
| `MIN_DEGREE` | 2 | Min citation degree for a paper to be "warm" |
| `EVAL_SAMPLE` | 1,000 | Query papers used for evaluation |

---

## Repo layout

```
research-paper-recommender-system/
├── notebook.ipynb          # main notebook: pipeline + EDA + all 9 models + eval
├── app.py                  # Gradio web UI
├── download_data.py        # fetches all artifacts from Google Drive
├── cmpe_256_report.md      # written project report
├── README.md
├── CONTRIBUTING.md
├── quick_wins.md           # model improvement ideas
├── LICENSE.md
└── v1/                     # archived exploratory notebooks
```

**Google Drive folder:** `cmpe_256_project_files/`

```
cmpe_256_project_files/
├── papers.csv              # 486,916 papers (arxiv_id, title, abstract, ...)
├── citations.csv           # 951,597 in-sample citation edges
├── tfidf_vectorizer.pkl    # fitted TF-IDF vectorizer
├── tfidf_matrix.npz        # pre-built TF-IDF sparse matrix (~310 MB)
├── paper_embeddings.npy    # 128-dim citation-graph SVD embeddings
├── pagerank_vector.npy     # PageRank scores per paper_idx
├── hybrid_embeddings.npy   # 128-dim text-SVD embeddings (Hybrid model)
├── gmf_model.pt            # GMF state dict
├── neumf_model.pt          # NeuMF state dict
├── model_comparison.png    # compare_models() chart
└── _ckpt_*/                # checkpoint files (safe to delete after pipeline)
```

---

## Evaluation framework

All models share the same 80/20 edge split (seed=256) and 1,000 fixed query papers (seed=42).
Metrics: Precision@K, Recall@K, NDCG@K, Hit Rate@K, MRR at K ∈ {5, 10, 20}.

```python
# Per-paper scoring:
evaluate_model("My Model", score_fn=lambda q_idx: scores_array)

# Batch scoring (faster for matrix models):
evaluate_model("My Model", batch_score_fn=lambda indices: score_matrix, batch_size=100)

compare_models()  # side-by-side table + bar chart + saves model_comparison.png
```

---

## References

- arXiv OAI-PMH API: https://info.arxiv.org/help/oa/index.html
- Semantic Scholar API: https://www.semanticscholar.org/product/api
