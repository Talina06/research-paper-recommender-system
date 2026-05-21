# ArXivists - Research Paper Citation Recommender

**CMPE 256 Recommender Systems, San Jose State University**
Sheetal Sattiraju, Talina Shrotriya, Manjula Ganesh

---

## What this project does

Given a research paper abstract, the system recommends papers you are likely to cite.
It combines TF-IDF content similarity with graph-based and neural collaborative filtering
and exposes all nine trained models through a Gradio web UI with a category distribution chart.

**Dataset:** up to 100,000 papers per arXiv domain (8 domains, 2015–2025) with citation
edges fetched from Semantic Scholar — 486,916 total papers, 171,289 warm papers (degree ≥ 2),
951,597 in-sample citation edges.

---

## Repo layout

```
research-paper-recommender-system/
├── notebook.ipynb          # full pipeline: data fetch, EDA, 9 models, eval
├── notebook.html           # pre-rendered static view of notebook with outputs
├── app.py                  # Gradio web UI
├── cmpe_256_report.md      # written project report
├── README.md
├── CONTRIBUTING.md
└── LICENSE.md
```

---

## Running the Gradio UI

### Step 1 — Run the notebook in Colab and export weights

Open `notebook.ipynb` in Google Colab and run all cells through **Section 8 (Export)**.
This saves all model artifacts to `cmpe_256_project_files/` in your Google Drive.

> No pre-trained weights exist elsewhere — the notebook trains and saves everything itself.

### Step 2 — Download artifacts from Drive

Download these files from `cmpe_256_project_files/` in Google Drive:

| File | Required? | Enables |
|---|---|---|
| `papers.csv` | **yes** | app won't start without this |
| `citations.csv` | **yes** | app won't start; needed for PPR-MC |
| `tfidf_vectorizer.pkl` | **yes** | app won't start without this |
| `tfidf_matrix.npz` | **yes** | app won't start without this |
| `paper_embeddings.npy` | optional | SVD model |
| `pagerank_vector.npy` | optional | PageRank model |
| `hybrid_embeddings.npy` | optional | Hybrid TF-IDF+SVD model |
| `gmf_model.pt` | optional | GMF model |
| `neumf_model.pt` | optional | NeuMF model |

Models not found on disk are silently skipped and won't appear in the dropdown.

### Step 3 — Install and run

```bash
pip install gradio scikit-learn pandas numpy scipy matplotlib torch
python app.py --data-dir /path/to/downloaded/files
# open http://127.0.0.1:7860
```

If `--data-dir` is omitted, `app.py` looks in the current directory.

### Run directly in Colab

```python
!pip install -q gradio
%run /content/app.py
```

Gradio prints a public share link valid for 72 hours.

---

## Models in the UI

All nine notebook models are available via dropdown. Every model accepts a free-text
abstract as input. Graph-based and neural models use a **TF-IDF anchor step**: the abstract
is matched to the closest paper in the dataset by cosine similarity, and that paper's
citation graph index is used as the query node.

| Model | Signal | Artifact needed |
|---|---|---|
| TF-IDF | Vocabulary / topic overlap | `tfidf_matrix.npz` |
| SVD | Citation co-occurrence (latent factors) | `paper_embeddings.npy` |
| PageRank | Global citation importance | `pagerank_vector.npy` |
| PPR-MC | Personalized PageRank from TF-IDF anchor | `citations.csv` |
| TF-IDF + PPR-MC | Blended 50% TF-IDF + 50% PPR-MC | `citations.csv` |
| Epsilon-Greedy | TF-IDF candidate pool + bandit reranking | `tfidf_matrix.npz` |
| GMF | Neural collaborative filtering | `gmf_model.pt` |
| NeuMF | Neural MF (GMF branch + MLP branch) | `neumf_model.pt` |
| Hybrid TF-IDF+SVD | Content + citation latent blend | `hybrid_embeddings.npy` |

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

## Running the full pipeline in Colab

1. Open `notebook.ipynb` in Colab.
2. Add your Semantic Scholar API key to **Colab Secrets** (key icon, left sidebar) as `SEMANTIC_API_KEY`.
   - Free tier: `S2_RPS=0.9`, `S2_WORKERS=1`
   - [Tier 2](https://www.semanticscholar.org/product/api): `S2_RPS=9.0`, `S2_WORKERS=5`
3. **Runtime > Run all.**

| Section | What runs |
|---|---|
| 1 — arXiv fetch | OAI-PMH fetch across 8 domains, resumes from checkpoints |
| 2 — S2 citations | Batch citation fetch, resumes from ID-level checkpoint |
| 3 — Data merge | In-sample filter + integer index |
| 4 — EDA | Charts and stats on 486k papers |
| 5 — Eval framework | 80/20 split, warm-edge computation |
| 6 — TF-IDF | Baseline model, NDCG@10 = 0.1021 |
| 7 — Advanced models | 8 models, best: TF-IDF+PPR-MC NDCG@10 = 0.1501 |
| 8 — Export | Save all artifacts to Drive |

All stages write checkpoints to Drive. Re-running resumes automatically from the last
completed point — no data is lost on Colab disconnect.

---

## Config

All knobs live in the config cell (Section 0) of `notebook.ipynb`:

| Variable | Value | Effect |
|---|---|---|
| `PAPERS_PER_SET` | 100,000 | Papers collected per arXiv domain |
| `MIN_YEAR` / `MAX_YEAR` | 2015 / 2025 | Year range |
| `S2_BATCH_SIZE` | 500 | S2 API hard limit — do not raise above 500 |
| `S2_RPS` / `S2_WORKERS` | 9.0 / 5 | Requires Tier 2 S2 key; use 0.9/1 for free tier |
| `MIN_DEGREE` | 2 | Min citation degree for a "warm" paper |
| `EVAL_SAMPLE` | 1,000 | Query papers used for evaluation |

---

## License

MIT — see [LICENSE.md](LICENSE.md).

---

## References

- arXiv OAI-PMH API: https://info.arxiv.org/help/oa/index.html
- Semantic Scholar API: https://www.semanticscholar.org/product/api
