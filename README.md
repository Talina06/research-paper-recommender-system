# ArXivists - Research Paper Citation Recommender

**CMPE 256 Recommender Systems, San Jose State University**
Sheetal Sattiraju, Talina Shrotriya, Manjula Ganesh

[GitHub](https://github.com/Talina06/research-paper-recommender-system) | [Report](cmpe_256_report.md) | [License](LICENSE.md) | [Contributing](CONTRIBUTING.md)

---

## What this project does

Given a research paper description or arXiv category, the system recommends papers
you are likely to cite. It combines TF-IDF content similarity with collaborative
filtering on the citation graph (SVD, GMF + NeuMF, PageRank) and a final hybrid model,
and exposes recommendations through a Gradio web UI.

The dataset covers 163,275 papers from arXiv (cs.LG, cs.AI, cs.CV, cs.IR, cs.CL,
stat.ML, 2018-2024) with 1.9 million citation edges fetched from Semantic Scholar.

---

## Quick start

### Prerequisites

- Python 3.9+
- A free [Semantic Scholar API key](https://www.semanticscholar.org/product/api)
  (only needed if re-fetching citation data from scratch)
- Google Colab account (recommended) or a local GPU/CPU with 16 GB+ RAM

### Run in Colab (recommended)

1. Open `notebook.ipynb` in Google Colab.
2. In the **Secrets** panel (key icon in the left sidebar), add `SEMANTIC_API_KEY`.
3. **Runtime > Run all.**

The pipeline runs in five stages:

| Section | What runs | Approx. time |
|---|---|---|
| 1 - arXiv fetch | 21 parallel OAI-PMH streams | 3-5 min |
| 2 - S2 citations | Batch API, 1 req/s (free key) | 4-6 hr |
| 3 - Data merging | In-sample filter + cold-start + index | < 1 min |
| 4 - EDA | Plots and stats | < 1 min |
| 5-6 - Eval + baseline | TF-IDF build + evaluation on 1,000 papers | 2-3 min |

All stages write atomic checkpoints to Drive. Re-running resumes from the last
completed batch automatically.

### Skip the fetch (dataset already in Drive)

Section 3 checks for `papers.csv` and `citations.csv` in Drive. If found, it sets
`SKIP_PIPELINE = True` and jumps straight to Section 4 EDA — no re-fetching needed.

### Run the Gradio UI locally

**Step 1 — Train models in Colab and export**

Run `notebook.ipynb` in Colab through Section 7. The export cell lists every file
in the Drive folder with its size.

**Step 2 — Download files from Drive**

Download these files from `cmpe_256_project_files/` in Google Drive:

| File | Size | Notes |
|---|---|---|
| `papers.csv` | ~60 MB | required |
| `tfidf_vectorizer.pkl` | ~10 MB | required, saved by notebook after TF-IDF trains |
| `tfidf_matrix.npz` | ~250 MB | recommended, skips ~60s rebuild at startup |
| `svd_model.npz` | varies | optional, auto-loaded if present |
| `pagerank_scores.json` | varies | optional, auto-loaded if present |
| `hybrid_weights.json` | small | optional, auto-loaded if present |

**Step 3 — Run the app**

```bash
git clone https://github.com/Talina06/research-paper-recommender-system.git
cd research-paper-recommender-system
pip install gradio scikit-learn pandas numpy scipy

python app.py --data-dir /path/to/downloaded/files
# open http://127.0.0.1:7860
```

If `--data-dir` is omitted, `app.py` looks for files in the current directory.

### Run the Gradio UI in Colab

Run Section 8 in the notebook. It installs Gradio, fetches `app.py` from Drive
(or downloads it from GitHub if not there), and launches a public share link
valid for 72 hours.

---

## Repo layout

```
research-paper-recommender-system/
├── notebook.ipynb                 # main notebook: pipeline + EDA + models + eval
├── app.py                         # Gradio web UI
├── cmpe_256_report.md             # written project report
├── README.md                      # this file
├── LICENSE.md                     # MIT license
├── CONTRIBUTING.md                # contributor guide
└── v1/                            # archived exploratory notebooks
```

**Google Drive folder:** `cmpe_256_project_files/`

```
cmpe_256_project_files/
├── papers.csv                  # 163,275 papers (arxiv_id, title, abstract, ...)
├── citations.csv               # 1,910,207 in-sample citation edges
├── citation_pop.csv            # normalized citation popularity scores
├── app.py                      # copy here so Colab can launch the UI
│
│   -- model artefacts (drop here to auto-register in the UI) --
├── svd_model.npz               # SVD latent factors
├── pagerank_scores.json        # PageRank scores per paper
├── hybrid_weights.json         # hybrid model mixing weights
│
│   -- checkpoint files (safe to delete after pipeline finishes) --
├── _ckpt_oai_{set}_{year}.csv  # arXiv per-(set,year) checkpoints (21 files)
├── _ckpt_s2_pairs.csv          # S2 citation pairs checkpoint
├── _ckpt_s2_fields.json        # S2 fields-of-study checkpoint
└── _ckpt_s2_done.json          # S2 completed batch IDs
```

---

## Adding a new model

The UI auto-detects model artefact files at startup. No UI code changes needed.

1. Train your model in the notebook using the shared evaluation framework (Section 5).
2. Save the artefact to Drive with the expected filename:

   | Model | Expected file | Format |
   |---|---|---|
   | SVD | `svd_model.npz` | `np.savez(path, user_factors=..., item_factors=...)` |
   | PageRank | `pagerank_scores.json` | `{"arxiv_id": score, ...}` |
   | Hybrid | `hybrid_weights.json` | `{"alpha": ..., "beta": ..., ...}` |

3. Fill in the stub scoring function in `app.py` (clearly marked with comments).
4. Re-launch `app.py` - your model appears in the dropdown automatically.

---

## Evaluation framework

All models share the same 80/20 edge split (seed=256) and 1,000 fixed query papers
(seed=42). Metrics: Precision@K, Recall@K, NDCG@K, Hit Rate@K, MRR at K in {5, 10, 20}.

```python
# Per-paper interface (simple):
def my_score_fn(query_idx: int) -> np.ndarray:
    # return shape (n_papers,) array - higher = more recommended
    ...

evaluate_model("My Model", score_fn=my_score_fn)

# Batch interface (faster, preferred for matrix models):
def my_batch_fn(indices: list[int]) -> np.ndarray:
    # return shape (B, n_papers) array
    ...

evaluate_model("My Model", batch_score_fn=my_batch_fn, batch_size=100)
compare_models()   # prints side-by-side table + saves model_comparison.png
```

**TF-IDF baseline to beat:**

| Metric | @5 | @10 | @20 |
|:---|---:|---:|---:|
| Precision | 0.0440 | 0.0324 | 0.0237 |
| Recall | 0.0724 | 0.1023 | 0.1453 |
| NDCG | 0.0712 | 0.0801 | 0.0944 |
| Hit Rate | 0.1930 | 0.2630 | 0.3540 |
| MRR | 0.1139 | 0.1232 | 0.1293 |

---

## Config

All knobs live in the config cell at the top of `notebook.ipynb`:

| Variable | Default | Effect |
|---|---|---|
| `MIN_YEAR` / `MAX_YEAR` | 2018 / 2024 | Year range of papers to collect |
| `TARGET_CATS` | 6 ML/CS cats | arXiv categories to filter for |
| `OAI_WORKERS` | 21 | Parallel arXiv fetch threads (reduce if 503s appear) |
| `S2_RPS` | 0.9 | S2 requests/sec (raise to 9.0 with Tier 2 key) |
| `S2_WORKERS` | 1 | Parallel S2 threads (raise to 5 with Tier 2 key) |
| `MIN_DEGREE` | 2 | Min citation degree to keep a paper |
| `EVAL_SAMPLE` | 1000 | Query papers used for evaluation |

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add
models, report bugs, and submit pull requests.

---

## License

MIT - see [LICENSE.md](LICENSE.md).

---

## References

- arXiv OAI-PMH API: <https://info.arxiv.org/help/oa/index.html>
- Semantic Scholar API: <https://www.semanticscholar.org/product/api>
- arXiv dataset on Kaggle: <https://www.kaggle.com/datasets/Cornell-University/arxiv>
- unarXive (full text + citations): <https://zenodo.org/records/7752754>
