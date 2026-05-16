# Contributing to ArXivists Citation Recommender

Thank you for your interest in contributing! This guide covers everything you need
to add a new model, fix a bug, or improve the project.

---

## Table of contents

- [Getting started](#getting-started)
- [Adding a new recommendation model](#adding-a-new-recommendation-model)
- [Reporting bugs](#reporting-bugs)
- [Submitting a pull request](#submitting-a-pull-request)
- [Code style](#code-style)

---

## Getting started

1. Fork the repository on GitHub.
2. Clone your fork:
   ```bash
   git clone https://github.com/<your-username>/research-paper-recommender-system.git
   cd research-paper-recommender-system
   ```
3. Install dependencies:
   ```bash
   pip install gradio scikit-learn pandas numpy scipy matplotlib
   ```
4. Download `papers.csv` and `citations.csv` from the shared Drive folder and place
   them in the project root (same directory as `app.py`).

---

## Adding a new recommendation model

This is the most common contribution. The project is designed so that adding a model
requires no changes to shared infrastructure - just train, save, and plug in.

### Step 1 - Train the model in the notebook

Open `notebook.ipynb`. Sections 1-4 build the dataset and Section 5
sets up the shared evaluation framework. Write your model in a new Section (7+) after
the existing baseline.

Use the shared evaluation helpers so your results are directly comparable:

```python
def my_batch_score_fn(indices: list[int]) -> np.ndarray:
    # indices: list of paper_idx values (integers)
    # return: np.ndarray of shape (len(indices), n_papers)
    # higher score = more recommended
    ...

results = evaluate_model("My Model Name", batch_score_fn=my_batch_score_fn, batch_size=100)
compare_models()
```

For models that can only score one paper at a time, use `score_fn` instead:

```python
def my_score_fn(query_idx: int) -> np.ndarray:
    # return shape (n_papers,) array
    ...

evaluate_model("My Model Name", score_fn=my_score_fn)
```

Both functions receive integer `paper_idx` values, not arXiv IDs.

### Step 2 - Save the model artefact to Drive

Save to `cmpe_256_project_files/` using the expected filename so `app.py` picks it
up automatically:

| Model | File | Format |
|---|---|---|
| SVD | `svd_model.npz` | `np.savez(path, user_factors=U, item_factors=V)` where U, V are (n_papers, k) |
| PageRank | `pagerank_scores.json` | `{"arxiv_id": float_score, ...}` |
| Hybrid | `hybrid_weights.json` | `{"alpha": float, "beta": float, ...}` |

### Step 3 - Implement the scoring stub in `app.py`

Each model has a stub loader (`_load_svd`, `_load_pagerank`, `_load_hybrid`) that
is called at startup. Fill in the `_recommend` function inside the loader:

```python
def _load_svd() -> bool:
    svd_path = DATA_DIR / "svd_model.npz"
    if not svd_path.exists():
        return False
    data         = np.load(svd_path)
    user_factors = data["user_factors"]   # (n_papers, k)
    item_factors = data["item_factors"]   # (n_papers, k)

    def _svd_recommend(query_text: str, query_cats: list[str], k: int = TOP_K) -> pd.DataFrame:
        # implement your scoring logic here
        # return a DataFrame with columns: rank, arxiv_id, year, categories, title, score
        ...

    register_model(name="SVD", inputs=["arxiv_id"], fn=_svd_recommend)
    return True
```

The returned DataFrame must have exactly these columns in this order:
`rank`, `arxiv_id`, `year`, `categories`, `title`, `score`.

### Step 4 - Test locally

```bash
python app.py
# visit http://127.0.0.1:7860
# select your model from the dropdown and verify recommendations look reasonable
```

### Step 5 - Open a pull request

See [Submitting a pull request](#submitting-a-pull-request).

---

## Reporting bugs

Open an issue on GitHub with:

- A short description of what went wrong.
- Steps to reproduce (notebook cell, input values, error message).
- Environment info: Python version, key package versions, Colab or local.

If the bug is in the data pipeline (arXiv fetch or S2 fetch), include the
checkpoint file names that were present when the error occurred.

---

## Submitting a pull request

1. Create a branch from `main`:
   ```bash
   git checkout -b feature/my-model-name
   ```
2. Make your changes. Keep commits focused - one logical change per commit.
3. Verify the notebook runs end-to-end without errors (at least from Section 5
   onwards if the full pipeline is too slow locally).
4. Push and open a PR against `main` on the upstream repo.
5. In the PR description, include your evaluation numbers from `compare_models()`
   so reviewers can see the performance gain.

---

## Code style

- Python: follow PEP 8. Functions and variables use `snake_case`.
- Keep notebook cells focused. One logical step per cell.
- Do not commit large binary files (model weights, CSVs) to the repo. Those live
  in the shared Drive folder.
- No hard-coded file paths inside functions. Use `DATA_DIR` from the config cell.
- New model loaders in `app.py` should degrade gracefully if the artefact file is
  missing (return `False` without raising).
