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
   pip install gradio scikit-learn pandas numpy scipy matplotlib torch gdown
   ```
4. Download the required data artifacts:
   ```bash
   python download_data.py --required-only   # papers.csv, citations.csv, TF-IDF files
   python download_data.py                   # all model artifacts
   ```

---

## Adding a new recommendation model

### Step 1 — Train in the notebook

Open `notebook.ipynb`. Sections 1–5 build the dataset and evaluation framework.
Add your model as a new subsection in Section 7.

Use `_get_train_edges("My Model")` to get the appropriate training edges (warm papers
only, or all papers, depending on `COLD_START_PER_MODEL`):

```python
# Register your model in COLD_START_PER_MODEL (Section 5 eval framework cell)
COLD_START_PER_MODEL["My Model"] = True  # True = warm papers only

# Train using _get_train_edges
my_train_edges = _get_train_edges("My Model")
```

Evaluate with the shared framework so results are directly comparable:

```python
# Batch scoring (preferred for matrix/embedding models — much faster):
def my_batch_score_fn(indices: list[int]) -> np.ndarray:
    # indices: list of paper_idx values
    # return: np.ndarray shape (len(indices), n_papers) — higher = more recommended
    ...

evaluate_model("My Model", batch_score_fn=my_batch_score_fn, batch_size=100)

# Per-paper scoring (for models where batching is awkward, e.g. PPR):
def my_score_fn(query_idx: int) -> np.ndarray:
    # return shape (n_papers,) array
    ...

evaluate_model("My Model", score_fn=my_score_fn)
compare_models()
```

### Step 2 — Save the artifact to Drive

Run the export cell (Section 8). For a new model, add a save line before the export loop:

```python
# Example: save a numpy embedding matrix
np.save(OUTPUT_DIR / "my_embeddings.npy", my_embeddings)

# Example: save a PyTorch model
torch.save(my_model.state_dict(), OUTPUT_DIR / "my_model.pt")
```

### Step 3 — Register in `app.py`

Add a loader block. Models use a text query as input — use `_text_to_paper_idx()` to
find the closest paper by TF-IDF before applying graph/neural scoring:

```python
_my_path = DATA_DIR / "my_model.pt"
if _my_path.exists():
    # load artifact
    _my_data = np.load(_my_path)   # or torch.load, etc.

    def _my_recommend(query_text: str, query_cats: list[str], arxiv_id: str, k: int = TOP_K) -> pd.DataFrame:
        if not query_text.strip():
            return pd.DataFrame()

        # For graph/neural models: anchor on closest TF-IDF paper
        idx = _text_to_paper_idx(query_text)
        if idx is None:
            return pd.DataFrame()

        scores = ...  # compute scores over all papers, shape (n_papers,)
        scores[idx] = -np.inf   # exclude query paper itself

        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return _build_result_df(top_idx, scores)

    register_model(
        name="My Model",
        inputs=["text"],          # "text", "categories", or "arxiv_id"
        fn=_my_recommend,
        score_label="my score label",
        score_info="**Score:** description of what the score means.",
    )
```

`_build_result_df` returns a DataFrame with columns:
`rank`, `arxiv_id`, `year`, `categories`, `title`, `abstract`, `score`.

If the artifact file is missing, the block is simply skipped — the model won't appear
in the dropdown. No error is raised.

### Step 4 — Add to `download_data.py`

Add an entry to the `FILES` list with the Google Drive file ID:

```python
("my_model.pt", "GOOGLE_DRIVE_FILE_ID", False, "My model weights"),
```

And add it to the `model_map` availability check:

```python
(["my_model.pt"], "My Model"),
```

### Step 5 — Test locally

```bash
python app.py --data-dir ./data
# visit http://127.0.0.1:7860
# select your model from the dropdown and verify recommendations look reasonable
```

### Step 6 — Open a pull request

See [Submitting a pull request](#submitting-a-pull-request).

---

## Reporting bugs

Open an issue on GitHub with:

- A short description of what went wrong.
- Steps to reproduce (notebook cell or UI action, input values, error message).
- Environment info: Python version, key package versions (`gradio`, `torch`, `sklearn`), Colab or local.

If the bug is in the data pipeline (arXiv fetch or S2 fetch), include the checkpoint
file names present when the error occurred and the output of the failing cell.

---

## Submitting a pull request

1. Create a branch from `main`:
   ```bash
   git checkout -b feature/my-model-name
   ```
2. Make your changes. Keep commits focused — one logical change per commit.
3. Verify the notebook runs without errors from Section 5 onwards.
4. Verify the app loads and your model appears in the dropdown.
5. Push and open a PR against `main` on the upstream repo.
6. Include your evaluation numbers from `compare_models()` in the PR description
   so reviewers can see the performance impact.

---

## Code style

- Python: follow PEP 8. Functions and variables use `snake_case`.
- Keep notebook cells focused — one logical step per cell.
- Do not commit binary files (model weights, CSVs) to the repo. Those live in Drive.
- Use `OUTPUT_DIR` / `DATA_DIR` for all file paths — no hard-coded paths in functions.
- New model loaders in `app.py` must degrade gracefully when the artifact is missing
  (skip silently, do not raise).
- Comments should explain *why*, not *what*. Avoid redundant comments that restate the code.
