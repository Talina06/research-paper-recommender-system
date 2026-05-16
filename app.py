"""
ArXivists Citation Recommender — Gradio UI
CMPE 256, San José State University

Run locally:  python app.py
Run in Colab: !python app.py  (or paste the cells below)

The app loads papers.csv and citations.csv from DRIVE_DIR (or the local directory)
and exposes each registered model through a unified interface.
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config — point to wherever your CSVs live
# ---------------------------------------------------------------------------
DRIVE_DIR  = Path("/content/drive/MyDrive/cmpe_256_project_files")
LOCAL_DIR  = Path(".")                     # fallback when not in Colab
DATA_DIR   = DRIVE_DIR if DRIVE_DIR.exists() else LOCAL_DIR

TOP_K      = 10
ALL_CATS   = ["cs.LG", "cs.AI", "cs.CV", "cs.IR", "cs.CL", "stat.ML"]

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
print(f"loading data from {DATA_DIR} …")
df_papers = pd.read_csv(DATA_DIR / "papers.csv", dtype={"arxiv_id": str})
df_papers["text"] = df_papers["text"].fillna("")
df_papers["categories"] = df_papers["categories"].fillna("")

paper_idx_to_row = df_papers.set_index("paper_idx")
arxiv_to_idx     = dict(zip(df_papers["arxiv_id"], df_papers["paper_idx"]))

print(f"  {len(df_papers):,} papers loaded")

# ---------------------------------------------------------------------------
# Model registry
# Each entry: {"name": str, "inputs": list["text"|"categories"|"arxiv_id"],
#              "fn": callable(...) -> pd.DataFrame of top-K results}
# ---------------------------------------------------------------------------
MODELS: dict[str, dict] = {}


def register_model(name: str, inputs: list[str], fn):
    """Add a model to the registry."""
    MODELS[name] = {"name": name, "inputs": inputs, "fn": fn}


# ---------------------------------------------------------------------------
# Baseline: TF-IDF
# ---------------------------------------------------------------------------
print("building TF-IDF model …")
from sklearn.feature_extraction.text import TfidfVectorizer

_tfidf_vec = TfidfVectorizer(
    max_features=50_000,
    min_df=3,
    ngram_range=(1, 2),
    sublinear_tf=True,
    norm="l2",
)
_tfidf_mat = _tfidf_vec.fit_transform(df_papers["text"])
print("  TF-IDF ready")


def _tfidf_recommend(query_text: str, query_cats: list[str], k: int = TOP_K) -> pd.DataFrame:
    """
    Score every paper by cosine similarity to the query text.
    Optionally boost papers that share at least one category with the query.
    """
    query = query_text.strip()
    if not query and not query_cats:
        return pd.DataFrame(columns=["rank", "arxiv_id", "year", "categories", "title", "score"])

    # TF-IDF scores
    if query:
        q_vec   = _tfidf_vec.transform([query])
        scores  = (q_vec @ _tfidf_mat.T).toarray().ravel()
    else:
        scores = np.zeros(len(df_papers), dtype=float)

    # Category boost: +0.05 per shared category (small nudge, text still dominates)
    if query_cats:
        cat_set = set(query_cats)
        for i, row_cats in enumerate(df_papers["categories"]):
            overlap = cat_set & set(str(row_cats).split())
            scores[i] += 0.05 * len(overlap)

    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    rows = []
    for rank, idx in enumerate(top_idx, start=1):
        row = paper_idx_to_row.loc[idx]
        rows.append({
            "rank":       rank,
            "arxiv_id":   row["arxiv_id"],
            "year":       int(row["year"]),
            "categories": row["categories"].split()[0] if row["categories"] else "",
            "title":      row["title"],
            "score":      round(float(scores[idx]), 4),
        })
    return pd.DataFrame(rows)


register_model(
    name   = "TF-IDF (baseline)",
    inputs = ["text", "categories"],
    fn     = _tfidf_recommend,
)

# ---------------------------------------------------------------------------
# Stub loaders for future models — teammates fill these in
# ---------------------------------------------------------------------------

def _load_svd() -> bool:
    """Load SVD model artefacts if available."""
    svd_path = DATA_DIR / "svd_model.npz"
    if not svd_path.exists():
        return False
    data = np.load(svd_path)
    user_factors = data["user_factors"]   # (n_papers, k)
    item_factors = data["item_factors"]   # (n_papers, k)

    def _svd_recommend(query_text: str, query_cats: list[str], k: int = TOP_K) -> pd.DataFrame:
        # SVD needs an arxiv_id or paper_idx as query — text is not directly used.
        # For demo: return most similar papers to a random seed paper.
        # Sheetal: replace the query logic here.
        return pd.DataFrame(columns=["rank", "arxiv_id", "year", "categories", "title", "score"])

    register_model(
        name   = "SVD",
        inputs = ["arxiv_id"],
        fn     = _svd_recommend,
    )
    return True


def _load_pagerank() -> bool:
    """Load PageRank scores if available."""
    pr_path = DATA_DIR / "pagerank_scores.json"
    if not pr_path.exists():
        return False
    pr_scores = json.loads(pr_path.read_text())   # {arxiv_id: score}

    def _pagerank_recommend(query_text: str, query_cats: list[str], k: int = TOP_K) -> pd.DataFrame:
        # PageRank returns globally popular papers, optionally filtered by category.
        cat_set = set(query_cats) if query_cats else None
        rows = []
        for arxiv_id, score in sorted(pr_scores.items(), key=lambda x: -x[1]):
            if arxiv_id not in arxiv_to_idx:
                continue
            row = paper_idx_to_row.loc[arxiv_to_idx[arxiv_id]]
            if cat_set and not (cat_set & set(str(row["categories"]).split())):
                continue
            rows.append({
                "rank":       len(rows) + 1,
                "arxiv_id":   arxiv_id,
                "year":       int(row["year"]),
                "categories": row["categories"].split()[0] if row["categories"] else "",
                "title":      row["title"],
                "score":      round(score, 6),
            })
            if len(rows) >= k:
                break
        return pd.DataFrame(rows)

    register_model(
        name   = "PageRank",
        inputs = ["categories"],
        fn     = _pagerank_recommend,
    )
    return True


def _load_hybrid() -> bool:
    """Load hybrid model if available (SVD + TF-IDF + PageRank + category score)."""
    hybrid_path = DATA_DIR / "hybrid_weights.json"
    if not hybrid_path.exists():
        return False
    # Manjula: implement hybrid scoring here
    return True


# Try to load optional models
for _loader in [_load_svd, _load_pagerank, _load_hybrid]:
    try:
        _loader()
    except Exception as e:
        print(f"  optional model skipped: {e}")

print(f"models available: {list(MODELS.keys())}")

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def _model_inputs(model_name: str) -> tuple:
    """Return which input widgets are visible for the selected model."""
    if model_name not in MODELS:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    inp = MODELS[model_name]["inputs"]
    return (
        gr.update(visible="text"       in inp),
        gr.update(visible="categories" in inp),
        gr.update(visible="arxiv_id"   in inp),
    )


def _recommend(model_name: str, query_text: str, query_cats: list, arxiv_id: str) -> tuple:
    if model_name not in MODELS:
        return "Please select a model.", pd.DataFrame()

    model = MODELS[model_name]
    try:
        df = model["fn"](
            query_text = query_text or "",
            query_cats = query_cats or [],
            k          = TOP_K,
        )
    except Exception as e:
        return f"Error: {e}", pd.DataFrame()

    if df.empty:
        return "No results — try different input.", pd.DataFrame()

    summary = (
        f"**{len(df)} recommendations** from **{model_name}**"
        + (f" for query: *{query_text[:80]}…*" if query_text else "")
        + (f" | categories: {', '.join(query_cats)}" if query_cats else "")
    )
    return summary, df


with gr.Blocks(title="ArXivists Citation Recommender", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📄 ArXivists Citation Recommender
        **CMPE 256 — San José State University**
        *Talina Shrotriya · Sheetal Sattiraju · Manjula Ganesh*

        Enter a research topic, select arXiv categories, or both — then pick a model
        and click **Recommend** to get the top-10 papers you should cite.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices = list(MODELS.keys()),
                value   = list(MODELS.keys())[0],
                label   = "Model",
            )

            text_input = gr.Textbox(
                label       = "Query text (title / abstract / description)",
                placeholder = "e.g. graph neural networks for node classification",
                lines       = 4,
                visible     = True,
            )

            cat_input = gr.CheckboxGroup(
                choices = ALL_CATS,
                label   = "arXiv categories (optional boost)",
                visible = True,
            )

            arxiv_input = gr.Textbox(
                label       = "arXiv ID (for ID-based models)",
                placeholder = "e.g. 2301.12345",
                visible     = False,
            )

            recommend_btn = gr.Button("Recommend", variant="primary")

        with gr.Column(scale=2):
            summary_md = gr.Markdown("Results will appear here.")
            results_table = gr.Dataframe(
                headers = ["rank", "arxiv_id", "year", "categories", "title", "score"],
                wrap    = True,
            )

    # Wire up model switch → toggle visible inputs
    model_dropdown.change(
        fn      = _model_inputs,
        inputs  = [model_dropdown],
        outputs = [text_input, cat_input, arxiv_input],
    )

    # Wire up recommend button
    recommend_btn.click(
        fn      = _recommend,
        inputs  = [model_dropdown, text_input, cat_input, arxiv_input],
        outputs = [summary_md, results_table],
    )

    gr.Markdown(
        """
        ---
        **Tips**
        - *TF-IDF*: enter any free-form research description; add categories for a small extra boost.
        - *SVD / PageRank / Hybrid*: available once Sheetal & Manjula upload their model artefacts to Drive.
        - Results are ranked by model score (higher = more similar / more relevant).
        """
    )


if __name__ == "__main__":
    demo.launch(share=True)
