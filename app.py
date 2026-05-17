"""
ArXivists Citation Recommender — Gradio UI
CMPE 256, San Jose State University

Run locally:
    python app.py --data-dir /path/to/downloaded/files
    # open http://127.0.0.1:7860

Run in Colab (Section 7 of notebook.ipynb):
    %run /content/app.py

Files needed in data-dir:
    papers.csv               required — paper metadata
    tfidf_vectorizer.pkl     recommended — skips ~60s rebuild on startup
    tfidf_matrix.npz         recommended — skips matrix transform on startup
    svd_model.npz            optional — auto-registered if present
    pagerank_scores.json     optional — auto-registered if present
    hybrid_weights.json      optional — auto-registered if present
"""

import argparse
import json
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data directory — Drive in Colab, or --data-dir / cwd locally
# ---------------------------------------------------------------------------
_DRIVE_DIR = Path("/content/drive/MyDrive/cmpe_256_project_files")

def _resolve_data_dir() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", default=None)
    args, _ = parser.parse_known_args()
    if args.data_dir:
        p = Path(args.data_dir).expanduser().resolve()
        if not p.exists():
            sys.exit(f"ERROR: --data-dir {p} does not exist")
        return p
    if _DRIVE_DIR.exists():
        return _DRIVE_DIR
    return Path(".")

DATA_DIR = _resolve_data_dir()
TOP_K    = 10
ALL_CATS = ["cs.LG", "cs.AI", "cs.CV", "cs.IR", "cs.CL", "stat.ML"]

# ---------------------------------------------------------------------------
# Load paper metadata
# ---------------------------------------------------------------------------
_papers_path = DATA_DIR / "papers.csv"
if not _papers_path.exists():
    sys.exit(
        f"ERROR: papers.csv not found in {DATA_DIR}\n"
        "Download it from the shared Drive folder and re-run:\n"
        "  python app.py --data-dir /path/to/files"
    )

print(f"loading data from {DATA_DIR} ...")
df_papers = pd.read_csv(_papers_path, dtype={"arxiv_id": str})
df_papers["text"]       = df_papers["text"].fillna("")
df_papers["categories"] = df_papers["categories"].fillna("")

paper_idx_to_row = df_papers.set_index("paper_idx")
arxiv_to_idx     = dict(zip(df_papers["arxiv_id"], df_papers["paper_idx"]))
print(f"  {len(df_papers):,} papers loaded")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS: dict[str, dict] = {}

def register_model(name: str, inputs: list[str], fn, score_label: str = "score", score_info: str = ""):
    MODELS[name] = {"name": name, "inputs": inputs, "fn": fn, "score_label": score_label, "score_info": score_info}

_SCORE_INFO = {
    "TF-IDF (baseline)": """\
**Score:** cosine similarity (0 to 1)

| Range | Signal |
|---|---|
| > 0.30 | Strong - very similar topic and vocabulary |
| 0.15 - 0.30 | Good - clearly related work |
| 0.05 - 0.15 | Weak - some shared terms, different focus |
| < 0.05 | Noise - likely a false positive |

Tip: use specific technical terms (e.g. *masked language modeling*, *contrastive learning*) for higher scores.
""",
    "SVD": """\
**Score:** latent factor dot product similarity

Higher = more co-citation overlap in the training graph.
No fixed upper bound - use relative ranking, not absolute value.
""",
    "PageRank": """\
**Score:** PageRank citation importance

Reflects how often a paper is cited by other highly-cited papers.
Results are filtered by selected categories, ranked by global importance.
""",
    "Hybrid": """\
**Score:** weighted combination of TF-IDF + SVD + PageRank

Balances content similarity, co-citation structure, and citation importance.
""",
}

def _arxiv_title_link(arxiv_id: str, title: str) -> str:
    return (
        f'<a href="https://arxiv.org/abs/{arxiv_id}" target="_blank" '
        f'style="color:#2563eb;text-decoration:underline;cursor:pointer;font-weight:500;">'
        f'{title}</a>'
    )

# ---------------------------------------------------------------------------
# TF-IDF — load pre-built artifacts if available, otherwise rebuild
# ---------------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import scipy.sparse as sp

_vec_path = DATA_DIR / "tfidf_vectorizer.pkl"
_mat_path = DATA_DIR / "tfidf_matrix.npz"

if _vec_path.exists() and _mat_path.exists():
    print("loading TF-IDF from saved artifacts ...")
    with open(_vec_path, "rb") as _f:
        _tfidf_vec = pickle.load(_f)
    _tfidf_mat = sp.load_npz(_mat_path)
    print(f"  loaded: {_tfidf_mat.shape[0]:,} papers x {_tfidf_mat.shape[1]:,} terms")

elif _vec_path.exists():
    print("loading TF-IDF vectorizer, transforming matrix ...")
    with open(_vec_path, "rb") as _f:
        _tfidf_vec = pickle.load(_f)
    _tfidf_mat = _tfidf_vec.transform(df_papers["text"])
    print(f"  done: {_tfidf_mat.shape[0]:,} papers x {_tfidf_mat.shape[1]:,} terms")

else:
    print("building TF-IDF from scratch (no saved artifacts found, ~60s) ...")
    _tfidf_vec = TfidfVectorizer(
        max_features=50_000,
        min_df=3,
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
    )
    _tfidf_mat = _tfidf_vec.fit_transform(df_papers["text"])
    print(f"  done: {_tfidf_mat.shape[0]:,} papers x {_tfidf_mat.shape[1]:,} terms")


_COLS = ["rank", "arxiv_id", "year", "categories", "title", "score"]

def _tfidf_recommend(query_text: str, query_cats: list[str], k: int = TOP_K) -> pd.DataFrame:
    query = query_text.strip()
    if not query and not query_cats:
        return pd.DataFrame(columns=_COLS)

    if query:
        q_vec  = _tfidf_vec.transform([query])
        scores = (q_vec @ _tfidf_mat.T).toarray().ravel()
    else:
        scores = np.zeros(len(df_papers), dtype=float)

    # Small category boost (+0.05 per shared category); text still dominates
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
        aid = row["arxiv_id"]
        rows.append({
            "rank":       rank,
            "arxiv_id":   aid,
            "year":       int(row["year"]),
            "categories": row["categories"].split()[0] if row["categories"] else "",
            "title":      _arxiv_title_link(aid, row["title"]),
            "score":      round(float(scores[idx]), 4),
        })
    return pd.DataFrame(rows)


register_model(
    name="TF-IDF (baseline)",
    inputs=["text", "categories"],
    fn=_tfidf_recommend,
    score_label="cosine similarity (0-1)",
    score_info=_SCORE_INFO["TF-IDF (baseline)"],
)

# ---------------------------------------------------------------------------
# Optional model loaders — teammates fill in the scoring logic
# ---------------------------------------------------------------------------

def _load_svd() -> bool:
    svd_path = DATA_DIR / "svd_model.npz"
    if not svd_path.exists():
        return False
    data         = np.load(svd_path)
    user_factors = data["user_factors"]   # (n_papers, k)
    item_factors = data["item_factors"]   # (n_papers, k)

    def _svd_recommend(query_text: str, query_cats: list[str], k: int = TOP_K) -> pd.DataFrame:
        # Sheetal: implement SVD query logic here.
        # user_factors and item_factors are available in this scope.
        return pd.DataFrame(columns=_COLS)

    register_model(name="SVD", inputs=["arxiv_id"], fn=_svd_recommend, score_label="latent factor similarity", score_info=_SCORE_INFO["SVD"])
    return True


def _load_pagerank() -> bool:
    pr_path = DATA_DIR / "pagerank_scores.json"
    if not pr_path.exists():
        return False
    pr_scores = json.loads(pr_path.read_text())   # {arxiv_id: score}

    def _pagerank_recommend(query_text: str, query_cats: list[str], k: int = TOP_K) -> pd.DataFrame:
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
                "title":      _arxiv_title_link(arxiv_id, row["title"]),
                "score":      round(score, 6),
            })
            if len(rows) >= k:
                break
        return pd.DataFrame(rows)

    register_model(name="PageRank", inputs=["categories"], fn=_pagerank_recommend, score_label="PageRank score (citation importance)", score_info=_SCORE_INFO["PageRank"])
    return True


def _load_hybrid() -> bool:
    hybrid_path = DATA_DIR / "hybrid_weights.json"
    if not hybrid_path.exists():
        return False
    # Manjula: implement hybrid scoring here.
    return True


for _loader in [_load_svd, _load_pagerank, _load_hybrid]:
    try:
        loaded = _loader()
        if loaded:
            print(f"  optional model loaded: {_loader.__name__}")
    except Exception as e:
        print(f"  optional model skipped ({_loader.__name__}): {e}")

print(f"models available: {list(MODELS.keys())}")

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def _model_inputs(model_name: str) -> tuple:
    if model_name not in MODELS:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), ""
    m = MODELS[model_name]
    inp = m["inputs"]
    return (
        gr.update(visible="text"       in inp),
        gr.update(visible="categories" in inp),
        gr.update(visible="arxiv_id"   in inp),
        m.get("score_info", ""),
    )


def _recommend(model_name: str, query_text: str, query_cats: list, arxiv_id: str) -> tuple:
    if model_name not in MODELS:
        return "Please select a model.", pd.DataFrame()

    model = MODELS[model_name]
    try:
        df = model["fn"](
            query_text=query_text or "",
            query_cats=query_cats or [],
            k=TOP_K,
        )
    except Exception as e:
        return f"Error: {e}", pd.DataFrame()

    if df.empty:
        return "No results - try different input.", pd.DataFrame()

    summary = (
        f"**{len(df)} recommendations** from **{model_name}**"
        + (f" for query: *{query_text[:80]}...*" if query_text else "")
        + (f" | categories: {', '.join(query_cats)}" if query_cats else "")
    )
    return summary, df


with gr.Blocks(title="ArXivists Citation Recommender", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # ArXivists Research Paper Recommender

        Enter a research topic, select arXiv categories, or both - then pick a model
        and click **Recommend** to get the top-10 papers you should cite.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=list(MODELS.keys()),
                value=list(MODELS.keys())[0],
                label="Model",
            )
            text_input = gr.Textbox(
                label="Query text (title / abstract / description)",
                value=(
                    "We propose a self-attention transformer architecture pre-trained on large "
                    "unlabeled text corpora using masked language modeling. The model learns "
                    "deep bidirectional contextual representations and is fine-tuned on "
                    "downstream NLP tasks including text classification, named entity "
                    "recognition, and question answering, achieving state-of-the-art results "
                    "on the GLUE benchmark."
                ),
                lines=4,
                visible=True,
            )
            cat_input = gr.CheckboxGroup(
                choices=ALL_CATS,
                label="arXiv categories (optional boost)",
                visible=True,
            )
            arxiv_input = gr.Textbox(
                label="arXiv ID (for ID-based models)",
                placeholder="e.g. 2301.12345",
                visible=False,
            )
            recommend_btn = gr.Button("Recommend", variant="primary")
            score_info_md = gr.Markdown(
                value=MODELS[list(MODELS.keys())[0]].get("score_info", ""),
                visible=True,
            )

        with gr.Column(scale=2):
            summary_md    = gr.Markdown("Results will appear here.")
            results_table = gr.Dataframe(
                headers=["rank", "arxiv_id", "year", "categories", "title", "score"],
                datatype=["number", "str", "number", "str", "html", "number"],
                wrap=True,
            )

    model_dropdown.change(
        fn=_model_inputs,
        inputs=[model_dropdown],
        outputs=[text_input, cat_input, arxiv_input, score_info_md],
    )
    recommend_btn.click(
        fn=_recommend,
        inputs=[model_dropdown, text_input, cat_input, arxiv_input],
        outputs=[summary_md, results_table],
    )

    gr.Markdown(
        """
        ---
        **Tips**
        - *TF-IDF*: enter any free-form research description; add categories for a small extra boost.
        - *SVD / PageRank / Hybrid*: appear in the dropdown once their artefact files are present in data-dir.
        - Results are ranked by model score (higher = more similar / more relevant).
        """
    )


if __name__ == "__main__":
    demo.launch(share=False)
