"""
ArXivists Citation Recommender — Gradio UI
CMPE 256, San Jose State University

Run locally:
    python app.py --data-dir /path/to/downloaded/files
    # open http://127.0.0.1:7860

Run in Colab (Section 8 of notebook):
    %run /content/app.py

Files needed in data-dir:
    papers.csv               required
    tfidf_vectorizer.pkl     required for TF-IDF / Hybrid models
    tfidf_matrix.npz         required for TF-IDF / Hybrid models
    paper_embeddings.npy     optional — enables SVD + Hybrid TF-IDF+SVD
    pagerank_vector.npy      optional — enables PageRank
    gmf_model.pt             optional — enables GMF
    neumf_model.pt           optional — enables NeuMF
"""

import argparse
import json
import sys
import pickle
import warnings
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Data directory
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
df_papers["text"]       = df_papers.get("text", pd.Series([""] * len(df_papers))).fillna("")
df_papers["categories"] = df_papers["categories"].fillna("")
df_papers["title"]      = df_papers["title"].fillna("")

paper_idx_to_row = df_papers.set_index("paper_idx")
arxiv_to_idx     = dict(zip(df_papers["arxiv_id"], df_papers["paper_idx"]))
N_PAPERS         = len(df_papers)

# Build citation adjacency for PPR — loads citations.csv if present
from collections import defaultdict
_ppr_adj: dict[int, list[int]] = defaultdict(list)
_cit_path = DATA_DIR / "citations.csv"
if _cit_path.exists():
    print("building citation graph for PPR ...")
    _cit_df = pd.read_csv(_cit_path, usecols=["source_idx", "target_idx"])
    for src, tgt in zip(_cit_df["source_idx"], _cit_df["target_idx"]):
        _ppr_adj[int(src)].append(int(tgt))
    print(f"  {len(_ppr_adj):,} papers with outgoing citations")
print(f"  {N_PAPERS:,} papers loaded")

# Same domains as OAI_SETS in the notebook config
ALL_CATS = ["cs", "math", "physics", "stat", "eess", "q-bio", "q-fin", "econ"]

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS: dict[str, dict] = {}

def register_model(name: str, inputs: list[str], fn, score_label: str = "score", score_info: str = ""):
    MODELS[name] = {
        "name": name, "inputs": inputs, "fn": fn,
        "score_label": score_label, "score_info": score_info,
    }

def _arxiv_link(arxiv_id: str, title: str) -> str:
    return (
        f'<a href="https://arxiv.org/abs/{arxiv_id}" target="_blank" '
        f'style="color:#2563eb;text-decoration:underline;font-weight:500;">'
        f'{title}</a>'
    )

def _build_result_df(top_idx: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    rows = []
    for rank, idx in enumerate(top_idx, start=1):
        idx = int(idx)
        if idx < 0 or idx >= len(scores) or idx not in paper_idx_to_row.index:
            continue
        row = paper_idx_to_row.loc[idx]
        aid = str(row["arxiv_id"])
        score = float(scores[idx])
        if not np.isfinite(score):
            continue
        if score < 0.001:
            score_str = f"{score:.6f}"
        else:
            score_str = str(round(score, 4))
        rows.append({
            "rank":       rank,
            "arxiv_id":   aid,
            "year":       int(row.get("year", 0)),
            "categories": str(row["categories"]) if row["categories"] else "",
            "title":      _arxiv_link(aid, row["title"]),
            "abstract":   str(row.get("abstract", ""))[:300],
            "score":      score_str,
        })
    return pd.DataFrame(rows)

def _make_pie_chart(df_results: pd.DataFrame) -> plt.Figure:
    # Count top-level domain for every category on every displayed paper.
    # A paper with "cs.LG cs.AI stat.ML" contributes to cs and stat.
    domain_counts: dict[str, int] = {}
    for idx in df_results.index:
        paper_idx = int(df_results.loc[idx, "rank"]) - 1  # rank is 1-based
        # Use the full categories string from paper_idx_to_row via arxiv_id
        aid = df_results.loc[idx, "arxiv_id"]
        if aid in arxiv_to_idx:
            pidx = arxiv_to_idx[aid]
            if pidx in paper_idx_to_row.index:
                cats_str = str(paper_idx_to_row.loc[pidx, "categories"])
                for cat in cats_str.split():
                    domain = cat.split(".")[0] if "." in cat else cat
                    if domain in set(ALL_CATS):
                        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    if not domain_counts:
        # fallback: use the categories column already in df_results
        domain_counts = df_results["categories"].apply(
            lambda c: c.split(".")[0] if "." in str(c) else str(c)
        ).value_counts().to_dict()

    labels = list(domain_counts.keys())
    values = list(domain_counts.values())

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        values,
        labels=labels,
        autopct="%1.0f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        textprops={"fontsize": 9},
    )
    ax.set_title(f"Domain Distribution\n({len(df_results)} recommendations)", fontsize=10, pad=10)
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# TF-IDF — load or rebuild
# ---------------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

_vec_path = DATA_DIR / "tfidf_vectorizer.pkl"
_mat_path = DATA_DIR / "tfidf_matrix.npz"
_tfidf_vec = None
_tfidf_mat = None

if _vec_path.exists() and _mat_path.exists():
    print("loading TF-IDF artifacts ...")
    with open(_vec_path, "rb") as _f:
        _tfidf_vec = pickle.load(_f)
    _tfidf_mat = sp.load_npz(_mat_path)
    print(f"  {_tfidf_mat.shape[0]:,} papers x {_tfidf_mat.shape[1]:,} terms")
elif _vec_path.exists():
    print("transforming TF-IDF matrix from vectorizer ...")
    with open(_vec_path, "rb") as _f:
        _tfidf_vec = pickle.load(_f)
    _tfidf_mat = _tfidf_vec.transform(df_papers["text"])
else:
    print("building TF-IDF from scratch (~60s) ...")
    _tfidf_vec = TfidfVectorizer(max_features=50_000, min_df=3, ngram_range=(1, 2),
                                  sublinear_tf=True, norm="l2")
    _tfidf_mat = _tfidf_vec.fit_transform(df_papers["text"])
    print(f"  {_tfidf_mat.shape[0]:,} papers x {_tfidf_mat.shape[1]:,} terms")

def _tfidf_recommend(query_text: str, query_cats: list[str], arxiv_id: str, k: int = TOP_K) -> pd.DataFrame:
    if not query_text.strip() and not query_cats:
        return pd.DataFrame()
    q_vec  = _tfidf_vec.transform([query_text.strip() or " ".join(query_cats)])
    scores = (q_vec @ _tfidf_mat.T).toarray().ravel()
    n_scores = len(scores)
    if query_cats:
        cat_set = set(query_cats)
        for i, row_cats in enumerate(df_papers["categories"]):
            if i >= n_scores:
                break
            paper_prefixes = {c.split(".")[0] for c in str(row_cats).split()}
            scores[i] += 0.05 * len(cat_set & paper_prefixes)
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return _build_result_df(top_idx, scores)

register_model(
    name="TF-IDF",
    inputs=["text", "categories"],
    fn=_tfidf_recommend,
    score_label="cosine similarity",
    score_info=(
        "**Score:** cosine similarity (0-1). "
        "Higher = more vocabulary overlap with query. "
        "Use specific technical terms for stronger matches."
    ),
)

def _text_to_paper_idx(query_text: str) -> int | None:
    """Find the closest paper in the dataset to query_text using TF-IDF."""
    if not query_text.strip() or _tfidf_mat is None:
        return None
    q_vec = _tfidf_vec.transform([query_text.strip()])
    scores = (q_vec @ _tfidf_mat.T).toarray().ravel()
    return int(np.argmax(scores))

# ---------------------------------------------------------------------------
# SVD — paper_embeddings.npy
# ---------------------------------------------------------------------------
_emb_path = DATA_DIR / "paper_embeddings.npy"
_paper_embeddings = None

if _emb_path.exists():
    print("loading SVD embeddings ...")
    _paper_embeddings = np.load(_emb_path)
    print(f"  {_paper_embeddings.shape[0]:,} papers x {_paper_embeddings.shape[1]} dims")

    _svd_n = _paper_embeddings.shape[0]

    def _svd_recommend(query_text: str, query_cats: list[str], arxiv_id: str, k: int = TOP_K) -> pd.DataFrame:
        idx = _text_to_paper_idx(query_text)
        if idx is None or idx >= _svd_n:
            return pd.DataFrame()
        q_emb  = _paper_embeddings[idx]
        scores = _paper_embeddings @ q_emb
        scores[idx] = -np.inf
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return _build_result_df(top_idx, scores)

    register_model(
        name="SVD",
        inputs=["text"],
        fn=_svd_recommend,
        score_label="latent factor similarity",
        score_info="**Score:** finds the closest paper to your abstract via TF-IDF, then ranks by citation-graph SVD similarity.",
    )

# ---------------------------------------------------------------------------
# PageRank — pagerank_vector.npy
# ---------------------------------------------------------------------------
_pr_path = DATA_DIR / "pagerank_vector.npy"

if _pr_path.exists():
    print("loading PageRank scores ...")
    _pr_vector = np.load(_pr_path)

    _pr_n = _pr_vector.shape[0]

    def _pagerank_recommend(query_text: str, query_cats: list[str], arxiv_id: str, k: int = TOP_K) -> pd.DataFrame:
        scores = _pr_vector[:N_PAPERS].copy()  # clamp to actual paper count
        if query_cats:
            # match on domain prefix: "cs" matches "cs.LG", "cs.AI", etc.
            cat_set = set(query_cats)
            for i, row_cats in enumerate(df_papers["categories"]):
                if i >= len(scores):
                    break
                paper_prefixes = {c.split(".")[0] for c in str(row_cats).split()}
                if not (cat_set & paper_prefixes):
                    scores[i] = -np.inf
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return _build_result_df(top_idx, scores)

    register_model(
        name="PageRank",
        inputs=["categories"],
        fn=_pagerank_recommend,
        score_label="PageRank citation importance",
        score_info="**Score:** global PageRank on the citation graph. Higher = cited by more influential papers.",
    )

# ---------------------------------------------------------------------------
# PPR-MC (requires citations.csv)
# text → TF-IDF best match → PPR walks from that paper
# ---------------------------------------------------------------------------
if _ppr_adj and _tfidf_mat is not None:
    import random as _random

    _PPR_ALPHA    = 0.85
    _PPR_WALKS    = 200
    _PPR_WALK_LEN = 30

    def _ppr_recommend(query_text: str, query_cats: list[str], arxiv_id: str, k: int = TOP_K) -> pd.DataFrame:
        if not query_text.strip():
            return pd.DataFrame()

        # Step 1: find best matching paper in dataset via TF-IDF
        anchor = _text_to_paper_idx(query_text)
        if anchor is None:
            return pd.DataFrame()

        # Step 2: run PPR walks from that paper through the real citation graph
        visits: dict[int, float] = defaultdict(float)
        for _ in range(_PPR_WALKS):
            node = anchor
            for _ in range(_PPR_WALK_LEN):
                visits[node] += 1.0
                if _random.random() > _PPR_ALPHA:
                    node = anchor
                    continue
                neighbors = _ppr_adj.get(node, [])
                node = _random.choice(neighbors) if neighbors else anchor

        n = _tfidf_mat.shape[0]
        total = sum(visits.values())
        scores = np.zeros(n, dtype=np.float32)
        for nd, count in visits.items():
            if nd < n:
                scores[nd] = count / (total + 1e-10)
        scores[anchor] = -np.inf  # exclude the anchor paper itself

        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return _build_result_df(top_idx, scores)

    register_model(
        name="PPR-MC",
        inputs=["text"],
        fn=_ppr_recommend,
        score_label="PPR visit frequency",
        score_info=(
            "**Score:** Personalized PageRank on the citation graph. "
            "Your abstract is matched to the closest paper via TF-IDF, "
            "then PPR walks from that paper surface its citation neighborhood."
        ),
    )

# ---------------------------------------------------------------------------
# TF-IDF + PPR-MC Ensemble (requires citations.csv + tfidf artifacts)
# ---------------------------------------------------------------------------
if _ppr_adj and _tfidf_mat is not None:
    _ENS_ALPHA = 0.5  # weight on PPR-MC; (1 - alpha) on TF-IDF

    def _ensemble_recommend(query_text: str, query_cats: list[str], arxiv_id: str, k: int = TOP_K) -> pd.DataFrame:
        if not query_text.strip():
            return pd.DataFrame()

        n = _tfidf_mat.shape[0]

        # TF-IDF scores
        q_vec = _tfidf_vec.transform([query_text.strip()])
        tfidf_s = (q_vec @ _tfidf_mat.T).toarray().ravel().astype(np.float64)

        # PPR scores — anchor on best TF-IDF match
        anchor = int(np.argmax(tfidf_s))
        visits: dict[int, float] = defaultdict(float)
        for _ in range(_PPR_WALKS):
            node = anchor
            for _ in range(_PPR_WALK_LEN):
                visits[node] += 1.0
                if _random.random() > _PPR_ALPHA:
                    node = anchor
                    continue
                neighbors = _ppr_adj.get(node, [])
                node = _random.choice(neighbors) if neighbors else anchor
        total = sum(visits.values())
        ppr_s = np.zeros(n, dtype=np.float64)
        for nd, count in visits.items():
            if nd < n:
                ppr_s[nd] = count / (total + 1e-10)

        # min-max normalise then blend
        def _norm(v):
            lo, hi = v.min(), v.max()
            return (v - lo) / (hi - lo + 1e-10)

        scores = _ENS_ALPHA * _norm(ppr_s) + (1.0 - _ENS_ALPHA) * _norm(tfidf_s)
        scores[anchor] = -np.inf

        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return _build_result_df(top_idx, scores)

    register_model(
        name="TF-IDF + PPR-MC",
        inputs=["text"],
        fn=_ensemble_recommend,
        score_label="blended score (TF-IDF + PPR-MC)",
        score_info="**Score:** 50% TF-IDF cosine similarity + 50% PPR visit frequency, both min-max normalised.",
    )

# ---------------------------------------------------------------------------
# Epsilon-Greedy Bandit (requires tfidf artifacts)
# ---------------------------------------------------------------------------
if _tfidf_mat is not None:
    _EG_EPSILON   = 0.1
    _EG_POOL_SIZE = 500
    _EG_SIM_STEPS = 50

    def _eg_recommend(query_text: str, query_cats: list[str], arxiv_id: str, k: int = TOP_K) -> pd.DataFrame:
        if not query_text.strip():
            return pd.DataFrame()

        # Candidate pool: top-_EG_POOL_SIZE by TF-IDF
        q_vec = _tfidf_vec.transform([query_text.strip()])
        sims  = (q_vec @ _tfidf_mat.T).toarray().ravel()
        pool_size = min(_EG_POOL_SIZE, len(sims))
        candidates = np.argpartition(sims, -pool_size)[-pool_size:].tolist()

        # Epsilon-greedy bandit over the candidate pool
        counts = np.zeros(pool_size)
        values = np.zeros(pool_size)
        item_arr = np.array(candidates)

        for _ in range(_EG_SIM_STEPS):
            if _random.random() < _EG_EPSILON or not values.any():
                i = _random.randrange(pool_size)
            else:
                i = int(np.argmax(values))
            # reward = TF-IDF similarity (proxy for relevance)
            reward = float(sims[item_arr[i]])
            counts[i] += 1
            values[i] += (reward - values[i]) / counts[i]

        # Rank by learned values
        ranked = item_arr[np.argsort(values)[::-1]]
        scores = np.full(len(sims), -np.inf)
        for rank, idx in enumerate(ranked):
            scores[idx] = float(values[np.where(item_arr == idx)[0][0]])

        top_idx = ranked[:k]
        return _build_result_df(top_idx, scores)

    register_model(
        name="Epsilon-Greedy",
        inputs=["text"],
        fn=_eg_recommend,
        score_label="bandit value estimate",
        score_info="**Score:** Epsilon-greedy bandit reranking over a TF-IDF candidate pool of 500 papers.",
    )

# ---------------------------------------------------------------------------
# Hybrid TF-IDF + SVD (requires both artifacts)
# ---------------------------------------------------------------------------
if _tfidf_mat is not None and _paper_embeddings is not None:
    _hybrid_n = min(_tfidf_mat.shape[0], _paper_embeddings.shape[0])

    def _hybrid_recommend(query_text: str, query_cats: list[str], arxiv_id: str, k: int = TOP_K) -> pd.DataFrame:
        if not query_text.strip():
            return pd.DataFrame()
        # TF-IDF cosine scores (clipped to the common size)
        q_vec    = _tfidf_vec.transform([query_text.strip()])
        tfidf_s  = (q_vec @ _tfidf_mat.T).toarray().ravel().astype(np.float64)[:_hybrid_n]
        # SVD scores via TF-IDF query proxy: find closest paper then use its embedding
        top1 = int(np.argmax(tfidf_s))
        svd_s = (_paper_embeddings[:_hybrid_n] @ _paper_embeddings[top1]).astype(np.float64)
        # min-max normalise then blend 50/50
        def _norm(v):
            lo, hi = v.min(), v.max()
            return (v - lo) / (hi - lo + 1e-10)
        scores = 0.5 * _norm(tfidf_s) + 0.5 * _norm(svd_s)
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return _build_result_df(top_idx, scores)

    register_model(
        name="Hybrid TF-IDF+SVD",
        inputs=["text"],
        fn=_hybrid_recommend,
        score_label="blended score (TF-IDF + SVD)",
        score_info="**Score:** 50% TF-IDF cosine similarity + 50% SVD latent factor similarity, both min-max normalised.",
    )

# ---------------------------------------------------------------------------
# GMF — gmf_model.pt  (requires torch)
# ---------------------------------------------------------------------------
def _load_gmf() -> bool:
    gmf_path = DATA_DIR / "gmf_model.pt"
    if not gmf_path.exists():
        return False
    try:
        import torch
        import torch.nn as nn

        class _GMF(nn.Module):
            def __init__(self, num_users, num_items, embedding_dim=32):
                super().__init__()
                self.user_embedding = nn.Embedding(num_users, embedding_dim)
                self.item_embedding = nn.Embedding(num_items, embedding_dim)
                self.fc = nn.Linear(embedding_dim, 1)
                self.sigmoid = nn.Sigmoid()
            def forward(self, user_idx, item_idx):
                u = self.user_embedding(user_idx)
                v = self.item_embedding(item_idx)
                return self.sigmoid(self.fc(u * v))

        # Load state dict first to infer the embedding size the model was trained with
        _sd = torch.load(gmf_path, map_location="cpu", weights_only=True)
        n = _sd["user_embedding.weight"].shape[0]
        _gmf = _GMF(n, n, embedding_dim=32)
        _gmf.load_state_dict(_sd)
        _gmf.eval()
        _all_items = torch.arange(n, dtype=torch.long)

        def _gmf_recommend(query_text: str, query_cats: list[str], arxiv_id: str, k: int = TOP_K) -> pd.DataFrame:
            idx = _text_to_paper_idx(query_text)
            if idx is None or idx >= n:
                return pd.DataFrame()
            with torch.no_grad():
                u = torch.tensor([idx], dtype=torch.long)
                scores = _gmf(u, _all_items).squeeze().numpy()
            if idx < len(scores):
                scores[idx] = -np.inf
            top_idx = np.argpartition(scores, -k)[-k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            return _build_result_df(top_idx, scores)

        register_model(
            name="GMF",
            inputs=["text"],
            fn=_gmf_recommend,
            score_label="GMF score",
            score_info="**Score:** finds the closest paper to your abstract via TF-IDF, then ranks by neural GMF citation similarity.",
        )
        return True
    except Exception as e:
        print(f"  GMF skipped: {e}")
        return False

# ---------------------------------------------------------------------------
# NeuMF — neumf_model.pt  (requires torch)
# ---------------------------------------------------------------------------
def _load_neumf() -> bool:
    neumf_path = DATA_DIR / "neumf_model.pt"
    if not neumf_path.exists():
        return False
    try:
        import torch
        import torch.nn as nn

        class _NeuMF(nn.Module):
            def __init__(self, num_users, num_items, gmf_dim=32, mlp_dims=(64, 32, 16)):
                super().__init__()
                self.gmf_user_emb = nn.Embedding(num_users, gmf_dim)
                self.gmf_item_emb = nn.Embedding(num_items, gmf_dim)
                mlp_emb_dim = mlp_dims[0] // 2
                self.mlp_user_emb = nn.Embedding(num_users, mlp_emb_dim)
                self.mlp_item_emb = nn.Embedding(num_items, mlp_emb_dim)
                layers = []
                in_dim = mlp_dims[0]
                for out_dim in mlp_dims[1:]:
                    layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
                    in_dim = out_dim
                self.mlp_network = nn.Sequential(*layers)
                self.fc_final = nn.Linear(gmf_dim + mlp_dims[-1], 1)
                self.sigmoid   = nn.Sigmoid()

            def forward(self, user_idx, item_idx):
                gmf_u = self.gmf_user_emb(user_idx)
                gmf_v = self.gmf_item_emb(item_idx)
                gmf_out = gmf_u * gmf_v
                mlp_u = self.mlp_user_emb(user_idx)
                mlp_v = self.mlp_item_emb(item_idx)
                mlp_out = self.mlp_network(torch.cat([mlp_u, mlp_v], dim=-1))
                return self.sigmoid(self.fc_final(torch.cat([gmf_out, mlp_out], dim=-1)))

        _sd = torch.load(neumf_path, map_location="cpu", weights_only=True)
        n = _sd["gmf_user_emb.weight"].shape[0]
        _neumf = _NeuMF(n, n)
        _neumf.load_state_dict(_sd)
        _neumf.eval()
        _all_items = torch.arange(n, dtype=torch.long)

        def _neumf_recommend(query_text: str, query_cats: list[str], arxiv_id: str, k: int = TOP_K) -> pd.DataFrame:
            idx = _text_to_paper_idx(query_text)
            if idx is None or idx >= n:
                return pd.DataFrame()
            with torch.no_grad():
                u = torch.tensor([idx], dtype=torch.long).expand(n)
                scores = _neumf(u, _all_items).squeeze().numpy()
            if idx < len(scores):
                scores[idx] = -np.inf
            top_idx = np.argpartition(scores, -k)[-k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            return _build_result_df(top_idx, scores)

        register_model(
            name="NeuMF",
            inputs=["text"],
            fn=_neumf_recommend,
            score_label="NeuMF score",
            score_info="**Score:** finds the closest paper to your abstract via TF-IDF, then ranks by NeuMF citation similarity (GMF + MLP).",
        )
        return True
    except Exception as e:
        print(f"  NeuMF skipped: {e}")
        return False

for _loader in [_load_gmf, _load_neumf]:
    try:
        _loader()
    except Exception as _e:
        print(f"  loader failed: {_e}")

print(f"models available: {list(MODELS.keys())}")

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def _model_inputs(model_name: str):
    if model_name not in MODELS:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), ""
    inp = MODELS[model_name]["inputs"]
    return (
        gr.update(visible=True),                        # text always shown
        gr.update(visible="categories" in inp),
        gr.update(visible="arxiv_id"   in inp),
        MODELS[model_name].get("score_info", ""),
    )


def _recommend(model_name: str, query_text: str, query_cats: list, arxiv_id: str):
    if model_name not in MODELS:
        return "Please select a model.", pd.DataFrame(), None

    try:
        df = MODELS[model_name]["fn"](
            query_text=query_text or "",
            query_cats=query_cats or [],
            arxiv_id=arxiv_id or "",
            k=TOP_K,
        )
    except Exception as e:
        return f"Error: {e}", pd.DataFrame(), None

    if df is None or df.empty:
        return "No results - try different input.", pd.DataFrame(), None

    summary = (
        f"**{len(df)} recommendations** from **{model_name}**"
        + (f" — query: *{(query_text or arxiv_id or '').strip()[:80]}*" if (query_text or arxiv_id) else "")
    )
    pie = _make_pie_chart(df)
    return summary, df, pie


_first_model = list(MODELS.keys())[0]

with gr.Blocks(title="ArXivists Citation Recommender", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # ArXivists Research Paper Citation Recommender
        Enter an abstract or arXiv ID, select categories, choose a model, and click **Recommend**.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=list(MODELS.keys()),
                value=_first_model,
                label="Model",
            )
            text_input = gr.Textbox(
                label="Abstract / query text",
                value=(
                    "We propose a self-attention transformer pre-trained with masked language "
                    "modeling on large unlabeled corpora, achieving state-of-the-art results "
                    "on GLUE, SQuAD, and named entity recognition benchmarks."
                ),
                lines=5,
                visible=True,
            )
            cat_input = gr.CheckboxGroup(
                choices=ALL_CATS,
                label="arXiv domains (optional filter/boost)",
                visible="categories" in MODELS[_first_model]["inputs"],
            )
            arxiv_input = gr.Textbox(
                label="arXiv ID (for graph-based models)",
                placeholder="e.g. 2301.12345",
                visible="arxiv_id" in MODELS[_first_model]["inputs"],
            )
            recommend_btn = gr.Button("Recommend", variant="primary")
            score_info_md = gr.Markdown(
                value=MODELS[_first_model].get("score_info", ""),
            )

        with gr.Column(scale=2):
            summary_md    = gr.Markdown("Results will appear here.")
            results_table = gr.Dataframe(
                headers=["rank", "arxiv_id", "year", "categories", "title", "abstract", "score"],
                datatype=["number", "str", "number", "str", "html", "str", "str"],
                wrap=True,
            )
            pie_chart = gr.Plot(label="Category distribution")

    model_dropdown.change(
        fn=_model_inputs,
        inputs=[model_dropdown],
        outputs=[text_input, cat_input, arxiv_input, score_info_md],
    )
    recommend_btn.click(
        fn=_recommend,
        inputs=[model_dropdown, text_input, cat_input, arxiv_input],
        outputs=[summary_md, results_table, pie_chart],
    )

    gr.Markdown(
        """
        ---
        **Model guide**

        | Model | Input | Signal |
        |---|---|---|
        | TF-IDF | abstract text | vocabulary / topic overlap |
        | SVD | arXiv ID | citation co-occurrence (latent factors) |
        | PageRank | categories | global citation importance |
        | Hybrid TF-IDF+SVD | abstract text | blended content + graph signal |
        | GMF | arXiv ID | neural collaborative filtering |
        | NeuMF | arXiv ID | neural MF (GMF + MLP) |
        """
    )


if __name__ == "__main__":
    demo.launch(share=False)
