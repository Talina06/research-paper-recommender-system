# CMPE 256 - Research Paper Citation Recommender
**Team:** The ArXivists - Talina Shrotriya, Sheetal Sattiraju, Manjula Ganesh
**Course:** CMPE 256 Recommender Systems, San José State University

---

## 1. Objective

Our objective is to help researchers discover papers they should cite by recommending
topically relevant cross-domain articles via a custom similarity scoring - not through
keyword search alone. This custom scoring covers a range of models: TF-IDF (content
baseline), SVD, NeuMF, PageRank, and a final hybrid recommender combining the best
individual performers.

The primary goal is cross-domain recommendations driven by citation data, category
codes, and abstract keywords. If a user studies stock-market modelling, we recommend
ML papers on time-series/forecasting *and* economics papers on market modelling - not
just papers that share the same vocabulary.

The system proposes a hybrid recommender combining content-based filtering (TF-IDF on
title + abstract) with collaborative filtering derived from citation interactions, and
exposes the models through a simple web UI where a user can enter free-text or select
arXiv categories and receive ranked paper recommendations.

---

## 2. Data Description

### 2.1 Sources

| Source | Access method | What we collect |
|---|---|---|
| **arXiv OAI-PMH** | Bulk metadata API (no credentials) | arxiv_id, title, abstract, categories, authors, year |
| **Semantic Scholar Graph API** | REST batch API (free key) | citation edges (source to cited), fields of study |

We targeted six arXiv categories: `cs.LG`, `cs.AI`, `cs.CV`, `cs.IR`, `cs.CL`,
`stat.ML` - covering machine learning, computer vision, NLP, information retrieval,
and statistical ML. We fetched papers from 2018–2024 across three OAI-PMH sets
(`cs`, `stat`, `eess`), running 21 parallel (set x year) tasks.

### 2.2 Dataset Statistics (from notebook run)

| Metric | Value |
|---|---|
| Papers (after cold-start filter) | **163,275** |
| Citation links (in-sample) | **1,910,207** |
| Avg references per paper | **11.7** |
| Citation matrix density | **0.0072%** (sparsity 99.99%) |
| Year range | 2018 – 2024 |
| Semantic Scholar field coverage | **100.0%** |
| Papers never cited in-sample | **30,286 (18.5%)** |

### 2.3 Year Distribution

| Year | Papers |
|---|---|
| 2018 | 13,724 |
| 2019 | 19,781 |
| 2020 | 25,860 |
| 2021 | 27,394 |
| 2022 | 30,030 |
| 2023 | 30,586 |
| 2024 | 15,900 |

Steady growth 2018–2023; 2024 is partial (papers up to mid-year only).

### 2.4 Top Semantic Scholar Fields of Study

S2 assigns multiple fields to a single paper (e.g. a paper can be tagged as both
Computer Science and Mathematics). The counts below are paper-field associations,
not unique paper counts, which is why they sum to 237,246 across 163,275 papers.

| Field | Associations |
|---|---|
| Computer Science | 162,615 |
| Mathematics | 34,409 |
| Engineering | 19,813 |
| Medicine | 11,017 |
| Physics | 6,033 |
| Biology | 2,421 |
| Economics | 938 |

### 2.5 Top 10 Most-Cited Papers in Dataset

| Rank | Citations | arXiv ID | Title |
|---|---|---|---|
| 1 | 14,911 | 1810.04805 | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding |
| 2 | 8,047 | 2005.14165 | Language Models are Few-Shot Learners |
| 3 | 6,930 | 1912.01703 | PyTorch: An Imperative Style, High-Performance Deep Learning Library |
| 4 | 6,728 | 2010.11929 | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale |
| 5 | 5,311 | 2103.00020 | Learning Transferable Visual Models From Natural Language Supervision |
| 6 | 4,853 | 1907.11692 | RoBERTa: A Robustly Optimized BERT Pretraining Approach |
| 7 | 4,473 | 1910.10683 | Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer |
| 8 | 3,836 | 2002.05709 | A Simple Framework for Contrastive Learning of Visual Representations |
| 9 | 3,359 | 2203.00667 | Generative Adversarial Networks |
| 10 | 3,011 | 2103.14030 | Swin Transformer: Hierarchical Vision Transformer using Shifted Windows |

Landmark ML papers dominating the top-10 is a strong sanity check - the citation graph is correct.

### 2.6 Sparsity Analysis

```
Citation matrix size : 163,275 x 163,275
Non-zero entries     : 1,910,207
Density              : 0.000072  (0.0072%)
Sparsity             : 0.999928  (99.9928%)
```

The matrix is extremely sparse - typical for academic citation graphs. Papers that cite
others: **110,989 (68.0%)**. Papers that get cited: **132,989 (81.5%)**.

Degree distribution (computed over papers that have at least one edge, not all 163k):

| Stat | Out-degree (110,989 papers) | In-degree (132,989 papers) |
|---|---|---|
| mean | 17.21 | 14.36 |
| median | 12 | 4 |
| max | 431 | 14,911 |

The overall average of 11.7 refs/paper in Section 2.2 is computed over all 163,275 papers
including those with zero citations (1,910,207 / 163,275 = 11.7). The means above are
higher because they exclude zero-degree papers from the denominator.

The in-degree distribution is highly skewed (power law) - a handful of landmark papers
attract the majority of citations.

### 2.7 Abstract Length

| Stat | Characters |
|---|---|
| mean | 1,221 |
| median | 1,211 |
| std | 316 |
| min | 3 |
| max | 3,214 |

No truncation issues. Zero null abstracts.

---

## 3. Methodology

### 3.1 Data Pipeline

1. **arXiv OAI-PMH harvest** - `metadataPrefix="arXiv"` for clean `<categories>` elements.
   21 parallel workers (3 sets x 7 years). Atomic checkpoint writes per (set, year).
2. **Semantic Scholar citation fetch** - 500 IDs per batch, shared token-bucket rate
   limiter, incremental checkpointing to Google Drive. Staleness check prevents using
   a checkpoint from a different OAI-PMH run.
3. **In-sample filtering** - keep edges where both source *and* target paper are in
   the collected dataset. `astype(str)` coercion prevents float-parse bug on IDs like
   `"2301.12345"`.
4. **Cold-start filter** - drop papers with total degree < 2. See Section 3.4 for details.
5. **Integer index** - 0-based `paper_idx` for direct scipy/ALS/NeuMF compatibility.
6. **Feature engineering**:
   - `text` = title + title + abstract (title doubled for higher TF-IDF weight)
   - `cite_pop` = normalized in-sample citation count (0–1)
   - `s2_fields` = Semantic Scholar fields of study

### 3.2 Models

#### Baseline - TF-IDF Cosine Similarity
- `TfidfVectorizer`: 50,000 features, bigrams (1,2), `sublinear_tf=True`, `norm="l2"`
- Ranking: dot product of l2-normalised vectors = cosine similarity
- TF-IDF matrix: 163,275 x 50,000 terms, 33,836,571 non-zeros, 99.59% sparse

#### Planned Models (Sections 7+)
| Model | Signal |
|---|---|
| SVD | Latent factors from citation matrix |
| PageRank | Graph centrality on citation DAG |
| GMF + NeuMF | Generalised Matrix Factorisation and Neural Matrix Factorisation |
| Hybrid | weighted combination of SVD + TF-IDF + PageRank + category score |

### 3.3 Evaluation Framework

All models share the same 80/20 random edge split (seed=256) and the same 1,000
fixed query papers (seed=42), ensuring fair comparison. A new model needs only to
provide a score function:

```python
evaluate_model("Model Name", score_fn=my_score_fn)   # per-paper
evaluate_model("Model Name", batch_score_fn=my_batch_fn)  # batched (faster)
compare_models()  # side-by-side table + chart
```

### 3.4 Cold-Start Filter

In recommender systems, the "cold-start problem" refers to items (or users) for
which the system has too little data to make useful predictions. In our case, a
paper is cold-start if it appears in very few or no citation edges after in-sample
filtering.

**Why it is a problem here**

The CF and graph-based models all learn from the citation graph. Each paper gets
represented as a row or node in that graph. If a paper has only 0 or 1 edges,
the model sees almost no signal for it:

- SVD learns latent factor vectors by factorizing the paper-by-paper citation
  matrix. A paper with no observed citation interactions simply never appears in
  the training objective, so its learned vector receives no updates and stays at
  its random initialization.
- NeuMF (Neural Matrix Factorisation) learns paper embeddings via neural network
  layers rather than direct matrix decomposition. A paper with no interactions
  similarly receives no gradient updates, leaving its embedding at random initialization.
- PageRank is a graph centrality algorithm, not a CF model. It assigns each paper
  an importance score by iteratively propagating rank through the citation edges.
  A paper with no incoming or outgoing edges is completely disconnected from the
  graph and receives only the minimum teleportation-based base score.

Including isolated papers in the graph does not cause them to add noise to the
training loss -- in matrix factorization, unobserved interactions are simply
absent from the objective. However, they inflate the embedding tables and the
adjacency structure without contributing any useful signal, and would produce
unreliable recommendations since there is no citation evidence to learn from.

**What the filter does**

We compute each paper's total degree: in-degree (how many papers in the dataset
cite it) plus out-degree (how many papers in the dataset it cites). Papers with a
total degree below `MIN_DEGREE = 2` are removed from both the paper table and the
citation edge list.

From the dataset:

| Stat | Value |
|---|---|
| Papers before filter | ~170k |
| Papers after filter | 163,275 |
| Papers removed | ~7k |
| Papers with cite_pop = 0 (never cited) | 30,286 (18.5%) |

Note that 18.5% of papers have zero in-sample citations but are still kept if their
out-degree is at least 2. These papers do cite others in the dataset, so the CF
models still have some signal to work with.

**What we lose**

Very new papers (submitted late 2024) and highly niche papers that fall outside our
six target categories tend to be filtered out. This is an acceptable trade-off: the
filtered papers are exactly the ones where any recommendation would be unreliable
anyway, and removing them makes the training graph denser and more informative for
the remaining 163k papers.

**Threshold choice**

`MIN_DEGREE = 2` is conservative on purpose. It removes only completely isolated
papers (degree 0) and papers with a single edge (degree 1). A higher threshold like
5 or 10 would improve training signal further but would also remove legitimately
relevant recent papers. The threshold is exposed as a config variable so teammates
can experiment with stricter values when training the CF models.

---

## 4. Benchmark & Evaluation

### 4.1 Evaluation Protocol

- **Train/test split:** 80% of citation edges for training, 20% held out for evaluation
- **Query papers:** 1,000 randomly sampled papers with ≥1 held-out citation
- **Exclusions:** self-links and training citations removed from ranked list
- **Top-K ranking:** `np.argpartition` (O(n)) for speed, then sort only top-K
- **Metrics:** Precision@K, Recall@K, NDCG@K, Hit Rate@K, MRR at K ∈ {5, 10, 20}

### 4.2 TF-IDF Baseline Results

| Metric | @5 | @10 | @20 |
|:---|---:|---:|---:|
| **Precision** | 0.0440 | 0.0324 | 0.0237 |
| **Recall** | 0.0724 | 0.1023 | 0.1453 |
| **NDCG** | 0.0712 | 0.0801 | 0.0944 |
| **Hit Rate** | 0.1930 | 0.2630 | 0.3540 |
| **MRR** | 0.1139 | 0.1232 | 0.1293 |

*Evaluated on 1,000 query papers, 80/20 edge split, seed=256/42.*

### 4.3 Interpretation

**529x lift over random** - a random recommender on 163,275 papers achieves
Precision@10 ≈ 0.00006. TF-IDF at 0.0324 is a meaningful signal.

**Hit Rate@10 = 26.3%** - 1 in 4 queries finds a real citation in the top-10. Solid
for a content-only model with no structural information.

**NDCG@10 = 0.0801** (moderate) - true citations are found but not consistently
ranked at the very top of the list.

**MRR@10 = 0.1232** - the Mean Reciprocal Rank is the average of 1/rank of the
first true citation across all 1,000 queries. Queries where no hit appears in
the top-10 contribute 0. An MRR of 0.1232 means the effective first-hit position,
weighted across all queries including misses, corresponds to around rank 8 for
those queries that do find a hit (Hit Rate@10 = 26.3%).

**Precision falls as K grows** (0.044 to 0.024): the top-5 slots are cleanest.
Recall grows with K as expected.

**Headroom for CF models:** Any model beating Precision@10 > 0.0324 and NDCG@10 >
0.0801 represents a measurable gain from co-citation structure beyond what text provides.

### 4.4 Planned Comparison Table (to be updated)

| Model | P@10 | R@10 | NDCG@10 | HR@10 | MRR@10 |
|---|---|---|---|---|---|
| TF-IDF (baseline) | 0.0324 | 0.1023 | 0.0801 | 0.2630 | 0.1232 |
| SVD | TBD | TBD | TBD | TBD | TBD |
| PageRank | TBD | TBD | TBD | TBD | TBD |
| NeuMF | TBD | TBD | TBD | TBD | TBD |
| Hybrid | TBD | TBD | TBD | TBD | TBD |

---

## 5. Lessons Learned & Challenges

**Category matching bug.** The arXiv OAI-PMH API offers two metadata formats. The default format (`oai_dc`) stores categories as human-readable strings like "Computer Science - Machine Learning", not as codes like `cs.LG`. Our regex filter matched only 1-2 records per set as a result. Switching to `metadataPrefix="arXiv"` fixed this because that format provides a dedicated `<categories>` field with space-separated arXiv codes that match exactly.

**Zero in-sample citations after fetching 3.2 million raw links.** This was the most confusing bug. The root cause was two-fold. First, pandas silently reads arXiv IDs like `"2301.12345"` as `float64` from CSV files because they look like numbers. A string comparison between a float and a string always returns False, so `isin()` reported no matches. Fixing this required adding `dtype=str` to every CSV read of ID columns. Second, if the Semantic Scholar checkpoint was saved during a previous run with a different set of papers, the source IDs in the checkpoint would not match the current paper set. We added a staleness check that discards the checkpoint if fewer than 50% of its IDs overlap with the current run.

**Evaluation speed.** The initial TF-IDF evaluation loop processed one query paper at a time, calling `toarray()` on a 163k-element sparse vector and then running `np.argsort` on all 163k scores. At 1 second per paper that was 17 minutes for 1,000 queries. Batching 100 queries into a single sparse matrix multiply and replacing `argsort` with `argpartition` (which only finds the top-K elements in O(n) instead of sorting everything in O(n log n)) brought the runtime under 2 minutes.

**Session crashes during long fetches.** The Semantic Scholar fetch takes several hours with a free API key. Colab sessions disconnect mid-run regularly. We solved this by writing checkpoints atomically: every N batches the code writes to a `.tmp` file and then renames it into place. A rename is atomic on all major filesystems, so a crash mid-write never leaves a corrupt checkpoint that would be silently loaded on the next run. The arXiv fetch uses the same pattern per (set, year) task.

**Cold-start papers.** About 18.5% of papers are never cited by any other paper in the dataset. Keeping them in the training graph adds noise without adding signal for the CF models. The cold-start filter drops papers whose total degree (in-degree plus out-degree) is below 2, which removed roughly 7k papers while keeping the 163k that have at least some citation connectivity.

---

## 6. Source Code

GitHub repository: https://github.com/Talina06/research-paper-recommender-system

| File | Contents |
|---|---|
| `notebook.ipynb` | Full pipeline: data collection, EDA, baseline model, evaluation framework |
| `app.py` | Gradio web UI |
| `README.md` | Setup and run instructions |

---

---

[^1]: arXiv dataset on Kaggle: https://www.kaggle.com/datasets/Cornell-University/arxiv
[^2]: unarXive (full text + citations): https://zenodo.org/records/7752754
