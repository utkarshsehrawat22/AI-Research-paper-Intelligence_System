# 📚 AI Research Paper Intelligence System

A semantic search engine for Machine Learning research papers that goes beyond keyword matching — it understands the **meaning** of a query, retrieves the most relevant papers from a corpus of 50,000 ArXiv abstracts, then automatically **summarizes** each result and **extracts key topics**, so you can scan a paper's relevance in seconds instead of reading the full abstract.

Built as part of the Coding Blocks Internship program.

---

## 🎯 What it does

Type a natural-language research query like:

> "deep learning for medical image analysis"

...and the system:

1. **Understands intent** — converts the query into a 384-dimensional semantic vector
2. **Searches by meaning, not keywords** — retrieves the top-k most similar papers using cosine similarity over a FAISS vector index
3. **Summarizes** — condenses each returned abstract into a short, readable summary using a BART-based transformer
4. **Extracts key phrases** — pulls out the core topics/keywords from each paper using KeyBERT

All of this is wrapped in a clean, interactive **Streamlit** web app.

---

## 🧠 How it works (architecture)

```
                     ┌─────────────────────────┐
                     │   ArXiv ML Papers (HF)   │
                     │  ~50,000 title+abstract  │
                     └────────────┬─────────────┘
                                  │  clean & merge
                                  ▼
                     ┌─────────────────────────┐
                     │   Sentence-Transformer   │
                     │    (all-MiniLM-L6-v2)    │
                     │   text --> 384-dim vec   │
                     └────────────┬─────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │      FAISS Index         │
                     │  (Inner Product / cosine)│
                     └────────────┬─────────────┘
                                  │
        user query ──encode──────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │   Top-K similar papers   │
                     └────────────┬─────────────┘
                                  │
                     ┌────────────┴─────────────┐
                     ▼                           ▼
          ┌────────────────────┐     ┌────────────────────┐
          │   BART Summarizer   │     │   KeyBERT Keywords  │
          │ (distilbart-cnn-12) │     │  (n-gram phrases)   │
          └────────────────────┘     └────────────────────┘
                     │                           │
                     └────────────┬──────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │      Streamlit UI        │
                     └─────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Dataset | [CShorten/ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) (Hugging Face) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim) |
| Vector Search | `faiss-cpu` — `IndexFlatIP` (cosine similarity via L2-normalized inner product) |
| Summarization | `transformers` — `sshleifer/distilbart-cnn-12-6` |
| Keyword Extraction | `keybert` |
| Data handling | `pandas`, `numpy` |
| Web App | `streamlit` |

---

## 📂 Project Structure

```
AI-Research-Paper-Intelligence-System/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md              # explains how generated data/embeddings/index are created
├── notebooks/
│   ├── 01_EDA_and_Embeddings.ipynb   # data exploration + embedding generation walkthrough
│   └── 02_Search_Engine.ipynb        # FAISS search + summarization + keyword extraction walkthrough
└── src/
    ├── data_prep.py            # load & clean the raw dataset
    ├── build_index.py          # generate embeddings + build the FAISS index
    ├── search_engine.py        # PaperSearchEngine class: search, summarize, extract keywords
    └── app.py                  # Streamlit web app
```

The `notebooks/` walk through the reasoning behind every step (useful for
understanding or presenting the project). The `src/` folder holds the same
logic refactored into clean, reusable, production-style modules that power
the actual app.

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the search index (one-time setup)

This downloads the dataset, generates embeddings for ~50,000 papers, and
builds the FAISS index. It's compute-heavy (~20-30 min on CPU) but only
needs to run once — results are cached in `data/`.

```bash
python src/data_prep.py
python src/build_index.py
```

### 3. Launch the app

```bash
streamlit run src/app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`) and
start searching.

### Or use it from Python directly

```python
from src.search_engine import PaperSearchEngine

engine = PaperSearchEngine()
results = engine.full_report("deep learning for medical image analysis", k=5)

for r in results:
    print(r["title"], "-", r["score"])
    print(r["summary"])
    print(r["keywords"])
```

---

## 💡 Key Design Decisions

- **Why FAISS `IndexFlatIP` instead of a database?** For 50k papers, an
  exact (non-approximate) flat index is fast enough and guarantees perfect
  recall — no accuracy trade-off from approximate search.
- **Why normalize embeddings before indexing?** Cosine similarity depends
  only on vector *direction*, not magnitude. Normalizing every vector to
  unit length lets us use FAISS's fast Inner Product search to get
  mathematically identical results to cosine similarity.
- **Why `all-MiniLM-L6-v2`?** A strong balance of speed and semantic
  quality — 384 dimensions is small enough to index and query instantly,
  while still capturing rich sentence-level meaning.
- **Why cache embeddings/index to disk?** Re-encoding 50,000 papers takes
  ~20-30 minutes; caching means the app starts in seconds on every
  subsequent run.

---

## 🔮 Possible Extensions

- Swap `IndexFlatIP` for `IndexIVFFlat` / `IndexHNSW` to scale to millions of papers
- Add filters (year, category) alongside semantic search
- Deploy the Streamlit app publicly (Streamlit Community Cloud / HF Spaces)
- Add a citation graph / "papers similar to this one" feature

---

## 🙋 Author

Built by **Singaram Hemalatha** — Coding Blocks Internship, Project 2.
