"""
search_engine.py
-----------------
The core intelligence layer of the project. Wraps semantic search (FAISS +
sentence-transformers), abstractive summarization (BART), and keyword
extraction (KeyBERT) into a single reusable `PaperSearchEngine` class.

Usage:
    from search_engine import PaperSearchEngine

    engine = PaperSearchEngine()
    results = engine.search("deep learning for medical image analysis", k=5)
    for r in results:
        print(r["title"], r["score"])
"""

import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from keybert import KeyBERT

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CLEANED_CSV_PATH = os.path.join(DATA_DIR, "cleaned_arxiv_papers.csv")
INDEX_PATH = os.path.join(DATA_DIR, "paper_faiss.index")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARIZER_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"  # 300MB, lighter than full BART


class PaperSearchEngine:
    """
    Semantic search + summarization + keyword extraction over the ArXiv
    ML papers corpus. Loads the cleaned dataframe, the FAISS index, and the
    three underlying models once, then serves fast repeated queries.
    """

    def __init__(self, load_summarizer: bool = True, load_keybert: bool = True, device: int = -1):
        if not os.path.exists(CLEANED_CSV_PATH) or not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(
                "Cleaned data / FAISS index not found. Run "
                "`python src/build_index.py` first to generate them."
            )

        print("Loading cleaned dataset...")
        self.df = pd.read_csv(CLEANED_CSV_PATH)

        print("Loading FAISS index...")
        self.index = faiss.read_index(INDEX_PATH)

        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        self.summarizer = None
        if load_summarizer:
            print(f"Loading summarization model: {SUMMARIZER_MODEL_NAME}")
            self.summarizer = pipeline(
                "summarization",
                model=SUMMARIZER_MODEL_NAME,
                device=device,  # -1 = CPU, 0 = first GPU
            )

        self.kw_model = None
        if load_keybert:
            print("Loading KeyBERT keyword extractor...")
            self.kw_model = KeyBERT(model=self.embed_model)

    # ------------------------------------------------------------------ #
    # Core semantic search
    # ------------------------------------------------------------------ #
    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return the top-k papers most semantically similar to the query."""
        query_embedding = self.embed_model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            row = self.df.iloc[idx]
            results.append({
                "rank": len(results) + 1,
                "score": float(score),
                "title": row["title"],
                "abstract": row["abstract"],
                "row_index": int(idx),
            })
        return results

    # ------------------------------------------------------------------ #
    # Summarization
    # ------------------------------------------------------------------ #
    def summarize(self, text: str, max_length: int = 120, min_length: int = 40) -> str:
        if self.summarizer is None:
            raise RuntimeError("Summarizer not loaded. Init with load_summarizer=True.")
        # BART has a token limit; truncate very long abstracts defensively.
        summary = self.summarizer(text[:2000], max_length=max_length, min_length=min_length)
        return summary[0]["summary_text"]

    # ------------------------------------------------------------------ #
    # Keyword extraction
    # ------------------------------------------------------------------ #
    def extract_keywords(self, text: str, top_n: int = 8) -> list[tuple[str, float]]:
        if self.kw_model is None:
            raise RuntimeError("KeyBERT not loaded. Init with load_keybert=True.")
        return self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=top_n,
        )

    # ------------------------------------------------------------------ #
    # Convenience: search + summarize + keywords in one call
    # ------------------------------------------------------------------ #
    def full_report(self, query: str, k: int = 5) -> list[dict]:
        """
        Run a search and enrich every hit with an AI-generated summary and
        extracted keywords -- this is what powers the Streamlit demo.
        """
        results = self.search(query, k=k)
        for r in results:
            if self.summarizer is not None:
                r["summary"] = self.summarize(r["abstract"])
            if self.kw_model is not None:
                r["keywords"] = self.extract_keywords(r["abstract"])
        return results


if __name__ == "__main__":
    engine = PaperSearchEngine()
    demo_query = "deep learning for medical image analysis"
    print(f"\nQuery: {demo_query}\n" + "-" * 60)
    for r in engine.full_report(demo_query, k=3):
        print(f"[{r['rank']}] {r['title']}  (score={r['score']:.3f})")
        print("Summary:", r.get("summary", ""))
        print("Keywords:", r.get("keywords", ""))
        print()
