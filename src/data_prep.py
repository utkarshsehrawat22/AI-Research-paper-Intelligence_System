"""
data_prep.py
------------
Loads the raw ArXiv ML papers dataset from Hugging Face, cleans it, and
prepares a single text field ("paper_text") that will be fed into the
sentence-transformer model for embedding generation.

Dataset: CShorten/ML-ArXiv-Papers (https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers)

Run directly:
    python src/data_prep.py
"""

import os
import pandas as pd
from datasets import load_dataset

RAW_DATASET_NAME = "CShorten/ML-ArXiv-Papers"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CLEANED_CSV_PATH = os.path.join(DATA_DIR, "cleaned_arxiv_papers.csv")

# Cap the number of papers we process. 50,000 keeps embedding generation
# and the FAISS index fast enough to demo on a laptop / free-tier GPU.
MAX_PAPERS = 50_000


def load_raw_data() -> pd.DataFrame:
    """Download (or load from cache) the raw ArXiv papers dataset."""
    print(f"Loading dataset '{RAW_DATASET_NAME}' from Hugging Face...")
    dataset = load_dataset(RAW_DATASET_NAME)
    df = dataset["train"].to_pandas()
    return df


def clean_data(df: pd.DataFrame, max_papers: int = MAX_PAPERS) -> pd.DataFrame:
    """
    Keep only the columns we need, cap the dataset size, drop rows with
    missing text, and build a combined `paper_text` field (title + abstract)
    that gives the embedding model full context per paper.
    """
    df = df[["title", "abstract"]].copy()
    df = df.head(max_papers)

    # Drop any rows where either field is missing -- an embedding of "None"
    # or an empty string is meaningless and would pollute the vector index.
    df = df.dropna(subset=["title", "abstract"]).reset_index(drop=True)

    # Merge title + abstract into one context-rich block for embedding.
    df["paper_text"] = df["title"] + " " + df["abstract"]

    # Strip newlines / stray whitespace so the transformer sees one clean
    # continuous paragraph rather than fragmented lines.
    df["paper_text"] = df["paper_text"].str.replace("\n", " ", regex=False)
    df["paper_text"] = df["paper_text"].str.strip()

    return df


def save_cleaned_data(df: pd.DataFrame, path: str = CLEANED_CSV_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Cleaned dataframe saved to {path} ({len(df):,} papers)")


def get_cleaned_data(force_reload: bool = False) -> pd.DataFrame:
    """
    Convenience entry point used by other modules: returns the cleaned
    dataframe, loading from the cached CSV if it already exists so we don't
    re-download the dataset every time.
    """
    if not force_reload and os.path.exists(CLEANED_CSV_PATH):
        print("Loading cached cleaned dataset...")
        return pd.read_csv(CLEANED_CSV_PATH)

    df = load_raw_data()
    df = clean_data(df)
    save_cleaned_data(df)
    return df


if __name__ == "__main__":
    get_cleaned_data(force_reload=True)
