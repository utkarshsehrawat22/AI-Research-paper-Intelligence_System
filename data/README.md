# Data Folder

This folder holds generated artifacts and is intentionally left empty in the
repository (the files are large and 100% regenerable from source).

Source dataset: [CShorten/ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) on Hugging Face.

Running the pipeline will create these files here automatically:

| File | Created by | Description |
|---|---|---|
| `cleaned_arxiv_papers.csv` | `src/data_prep.py` | Cleaned title + abstract + combined `paper_text` for ~50,000 papers |
| `arxiv_embeddings.npy` | `src/build_index.py` | 384-dim sentence embeddings for every paper (all-MiniLM-L6-v2) |
| `paper_faiss.index` | `src/build_index.py` | FAISS `IndexFlatIP` index used for fast cosine-similarity search |

## To generate everything from scratch

```bash
python src/data_prep.py     # downloads dataset, cleans it -> cleaned_arxiv_papers.csv
python src/build_index.py   # generates embeddings + FAISS index (~20-30 min on CPU for 50k papers)
```

After that, `src/search_engine.py` and `src/app.py` will load these cached
files instantly instead of recomputing them.
