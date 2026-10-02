# AI Research Paper Intelligence System

An AI-powered research assistant for discovering and understanding machine-learning papers. The system combines semantic search, FAISS vector retrieval, local NLP models, and an optional Groq-powered RAG assistant in a Streamlit application.

## What the project does

The application supports two modes:

### Local semantic search

1. Converts paper titles and abstracts into 384-dimensional embeddings.
2. Stores the embeddings in a FAISS index.
3. Converts a user's query into an embedding.
4. Retrieves the most semantically similar papers.
5. Summarizes results with DistilBART.
6. Extracts key phrases with KeyBERT.

This mode works without an API key.

### Groq RAG assistant

When `GROQ_API_KEY` is configured, the application can retrieve relevant papers and provide their abstracts as context to a Groq language model. It supports:

- Research questions over retrieved papers
- Topic and keyword questions
- Paper comparison questions
- Source-paper display alongside the generated answer

The RAG mode uses LangChain prompt composition and deterministic routing so that comparison questions are handled differently from ordinary search questions.

## Architecture

```text
ArXiv ML papers
      |
      v
Data cleaning and title + abstract combination
      |
      v
Sentence Transformer embeddings
      |
      v
Normalized FAISS inner-product index
      |
      +----------------------+
      |                      |
      v                      v
Local semantic search        Groq RAG assistant
      |                      |
      v                      v
BART summaries + KeyBERT    Retrieved context + Groq answer
      \                      /
       v                    v
              Streamlit UI
```

## Technology stack

- Python
- Streamlit
- Pandas and NumPy
- Hugging Face Datasets
- Sentence Transformers: `all-MiniLM-L6-v2`
- FAISS for vector similarity search
- Transformers: `distilbart-cnn-12-6`
- KeyBERT for keyword extraction
- LangChain Core for RAG prompt composition
- Groq API for optional cloud LLM generation

## Project structure

```text
AI-Research-Intelligence-System/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── README.md
│   ├── cleaned_arxiv_papers.csv       # generated locally
│   ├── arxiv_embeddings.npy           # generated locally
│   └── paper_faiss.index               # generated locally
└── src/
    ├── app.py                          # Streamlit interface
    ├── data_prep.py                    # dataset download and cleaning
    ├── build_index.py                  # embeddings and FAISS index
    ├── search_engine.py                # local search, summaries, keywords
    └── rag_agent.py                    # optional Groq RAG assistant
```

## Setup on macOS or Linux

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "pip==24.3.1"
python -m pip install -r requirements.txt
```

The project currently supports Python 3.9+, although Python 3.11 is recommended for a more modern scientific-Python environment.

## Prepare the data and index

The first setup downloads the public Hugging Face dataset and generates embeddings for up to 50,000 papers:

```bash
python -m src.data_prep
python -m src.build_index
```

This can take a while on a CPU. The generated files are cached under `data/` and are excluded from Git because they are large.

On some macOS Python 3.9 installations, limit native CPU threading before indexing or launching the app:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

## Configure Groq RAG mode

Local search does not require an API key. To use the RAG assistant:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_actual_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Never commit `.env` or share the key. `.env` is excluded by `.gitignore`.

## Run the application

```bash
streamlit run src/app.py
```

Open `http://localhost:8501` in a browser.

For local search, enter a query and click **Search**. For RAG mode, enable **Groq RAG assistant**, enter a question, and click **Ask RAG assistant**.

Example questions:

```text
deep learning for medical image analysis
```

```text
Compare Vision Transformers with CNN architectures.
```

```text
What are the main topics in papers about reinforcement learning?
```

## How retrieval works

Each paper is represented by a normalized embedding. The query is embedded using the same model. FAISS uses inner-product search; because the vectors are L2-normalized, inner product is equivalent to cosine similarity.

The application retrieves the top-k papers and uses their titles and abstracts as the evidence for summaries or RAG responses.

## Limitations

- The dataset is a snapshot and does not automatically include new papers.
- Summaries and LLM answers can contain mistakes; the original abstracts should be checked.
- RAG quality depends on retrieval quality and Groq availability.
- Exact `IndexFlatIP` search is suitable for this corpus but may not scale to millions of papers.
- Generated data files are not included in GitHub and must be regenerated on another machine.

## Future improvements

- Add publication-year and category filters.
- Add ArXiv links, authors, and citation metadata.
- Add hybrid BM25 + vector retrieval.
- Add cross-encoder reranking.
- Add PDF upload and analysis.
- Add evaluation metrics such as Precision@k and Recall@k.
- Deploy the Streamlit app.

## GitHub safety

Before committing, verify that `.env`, `.venv/`, and generated data files are not staged:

```bash
git status
```

Commit the README and source changes with:

```bash
git add README.md src requirements.txt .env.example .gitignore
git commit -m "Document advanced research paper RAG system"
git push origin main
```

