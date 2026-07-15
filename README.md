---
title: FinRAG
emoji: 📈
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.41.1"
app_file: app.py
pinned: false
---

# FinRAG

FinRAG is a Retrieval-Augmented Generation (RAG) system built for querying real financial documents. You type a natural language question — "What did NVIDIA say about AI demand?" or "What are Tesla's key risk factors?" — and the system finds the most relevant passages from a corpus of SEC filings and earnings call transcripts, then generates a grounded answer using an LLM. Every answer cites the exact source passages it was derived from, so you can verify the claim directly.

Live demo: [huggingface.co/spaces/aekankpatel/finrag](https://huggingface.co/spaces/aekankpatel/finrag)

---

## What problem it solves

Financial documents are long and dense. A 10-K filing can run 150+ pages. An earnings call transcript is 30–50 pages of executive commentary. Reading across multiple companies to find a specific piece of information — say, how three different companies characterize AI investment risk — takes hours.

FinRAG compresses that into a single search. It lets you ask questions in plain language and surfaces the relevant passages across all 24 documents in the corpus instantly, with an LLM synthesizing a clear answer from them.

---

## How it works

### Step 1 — Document ingestion (offline, done once)

Each financial PDF is parsed to plain text using `pdfplumber` (`ingest_pdfs.py`). The text is then chunked into overlapping token windows using LlamaIndex's `TokenTextSplitter`:

- **Chunk size:** 128 tokens per passage
- **Chunk overlap:** 20 tokens (so context is not cut off at boundaries)
- **Metadata attached to each chunk:** the source filename, so answers can be traced back to a specific document

Each chunk is embedded into a 384-dimensional vector using `BAAI/bge-small-en-v1.5`, a lightweight but accurate sentence embedding model from HuggingFace. The resulting vectors — along with the original text and metadata — are saved to a LlamaIndex `SimpleVectorStore` and persisted to disk as JSON files.

This entire process is run once via `build_index.py` and the output is stored in the [finrag-index](https://github.com/aekankpatel/finrag-index) repository.

### Step 2 — Index loading (on app startup)

When the HuggingFace Space starts, `app.py` checks if the index files exist locally. If not, it downloads them from GitHub:

- `docstore.json` (~26 MB) — all text chunks with metadata, downloaded via `raw.githubusercontent.com`
- `index_store.json` (~1 MB) — index structure and node ID mappings
- `default__vector_store.json` (~124 MB) — the embedding vectors, stored in Git LFS and downloaded via `media.githubusercontent.com`
- `graph_store.json` — empty placeholder, not used

LlamaIndex reconstructs the full in-memory index from these files. The embedding model (`bge-small-en-v1.5`) is also loaded at this point so that query embeddings use the same vector space as the stored document embeddings.

### Step 3 — Query and retrieval

When you submit a question:

1. The question is embedded using the same `bge-small-en-v1.5` model
2. The query vector is compared against all stored chunk vectors using cosine similarity
3. The top-k most similar chunks are retrieved (default: 8, configurable via slider)
4. If a document filter is active (auto-detected from keywords or manually selected), only chunks from that document are considered

The **auto-detect** feature works by matching keywords in your question against a company/topic map. If you mention "Tesla" or "tsla", it restricts retrieval to the Tesla 10-K. If no keyword matches, retrieval runs across the full corpus.

### Step 4 — Answer generation

The top 5 retrieved chunks (up to 500 tokens each) are assembled into a context block and sent to the Groq API along with the question. The model used is `llama-3.1-8b-instant`, which is fast and runs on Groq's inference hardware.

The system prompt instructs the model to answer strictly from the provided context and not to introduce outside knowledge. This keeps answers grounded and prevents hallucination on financial specifics like revenue figures or risk disclosures.

### Step 5 — Display

The app shows:
- The document(s) searched
- A retrieval confidence bar (based on the top cosine similarity score)
- The generated answer
- Each source passage with its document name and similarity score
- A download button to export the full answer + sources as a `.txt` file

```
User types question
        |
        v
bge-small-en-v1.5 embeds the question into a 384-dim vector
        |
        v
Cosine similarity search across all stored chunk vectors
        |
        v
Top-k chunks retrieved (optionally filtered to one document)
        |
        v
Top 5 chunks sent as context to llama-3.1-8b-instant via Groq
        |
        v
LLM generates answer grounded in context
        |
        v
Answer + source passages + confidence score displayed
```

---

## Documents covered

| Company / Topic | Documents |
|---|---|
| Apple | 10-K 2025, 10-Q Q1 2025, 10-Q Q4 2025 |
| Amazon | 10-K 2025, 10-Q Q3 2025, Q4 2025 earnings call |
| NVIDIA | 10-Q Q3 2025, Q4 2025 earnings call |
| Meta | 10-K 2025 |
| Microsoft | 10-Q Q3 2025, Q2 2025 earnings call |
| Tesla | 10-K 2025, 10-Q Q3 2025 |
| Goldman Sachs | BDC 10-Q Q2 2025, 2026 M&A outlook |
| Bank of America | 2024 Annual Report, Q4 2025 earnings call |
| JPMorgan | Q4 2025 earnings call |
| Walmart | Q4 2026 earnings call |
| Global macro | World Bank Global Economic Prospects Jan 2026 |
| Banking sector | EY Global Banking Outlook 2025 |
| Capital markets | Capital Markets Forecast 2026 |

---

## Features

- **Auto-detect** — Keywords in your question (company names, tickers) automatically narrow the search to the most relevant document
- **Manual filter** — Override auto-detect and pin the query to any specific document
- **Compare mode** — Run the same question against two documents side by side to directly compare how companies describe the same topic
- **Confidence score** — The top cosine similarity score is shown as a percentage bar, giving a rough signal of how well the corpus covers your question
- **Chat history** — Previous questions and answers are shown in the session so you can scroll back through your research
- **Export** — Every answer can be downloaded as a `.txt` file with the full source passages included

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Embeddings | `BAAI/bge-small-en-v1.5` (HuggingFace) |
| Vector index | LlamaIndex `SimpleVectorStore` |
| LLM | `llama-3.1-8b-instant` via Groq API |
| Index storage | GitHub + Git LFS ([aekankpatel/finrag-index](https://github.com/aekankpatel/finrag-index)) |
| Hosting | HuggingFace Spaces |

---

## Project structure

```
finrag/
├── app.py               # Streamlit app — handles index loading, retrieval, and UI
├── requirements.txt     # Python dependencies
├── build_index.py       # Offline script — chunks documents and builds the vector index
├── ingest_pdfs.py       # Parses PDFs to plain text files
├── evaluate.py          # Evaluation script for retrieval quality
├── query.py             # Standalone query script (no UI)
├── data/
│   ├── raw/             # Source PDF files
│   └── processed/       # Extracted plain text files (one per document)
└── finrag/
    └── index/           # Vector index files (downloaded at runtime from finrag-index repo)
```

### Key files explained

**`ingest_pdfs.py`** — Reads each PDF from `data/raw/` using `pdfplumber`, extracts the text page by page, and writes it to `data/processed/` as a `.txt` file. Handles encoding issues and strips junk characters.

**`build_index.py`** — Reads the processed `.txt` files, attaches the filename as `source` metadata to each document, splits everything into 128-token chunks with 20-token overlap using `TokenTextSplitter`, embeds each chunk with `bge-small-en-v1.5`, and persists the resulting index to `finrag/index/`.

**`app.py`** — On startup: downloads the pre-built index from GitHub if not cached, loads it into memory, and initializes the embedding model. On each query: embeds the question, runs similarity search, sends context to the Groq API, and renders the answer with sources.

---

## Running locally

```bash
git clone https://github.com/aekankpatel/finrag.git
cd finrag
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Run the app:

```bash
streamlit run app.py
```

On first run the vector index (~150 MB total) is downloaded from GitHub automatically into `finrag/index/`. Subsequent runs load from the local cache.

To rebuild the index from scratch (e.g. after adding new documents):

```bash
python ingest_pdfs.py     # parse PDFs to text
python build_index.py     # embed and index
```

---

## Index repository

The pre-built vector index is stored separately at [github.com/aekankpatel/finrag-index](https://github.com/aekankpatel/finrag-index). It is kept in its own repo to avoid bloating the main repo with large binary files. The `default__vector_store.json` file (~124 MB) is stored using Git LFS because it exceeds GitHub's 100 MB file size limit.
