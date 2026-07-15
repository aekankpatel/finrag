---
title: FinRAG
emoji: 
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.41.1"
app_file: app.py
pinned: false
---

# FinRAG

A retrieval-augmented generation (RAG) system for querying financial documents. Ask natural language questions across earnings call transcripts, 10-K filings, 10-Q filings, and market outlook reports — and get grounded answers with cited source passages.

Live demo: [huggingface.co/spaces/aekankpatel/finrag](https://huggingface.co/spaces/aekankpatel/finrag)

---

## How it works

1. **Document ingestion** — Financial PDFs are parsed and chunked into passages. Each chunk is embedded using `BAAI/bge-small-en-v1.5` and stored in a LlamaIndex vector index.

2. **Query** — When you ask a question, the app retrieves the most relevant passages using cosine similarity search across the vector index.

3. **Answer generation** — The retrieved passages are passed as context to `llama-3.1-8b-instant` via the Groq API, which generates a grounded answer. The model is instructed to only use the provided context and not hallucinate.

4. **Source attribution** — Every answer shows the source document and relevance score for each retrieved passage.

```
User question
      |
      v
Vector similarity search (LlamaIndex + bge-small-en-v1.5)
      |
      v
Top-k passages retrieved
      |
      v
Groq API (llama-3.1-8b-instant) generates answer from context
      |
      v
Answer + source passages displayed
```

---

## Documents covered

| Company / Topic | Document type |
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

## Tech stack

- **Frontend** — Streamlit
- **Vector index** — LlamaIndex with SimpleVectorStore
- **Embeddings** — `BAAI/bge-small-en-v1.5` via HuggingFace
- **LLM** — `llama-3.1-8b-instant` via Groq API
- **Index storage** — GitHub ([aekankpatel/finrag-index](https://github.com/aekankpatel/finrag-index)), downloaded at startup
- **Hosting** — HuggingFace Spaces

---

## Features

- **Auto-detect** — Automatically identifies which document is most relevant to your question based on keyword matching
- **Manual filter** — Pin your query to a specific document for precise retrieval
- **Compare mode** — Ask the same question across two documents side by side
- **Confidence score** — Shows retrieval confidence based on the top similarity score
- **Export** — Download any answer with its source passages as a `.txt` file
- **Chat history** — Keeps a session history of questions and answers

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

```bash
streamlit run app.py
```

On first run, the vector index (~150 MB total) is downloaded from GitHub into `finrag/index/`.

---

## Project structure

```
finrag/
├── app.py              # Streamlit app
├── requirements.txt    # Dependencies
├── build_index.py      # Script used to build the vector index from raw docs
├── ingest_pdfs.py      # PDF parsing and text extraction
├── data/
│   ├── raw/            # Source PDFs
│   └── processed/      # Extracted text files
└── finrag/index/       # Vector index (downloaded at runtime)
```

---

## Index repository

The pre-built vector index lives at [github.com/aekankpatel/finrag-index](https://github.com/aekankpatel/finrag-index). The large vector store file (`default__vector_store.json`, ~124 MB) is stored using Git LFS.
