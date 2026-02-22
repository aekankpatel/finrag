
# 📊 FinRAG — Financial Retrieval-Augmented Intelligence System

A fully local RAG (Retrieval-Augmented Generation) system for querying and analyzing financial documents including SEC filings, earnings call transcripts, and macro research reports — with zero OpenAI API costs.

---

## 🎯 What It Does

- Ask natural language questions across 25+ financial documents

- Get grounded answers with cited sources and relevance scores

- Auto-detects which company you're asking about

- Compare two companies side by side on the same question

- Fully local — runs on your machine using Ollama

---

## 🏗️ Architecture

```

PDFs / Documents

      ↓

Text Extraction & Cleaning (ingest_pdfs.py)

      ↓

Chunking (TokenTextSplitter, 128 tokens)

      ↓

Embeddings (nomic-embed-text via Ollama)

      ↓

Vector Index (LlamaIndex SimpleVectorStore)

      ↓

User Query

      ↓

Semantic Retrieval (top-k chunks)

      ↓

LLM Generation (llama3.1:8b via Ollama)

      ↓

Answer + Source Citations

```

---

## 🧰 Tech Stack

| Component | Tool |

|---|---|

| LLM | llama3.1:8b (Ollama) |

| Embeddings | nomic-embed-text (Ollama) |

| Vector Store | LlamaIndex SimpleVectorStore |

| Document Parsing | LlamaIndex + custom ingestion |

| UI | Streamlit |

| Language | Python 3.13 |

---

## 📁 Document Corpus (25 files)

- **SEC Filings**: Meta 10-K, Amazon 10-K/10-Q, NVIDIA 10-Q, Apple 10-K/10-Q, Tesla 10-K/10-Q, Microsoft 10-Q, Goldman Sachs BDC 10-Q, Bank of America Annual Report

- **Earnings Transcripts**: Amazon Q4, BofA Q4, Goldman Sachs Q4, JPMorgan Q4, NVIDIA Q4, Walmart Q4

- **Macro Research**: EY Global Banking Outlook 2025, Goldman Sachs 2026 M&A Outlook, World Bank Global Economic Prospects Jan 2026, Capital Markets Forecast 2026

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+

- [Ollama](https://ollama.com) installed and running

### 1. Clone the repo

```bash

git clone https://github.com/YOUR_USERNAME/finrag.git

cd finrag

```

### 2. Install dependencies

```bash

pip install llama-index llama-index-embeddings-ollama llama-index-llms-ollama streamlit tqdm faiss-cpu

```

### 3. Pull Ollama models

```bash

ollama pull llama3.1:8b

ollama pull nomic-embed-text

```

### 4. Add your documents

Place `.pdf` files in `finrag/data/raw/` then run:

```bash

python finrag/ingest_pdfs.py

```

### 5. Build the vector index

```bash

python finrag/build_index.py

```

### 6. Launch the UI

```bash

streamlit run finrag/app.py

```

---

## 🖥️ Features

- **🔍 Auto-detect** — detects company name from your question automatically

- **⚖️ Compare mode** — side-by-side answers from two different documents

- **📊 Confidence bar** — visual retrieval confidence score per query

- **💬 Chat history** — scrollable Q&A log within the session

- **⬇️ Export** — download any answer + sources as a `.txt` file

- **🎛️ Manual override** — filter to any specific document via dropdown

---

## 📊 Evaluation

Evaluated on 10 domain-specific financial questions across 5 companies:

| Metric | Result |

|---|---|

| Total questions | 10 |

| Pass rate (score ≥ 0.5) | 8/10 (80%) |

| Average keyword hit score | 0.57 / 1.0 |

| Avg response time | ~20 seconds |

| LLM | llama3.1:8b (fully local) |

Evaluation script: `finrag/evaluate.py`

Full results: `finrag/eval_results.json`

---

## 📂 Project Structure

```

finrag/

├── data/

│   ├── raw/           # Original PDFs

│   └── processed/     # Cleaned .txt files

├── index/             # Vector store (auto-generated)

├── ingest_pdfs.py     # PDF → text pipeline

├── build_index.py     # Chunking + embedding + indexing

├── query.py           # Terminal query interface

├── app.py             # Streamlit web UI

├── evaluate.py        # Evaluation script

└── eval_results.json  # Evaluation output

```

---

## 💼 Skills Demonstrated

- Document ingestion & preprocessing pipeline

- Semantic search with vector embeddings

- Retrieval-Augmented Generation (RAG)

- Local LLM inference (no API costs)

- Evaluation framework for LLM outputs

- Full-stack AI application with Streamlit UI

---

## 🔮 Future Work

- LLM-based query routing (replace keyword detection)

- Re-ranking retrieved chunks for better precision

- Multi-turn conversation memory

- Deployment to cloud (Streamlit Cloud / HuggingFace Spaces)

---

*Built with LlamaIndex, Ollama, and Streamlit. Runs fully locally.*

