# Vishnu RAG (Profi — AI Study Abroad Advisor)

> Grounded, high-precision hybrid RAG assistant designed for the **ProfileSity** study-abroad platform ([codejobz.com](https://www.codejobz.com/)).

Vishnu RAG powers **Profi**, an AI counselor that helps students navigate university admissions, program prerequisites, visa timelines, and verified advisors across 500+ global partner institutions.

---

## 📑 Table of Contents
1. [Core Features](#-core-features)
2. [How It Works (Architecture Pipeline)](#-how-it-works-architecture-pipeline)
3. [Project Directory Layout](#-project-directory-layout)
4. [Quick Start & Setup](#-quick-start--setup)
5. [CLI Commands Guide](#-cli-commands-guide)
6. [API Endpoints Reference](#-api-endpoints-reference)
7. [Configuration (.env) Reference](#-configuration-env-reference)
8. [Testing Suite](#-testing-suite)
9. [Troubleshooting & FAQ](#-troubleshooting--faq)

---

## 🌟 Core Features

- **Hybrid Dual-Retrieval**: Combines **Dense vector embeddings** (semantic understanding) with **BM25 sparse search** (exact keywords, counselor names, university codes).
- **Reciprocal Rank Fusion (RRF $k=60$)**: Mathematical rank-blending that fairly combines dense and sparse hits without score-scale biases.
- **Boundary-Aware Sliding-Window Chunking**: 800-character windows with 100-character overlap, splitting strictly on paragraph breaks (`\n\n`) and sentence stops (`. `).
- **Deterministic 24-char SHA-256 Chunk IDs**: Ensures reproducible, idempotent upserts across re-indexing runs.
- **Strict Grounding & Fail-Closed Guardrails**: If information is missing from the indexed documents, the system returns a safe fallback (`NO_ANSWER`) rather than hallucinating admissions requirements or visa policies.
- **Traceable Citations**: Every answer provides exact anchors formatted as `[doc_id#chunk_index]` alongside source URLs and fusion scores.
- **Provider-Agnostic Flexibility**: Works 100% offline with local hash embeddings and in-memory storage, or seamlessly switches to LM Studio / vLLM / OpenAI and local embedded Qdrant.

---

## 🧠 How It Works (Architecture Pipeline)

```
User Question
     │
     ├──► Dense Search ─────► Top-20 Semantic Candidates ──┐
     │                                                     │
     └──► BM25 Sparse Search ─► Top-20 Keyword Candidates  ──┴─► RRF Fusion (k=60)
                                                                    │
                                                              Top-5 Fused Chunks
                                                                    │
                                                    Context Construction [doc_id#chunk]
                                                                    │
                                                               LLM Reasoner
                                                                    │
                                                          Grounded Answer + Sources
```

1. **Ingestion**: Markdown documents in `data/docs/` contain metadata headers (e.g. `<!-- source: https://... | page: universities -->`).
2. **Chunking**: Text is split cleanly using paragraph-preferring boundary detection with deterministic SHA-256 IDs.
3. **Dual Indexing**: Chunks are stored with both dense vectors and BM25 term frequencies.
4. **Retrieval**: Both search engines retrieve candidates in parallel.
5. **RRF Blending**: Each candidate's score is computed via:
   $$\text{Score}(c) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{60 + \text{rank}_m(c)}$$
6. **Synthesis**: The top fused chunks are provided to the LLM with a strict instruction to quote from the context and cite chunk IDs.

---

## 📁 Project Directory Layout

```text
vishnu-rag/
├── app/
│   ├── chunking.py       # Sliding-window chunker with boundary heuristics
│   ├── config.py         # Pydantic Settings & environment variables
│   ├── main.py           # FastAPI application endpoints & lifespan loader
│   ├── models.py         # Pydantic models (Chunk, Hit, QueryRequest, IngestRequest)
│   ├── providers.py      # Embedder & LLM contracts (Local Hash, OpenAI-compatible)
│   ├── retrieval.py      # Reciprocal Rank Fusion (RRF k=60) implementation
│   ├── service.py        # Core RagService pipeline (ingest, retrieve, answer)
│   └── stores.py         # In-Memory & embedded Qdrant hybrid storage adapters
├── data/
│   └── docs/             # Grounded Markdown corpus
│       ├── about.md      # Platform mission & background
│       ├── advisors.md   # Verified education counselors
│       ├── courses.md    # 10K+ programs and courses overview
│       ├── expos.md      # Virtual student expos & webinars
│       ├── faq.md        # Admission timelines, visa & SOP rules
│       ├── home.md       # ProfileSity highlights & student stories
│       ├── universities.md # 500+ partner universities
│       └── crawled/      # Snapshots crawled from live site
├── scripts/
│   ├── client_demo.py    # Example HTTP client script
│   └── crawl_site.py     # Static crawler for codejobz.com
├── tests/
│   ├── test_rag.py       # Core regression tests
│   └── test_all_cases.py # Comprehensive unit, integration & API test suite
├── .env                  # Active environment settings
├── .env.example          # Environment template
├── requirements.txt      # Core runtime dependencies
├── requirements-dev.txt  # Test & dev dependencies (pytest, httpx)
└── run.py                # Unified CLI launcher
```

---

## 🚀 Quick Start & Setup

### 1. Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- Optional: LM Studio running locally if using neural LLMs/embeddings

### 2. Installation
Open PowerShell or your terminal and navigate to the project root:
```powershell
cd C:\Users\user\Downloads\vishnu-rag\vishnu-rag
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

### 3. Initialize Configuration
Generate your `.env` configuration file:
```powershell
python run.py setup
```

---

## 💻 CLI Commands Guide

The unified CLI `run.py` provides simple one-command operations:

| Command | Description | Example |
|---|---|---|
| `python run.py test` | Run the pytest regression suite | `python run.py test` |
| `python run.py ask "<query>"` | Ask a question directly via CLI | `python run.py ask "What is ProfileSity?"` |
| `python run.py serve` | Start the local FastAPI server | `python run.py serve` (runs on `http://127.0.0.1:8000`) |
| `python run.py ingest [dir]` | Manually ingest a directory of docs | `python run.py ingest data/docs` |
| `python run.py crawl` | Re-crawl static pages from codejobz.com | `python run.py crawl` |

---

## 🌐 API Endpoints Reference

Once running (`python run.py serve`), access the interactive Swagger documentation at:  
👉 **http://127.0.0.1:8000/docs**

### 1. Health Check
```http
GET /health
```
**Response (200 OK):**
```json
{
  "status": "healthy"
}
```

### 2. Query Question (Main RAG)
```http
POST /query
Content-Type: application/json

{
  "question": "What universities are available?",
  "top_k": 5,
  "filters": {}
}
```
**Response (200 OK):**
```json
{
  "answer": "ProfileSity offers verified profiles for over 500 universities worldwide, including MIT, Stanford, Oxford, and Cambridge [universities.md#0].",
  "grounded": true,
  "sources": [
    {
      "doc_id": "universities.md",
      "url": "https://www.codejobz.com/universities",
      "chunk_id": "90e633d159a68c741e97d1be",
      "score": 0.0328
    }
  ]
}
```

### 3. Raw Ranked Retrieval
```http
POST /retrieve
Content-Type: application/json

{
  "question": "visa assistance",
  "top_k": 3,
  "filters": {"page": "faq"}
}
```
**Response (200 OK):**
Returns structured chunk objects with exact fused RRF scores and metadata.

### 4. Sources Catalog
```http
GET /sources
```
Returns all indexed document filenames and their total chunk counts.

---

## ⚙️ Configuration (.env) Reference

| Variable | Default | Purpose / Recommended Value |
|---|---|---|
| `LLM_PROVIDER` | `local` | `local` (EchoLLM for offline tests) or `openai_compatible` |
| `EMBEDDING_PROVIDER` | `local` | `local` (Fast hash embedder) or `openai_compatible` |
| `VECTOR_STORE_PROVIDER` | `memory` | `memory` (RAM) or `qdrant` (Persistent disk/Docker) |
| `OPENAI_BASE_URL` | `http://localhost:1234/v1` | URL for LM Studio / vLLM (e.g. `http://192.168.88.10:1234/v1`) |
| `OPENAI_API_KEY` | `lm-studio` | API key or token for LLM endpoint |
| `LLM_MODEL` | `qwen` | Model identifier in LM Studio (e.g. `qwen3.6-14b-a3b-fablevibes@q5_k_m`) |
| `EMBEDDING_MODEL` | `text-embedding-model` | Embedding model (e.g. `text-embedding-bge-m3@q8_0`) |
| `EMBEDDING_DIMENSION`| `256` | Vector dimensions (e.g., `1024` for BGE-M3, `256` for hash) |
| `QDRANT_PATH` | `""` | Local disk folder for pure-Python embedded Qdrant (e.g. `./data/qdrant_local`) |
| `CHUNK_SIZE` | `800` | Target characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlapping characters between consecutive chunks |
| `TOP_K` | `5` | Maximum number of chunks passed to the LLM |
| `MAX_CANDIDATES` | `20` | Candidate pool size retrieved per search engine before RRF |

---

## 🧪 Testing Suite

Run all test suites at any time:

```powershell
# Run baseline tests
python run.py test

# Run the comprehensive multi-category test suite
pytest -v tests\test_all_cases.py
```

### Verified Test Categories
- **Chunking & Tokenizer**: Verifies boundary heuristics, stopword stripping, and deterministic SHA-256 IDs.
- **RRF Math**: Verifies exact Reciprocal Rank Fusion scoring $\sum \frac{1}{60 + \text{rank}}$ and tie-breaking.
- **Grounded Retrieval**: Verifies multi-document recall and metadata filtering (`page: expos`, etc.).
- **Fail-Closed Safety**: Verifies that unindexed queries safely return `NO_ANSWER` to eliminate hallucinations.
- **Contract & Validation**: Verifies HTTP 200 responses for valid queries and HTTP 422 for malformed payloads.

---

## ❓ Troubleshooting & FAQ

### 1. `Cannot find path 'app\service.py'` in PowerShell
- **Cause**: Your terminal is in the outer directory (`C:\Users\user\Downloads\vishnu-rag`).
- **Fix**: Run `cd vishnu-rag` so you are inside the folder containing `app/`, `data/`, and `tests/`.

### 2. `HTTP 422 Unprocessable Entity` on `/query`
- **Cause**: Sending `{"query": "..."}` instead of `{"question": "..."}`.
- **Fix**: The API schema expects `question`:
  ```json
  { "question": "Your question here" }
  ```

### 3. How to use persistent vector storage?
- In `.env`, set:
  ```env
  VECTOR_STORE_PROVIDER=qdrant
  QDRANT_PATH=./data/qdrant_local
  ```
- This runs Qdrant locally in pure Python without requiring Docker.
