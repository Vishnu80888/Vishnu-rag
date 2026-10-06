# Vishnu RAG (Profi — AI Study Abroad Advisor)

> Grounded, high-precision hybrid RAG assistant designed for the **ProfileSity** study-abroad platform ([codejobz.com](https://www.codejobz.com/)).

Vishnu RAG powers **Profi**, an AI study-abroad counselor that helps students navigate university admissions, program prerequisites, visa timelines, and verified advisors across 500+ global partner institutions.

---

## 📑 Table of Contents
1. [Core Features](#-core-features)
2. [Architecture Pipeline](#-architecture-pipeline)
3. [Project Directory Layout](#-project-directory-layout)
4. [Installation & Setup](#-installation--setup)
5. [Running the Application](#-running-the-application)
6. [API Endpoints & cURL Reference](#-api-endpoints--curl-reference)
7. [Semantic Embeddings & Providers](#-semantic-embeddings--providers)
8. [Configuration (.env) Reference](#-configuration-env-reference)
9. [Automated Test Suite & Fixtures](#-automated-test-suite--fixtures)
10. [Security & Design Practices](#-security--design-practices)

---

## 🌟 Core Features

- **Hybrid Dual-Retrieval**: Combines **Dense vector embeddings** (semantic understanding) with **BM25 sparse search** (exact keywords, counselor names, university codes).
- **Reciprocal Rank Fusion (RRF $k=60$)**: Mathematical rank-blending that fairly combines dense and sparse candidate hits without score-scale biases.
- **Multiple Embedding Providers**: Supports **FastEmbed** (`BAAI/bge-small-en-v1.5`, local pure-Python ONNX semantic embeddings), **OpenAI-compatible** (LM Studio / vLLM / OpenAI), or **Hash Embedder** (instant deterministic testing).
- **Boundary-Aware Sliding-Window Chunking**: 800-character windows with 100-character overlap, splitting strictly on paragraph breaks (`\n\n`) and sentence stops (`. `).
- **Deterministic 24-char SHA-256 Chunk IDs**: Ensures reproducible, idempotent upserts across re-indexing runs.
- **Strict Grounding & Fail-Closed Guardrails**: If information is missing from the indexed documents, the system returns a safe fallback (`NO_ANSWER`) rather than hallucinating admissions requirements or visa policies.
- **Traceable Citations**: Every answer provides exact anchors formatted as `[doc_id#chunk_index]` alongside source URLs and fusion scores.
- **Clean Dependency Injection**: Built on FastAPI's `Depends()`, enabling clean testing via `app.dependency_overrides` with zero mutable module-level globals.
- **Path-Traversal Guardrails**: `/ingest` is restricted to configured project document directories, preventing callers from reading arbitrary server folders.

---

## 🧠 Architecture Pipeline

```text
User Question
     │
     ├──► Dense Search ─────► Top-20 Semantic Candidates ──┐
     │    (FastEmbed / OpenAI / Hash)                      │
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

1. **Ingestion**: Markdown documents in `data/docs/` contain structured metadata headers (e.g. `<!-- source: https://... | page: universities -->`).
2. **Chunking**: Text is split cleanly using paragraph-preferring boundary detection with deterministic SHA-256 IDs.
3. **Dual Indexing**: Chunks are stored with both dense vectors and BM25 term frequencies.
4. **Retrieval**: Both search engines retrieve candidates in parallel.
5. **RRF Blending**: Each candidate's score is computed via:
   $$\text{Score}(c) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{60 + \text{rank}_m(c)}$$
6. **Synthesis**: The top fused chunks are provided to the LLM with strict instructions to quote from the context and cite chunk IDs.

---

## 📁 Project Directory Layout

The repository uses a single, flat root structure:

```text
vishnu-rag/
├── app/
│   ├── __init__.py
│   ├── chunking.py          # Sliding-window chunker with boundary heuristics
│   ├── config.py            # Pydantic Settings & environment variables
│   ├── main.py              # FastAPI app, routes, Depends() injection & lifespan
│   ├── models.py            # Pydantic models (Chunk, Hit, QueryRequest, IngestRequest)
│   ├── providers.py         # Embedders (FastEmbed, Hash, OpenAI) & LLMs (Echo, OpenAI)
│   ├── retrieval.py         # Reciprocal Rank Fusion (RRF k=60) implementation
│   ├── service.py           # Core RagService pipeline (ingest, retrieve, answer)
│   └── stores.py            # In-Memory & embedded Qdrant hybrid storage adapters
├── data/
│   └── docs/                # Production Markdown corpus
│       ├── about.md         # Platform mission, leadership & values
│       ├── advisors.md      # 500+ verified education counselors
│       ├── contact_and_links.md # Portal logins, student signups & URLs
│       ├── courses.md       # 100+ expert test-prep & admissions courses
│       ├── expos.md         # 50+ virtual education fairs & webinars
│       ├── faq.md           # Admission timelines, visa & SOP rules
│       ├── home.md          # ProfileSity highlights & student success stories
│       ├── universities.md  # 500+ partner universities across 50+ countries
│       └── crawled/         # Snapshots crawled from live site
├── scripts/
│   ├── client_demo.py       # Example Python HTTP client script (httpx)
│   └── crawl_site.py        # Static crawler for codejobz.com
├── tests/
│   ├── fixtures/
│   │   └── docs/            # Isolated test fixture documents (never tied to prod docs)
│   │       ├── advisors.md
│   │       ├── expos.md
│   │       └── faq.md
│   ├── test_all_cases.py    # Comprehensive test suite (unit, API, security, DI)
│   └── test_rag.py          # Core RAG regression suite
├── .env.example             # Environment configuration template
├── .gitignore               # Comprehensive ignores (.env, caches, venvs, logs)
├── pyproject.toml           # PEP 621 packaging metadata & pytest config
├── pytest.ini               # Pytest import configuration
├── requirements.txt         # Pinned production dependencies
├── requirements-dev.txt     # Pinned development/testing dependencies
└── run.py                   # Unified CLI launcher
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.10+ (Python 3.11 recommended)

### 2. Create Virtual Environment & Install Dependencies
Open PowerShell or your terminal in the repository root:

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements-dev.txt
```

### 3. Generate Configuration
Create your local `.env` from the template:

```powershell
python run.py setup
```

---

## 💻 Running the Application

### Option A: Direct with Uvicorn (Recommended for development)
```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Option B: Unified CLI Launcher
```powershell
# Start API server
python run.py serve

# Ask question directly from terminal
python run.py ask "What is ProfileSity?"

# Ingest document directory
python run.py ingest data/docs

# Re-crawl static pages
python run.py crawl

# Run test suite
python run.py test
```

---

## 🌐 API Endpoints & cURL Reference

Interactive documentation is available at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 1. Healthcheck (`GET /health`)
```bash
curl -X GET "http://127.0.0.1:8000/health"
```
**Response (200 OK):**
```json
{"status": "healthy"}
```

### 2. RAG Query (`POST /query`)
```bash
curl -X POST "http://127.0.0.1:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "How long does the admission process take?", "top_k": 3}'
```
**PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/query" -Method Post -ContentType "application/json" -Body '{"question": "How long does the admission process take?", "top_k": 3}'
```
**Response (200 OK):**
```json
{
  "answer": "The typical timeline from profile assessment to receiving an admission offer is 8-12 weeks [faq.md#0].",
  "grounded": true,
  "sources": [
    {
      "doc_id": "faq.md",
      "url": "https://www.codejobz.com/",
      "chunk_id": "03bb5454652c6f1d06371cf7",
      "score": 0.0328
    }
  ]
}
```

### 3. Raw Ranked Retrieval (`POST /retrieve`)
```bash
curl -X POST "http://127.0.0.1:8000/retrieve" \
     -H "Content-Type: application/json" \
     -d '{"question": "virtual student fair", "top_k": 3, "filters": {"page": "expos"}}'
```

### 4. Ingest Documents (`POST /ingest`)
```bash
curl -X POST "http://127.0.0.1:8000/ingest" \
     -H "Content-Type: application/json" \
     -d '{"directory": "data/docs"}'
```
*Note: Paths outside the allowed project directory are rejected with `HTTP 400 Bad Request` to prevent path traversal.*

### 5. Inspect Sources (`GET /sources` and `GET /sources/{doc_id}`)
```bash
curl -X GET "http://127.0.0.1:8000/sources"
curl -X GET "http://127.0.0.1:8000/sources/faq.md"
```

---

## 🧩 Semantic Embeddings & Providers

In `.env`, configure `EMBEDDING_PROVIDER`:

| Provider | Setting | Description |
|---|---|---|
| **FastEmbed (Recommended)** | `EMBEDDING_PROVIDER=fastembed` | Real offline neural embeddings via ONNX (`BAAI/bge-small-en-v1.5`, 384 dim). No GPU or external server required. |
| **OpenAI-Compatible** | `EMBEDDING_PROVIDER=openai_compatible` | Uses external LM Studio, vLLM, Ollama, or OpenAI embedding endpoints. |
| **Local Hash** | `EMBEDDING_PROVIDER=local` | Deterministic MD5 feature hashing (256 dim). Ideal for instantaneous offline unit tests. |

---

## ⚙️ Configuration (.env) Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `local` | `local` (EchoLLM) or `openai_compatible` (LM Studio / vLLM / OpenAI) |
| `EMBEDDING_PROVIDER` | `fastembed` | `fastembed` (Local neural ONNX), `openai_compatible`, or `local` |
| `FASTEMBED_MODEL` | `BAAI/bge-small-en-v1.5` | ONNX model name for FastEmbed |
| `VECTOR_STORE_PROVIDER` | `memory` | `memory` (RAM) or `qdrant` (Embedded disk or remote server) |
| `QDRANT_PATH` | `./data/qdrant_local` | Directory for pure-Python embedded Qdrant (no Docker needed) |
| `OPENAI_BASE_URL` | `http://localhost:1234/v1` | URL for remote LLM / embedding server |
| `OPENAI_API_KEY` | `lm-studio` | API key or token |
| `LLM_MODEL` | `qwen` | Model identifier loaded in LLM server |
| `DOCS_DIR` | `data/docs` | Default directory for document ingestion |
| `CHUNK_SIZE` | `800` | Target characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap characters between chunks |
| `TOP_K` | `5` | Final chunk count passed to LLM |
| `MAX_CANDIDATES` | `20` | Candidate pool size retrieved per search engine before RRF |

---

## 🧪 Automated Test Suite & Fixtures

### Running Tests
Execute all tests with pytest:

```powershell
pytest -v
```

Or using the CLI launcher:
```powershell
python run.py test
```

### Dedicated Test Fixtures
Tests are **never tied to production documents**. All retrieval and API test cases run against isolated test fixtures in `tests/fixtures/docs/`. Edits or updates to production docs in `data/docs/` will never cause tests to break.

### Test Categories (20 Passing Tests)
1. **Unit Tests (Chunking & Tokenizer)**: Verifies sentence/paragraph boundary detection, stopword filtering, chunk index tracking, and deterministic SHA-256 IDs.
2. **Unit Tests (Embedders & RRF Math)**: Verifies HashEmbedder normalization, FastEmbed neural embedding generation, Reciprocal Rank Fusion scoring, and empty-list edge cases.
3. **Integration Tests (RAG Service & Retrieval)**: Verifies ranking accuracy, multi-document retrieval, and metadata filtering against isolated test fixtures.
4. **API Contract & Dependency Injection**: Verifies HTTP 200 responses, schema validation, and clean testing via `app.dependency_overrides[main.get_rag_service]`.
5. **Security & Validation Tests**: Verifies rejection of path-traversal directory inputs (HTTP 400), missing fields (HTTP 422), empty strings (HTTP 422), and out-of-bounds `top_k` (HTTP 422).
6. **Safety & Fail-Closed Guardrails**: Verifies that queries against empty or ungrounded knowledge bases return safe fallback responses (`NO_ANSWER`) with zero hallucination.

---

## 🔒 Security & Design Practices

1. **No Committed Secrets or Junk**:
   - `.env`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `*.log`, and `data/qdrant_local/` are strictly ignored by `.gitignore`.
2. **Pinned Dependencies**:
   - `requirements.txt` and `pyproject.toml` specify exact pinned package versions for reproducible builds.
3. **Zero Module-Level Global State**:
   - `app/main.py` uses FastAPI's `Depends(get_rag_service)` pattern for idiomatic dependency injection and simple mock overrides.
4. **Path-Traversal Protection**:
   - Ingestion paths are strictly validated to reside within the project boundary.
