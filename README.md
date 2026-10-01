# Vishnu RAG

Grounded hybrid RAG assistant for the **ProfileSity** study-abroad platform (https://www.codejobz.com/).
Hybrid dense + BM25 retrieval fused with RRF (k=60), fail-closed answers with source URLs, provider-neutral LLM/embeddings, memory or Qdrant store. Pure Python.

## Run
```
python -m venv .venv          # then activate it
pip install -r requirements-dev.txt
python run.py setup           # creates .env
python run.py test
python run.py ask "How long does the admission process take?"
python run.py serve           # API docs at http://127.0.0.1:8000/docs
python run.py crawl           # re-crawl codejobz.com into data/docs/crawled
python scripts/client_demo.py # call the running API
```
Knowledge base: `data/docs/*.md` (home, about, faq, universities, courses, advisors, expos, links). Each file starts with `<!-- source: URL | page: name -->`, which becomes metadata (filter with `{"filters": {"page": "faq"}}`).

Limitation: individual university/course/advisor/expo listings load via JavaScript, so they are not in the static pages. Add their data as `.md` files in `data/docs/` (or ask for an API-based loader).

For real answers set `LLM_PROVIDER` / `EMBEDDING_PROVIDER=openai_compatible` in `.env`; for persistence set `VECTOR_STORE_PROVIDER=qdrant` with `QDRANT_PATH=./data/qdrant_local`.
