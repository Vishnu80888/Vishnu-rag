"""Call the running API from Python (start it first: python run.py serve)."""
import httpx

BASE = "http://127.0.0.1:8000"
with httpx.Client(base_url=BASE, timeout=120) as c:
    print(c.post("/ingest", json={"directory": "data/docs"}).json())
    r = c.post("/query", json={"question": "How long does the admission process take?"}).json()
    print(r["answer"])
    print(r["sources"])
    print(c.get("/sources").json())
