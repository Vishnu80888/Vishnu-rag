"""One-file launcher.  Usage:
    python run.py crawl     # refresh data/docs from codejobz.com
    python run.py setup     # create .env from .env.example if missing
    python run.py test      # run the test suite
    python run.py serve     # start the API on http://127.0.0.1:8000
    python run.py ingest [dir]
    python run.py ask "your question"
"""
import asyncio
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def setup():
    env, ex = ROOT / ".env", ROOT / ".env.example"
    if not env.exists():
        shutil.copy(ex, env)
        print("Created .env")
    else:
        print(".env already exists")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd == "setup":
        setup()
    elif cmd == "crawl":
        import runpy
        runpy.run_path(str(ROOT / "scripts" / "crawl_site.py"), run_name="__main__")
    elif cmd == "test":
        import pytest
        sys.exit(pytest.main(["-q", str(ROOT / "tests")]))
    elif cmd == "serve":
        import uvicorn
        uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
    elif cmd in ("ingest", "ask"):
        from app.config import get_settings
        from app.service import RagService
        svc = RagService(get_settings())
        if cmd == "ingest":
            d = sys.argv[2] if len(sys.argv) > 2 else "data/docs"
            print(asyncio.run(svc.ingest_dir(d)))
        else:
            q = " ".join(sys.argv[2:]) or input("Question: ")
            asyncio.run(svc.ingest_dir("data/docs")) if get_settings().vector_store_provider == "memory" else None
            r = asyncio.run(svc.answer(q))
            print(r["answer"], "\n\nSources:", [(s["doc_id"], round(s["score"], 4)) for s in r["sources"]])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
