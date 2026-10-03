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

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
        uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True, app_dir=str(ROOT))
    elif cmd in ("ingest", "ask"):
        from app.config import get_settings
        from app.service import RagService
        svc = RagService(get_settings())
        docs_dir = str(ROOT / "data" / "docs")
        if cmd == "ingest":
            d = sys.argv[2] if len(sys.argv) > 2 else docs_dir
            print(asyncio.run(svc.ingest_dir(d)))
        else:
            q = " ".join(sys.argv[2:]) or input("Question: ")
            if get_settings().vector_store_provider == "memory":
                asyncio.run(svc.ingest_dir(docs_dir))
            else:
                sources = asyncio.run(svc.store.sources())
                if not sources:
                    asyncio.run(svc.ingest_dir(docs_dir))
            r = asyncio.run(svc.answer(q))
            print(r["answer"], "\n\nSources:", [(s["doc_id"], round(s["score"], 4)) for s in r["sources"]])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
