"""Provider contracts + adapters. One active implementation per capability."""
import hashlib
import math
import re
from typing import Protocol

import httpx

from .config import Settings

TOKEN = re.compile(r"[a-z0-9]+")


STOPWORDS = frozenset("a an the is are was were be been of to in on for and or with as at by it its this that "
                      "who what which when where how do does did can i you we my your our from".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class LLM(Protocol):
    async def generate(self, system: str, user: str) -> str: ...


class HashEmbedder:
    """Deterministic offline embedder (feature hashing). For tests/dev only."""

    def __init__(self, dim: int):
        self.dim = dim

    async def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in tokenize(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class OpenAICompatEmbedder:
    def __init__(self, s: Settings):
        self.s = s

    async def embed(self, texts):
        if not texts:
            return []
        out = []
        batch_size = 32
        async with httpx.AsyncClient(timeout=120) as c:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                r = await c.post(
                    f"{self.s.openai_base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.s.openai_api_key}"},
                    json={"model": self.s.embedding_model, "input": batch},
                )
                r.raise_for_status()
                data = sorted(r.json()["data"], key=lambda d: d["index"])
                out.extend(d["embedding"] for d in data)
        return out


class EchoLLM:
    """Offline LLM stand-in: returns the retrieved context verbatim."""

    async def generate(self, system, user):
        return "Based on the retrieved sources:\n" + user.split("CONTEXT:\n", 1)[-1].split("\n\nQUESTION:", 1)[0]


class OpenAICompatLLM:
    def __init__(self, s: Settings):
        self.s = s

    async def generate(self, system, user):
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(
                f"{self.s.openai_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.s.openai_api_key}"},
                json={
                    "model": self.s.llm_model,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"].get("content") or ""
            # Strip <think> tags from reasoning models
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content


class FastEmbedEmbedder:
    """Local, offline neural semantic embeddings via ONNX (fastembed)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import asyncio
        loop = asyncio.get_running_loop()

        def _compute():
            model = self._get_model()
            embeddings_gen = model.embed(texts)
            return [vec.tolist() for vec in embeddings_gen]

        return await loop.run_in_executor(None, _compute)


def build_embedder(s: Settings) -> Embedder:
    if s.embedding_provider == "fastembed":
        return FastEmbedEmbedder(getattr(s, "fastembed_model", "BAAI/bge-small-en-v1.5"))
    elif s.embedding_provider == "openai_compatible":
        return OpenAICompatEmbedder(s)
    else:
        return HashEmbedder(s.embedding_dimension)


def build_llm(s: Settings) -> LLM:
    return OpenAICompatLLM(s) if s.llm_provider == "openai_compatible" else EchoLLM()
