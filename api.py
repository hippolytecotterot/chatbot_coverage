"""
FastAPI server for the MindTheBot RAG.
Exposes POST /chat with SSE streaming.
Start with: uvicorn api:app --port 8000
Run from:   backend/src/main/python/
"""

import asyncio
import json
import shutil
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llm import LLMManager
from rag import RAGManager

_BASE = Path(__file__).parent
load_dotenv(_BASE / ".env")

app = FastAPI(title="MindTheBot RAG API")
SESSION_ID = str(uuid.uuid4())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_rag = RAGManager()
_llm = LLMManager()
_chain = None
_retriever = None


@app.on_event("startup")
async def startup() -> None:
    global _chain, _retriever
    vs_path = Path(_rag.vector_store_path)
    if vs_path.exists() and any(vs_path.iterdir()):
        vs = _rag.load_existing()
    else:
        vs = _rag.index_from_db()
    _retriever = _rag.build_retriever(vs)
    _chain = _llm.build_chain(_retriever)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "session_id": SESSION_ID}


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    loop = asyncio.get_event_loop()
    docs = await loop.run_in_executor(None, lambda: _retriever.invoke(req.question))
    source_urls = _rag.get_source_urls(docs)

    queue: asyncio.Queue = asyncio.Queue()

    def _stream_in_thread() -> None:
        try:
            for chunk in _chain.stream(req.question):
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, f"[Error] {exc}")
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=_stream_in_thread, daemon=True).start()

    async def generate():
        while True:
            chunk = await queue.get()
            if chunk is None:
                if source_urls:
                    yield f"data: {json.dumps({'sources': source_urls})}\n\n"
                break
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/reindex")
async def reindex() -> dict:
    global _chain, _retriever
    shutil.rmtree(_rag.vector_store_path, ignore_errors=True)
    vs = _rag.index_from_db()
    _retriever = _rag.build_retriever(vs)
    _chain = _llm.build_chain(_retriever)
    return {"status": "ok"}
