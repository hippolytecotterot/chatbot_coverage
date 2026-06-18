import os
import sys
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from database import DatabaseManager

_BASE = Path(__file__).parent
_LOCAL_MODEL_PATH = _BASE / "models" / "all-MiniLM-L6-v2"

VECTOR_STORE_PATH = str(_BASE / "vectorstore")
EMBEDDING_MODEL = str(_LOCAL_MODEL_PATH) if _LOCAL_MODEL_PATH.exists() else "all-MiniLM-L6-v2"


class RAGManager:
    def __init__(self, vector_store_path: str = VECTOR_STORE_PATH, embedding_model: str = EMBEDDING_MODEL):
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def load_existing(self) -> Chroma:
        return Chroma(persist_directory=self.vector_store_path, embedding_function=self.get_embeddings())

    def index_from_db(self) -> Chroma:
        if os.getenv("USE_PLAYWRIGHT", "false").lower() == "true":
            from scraper_playwright import ArticleScraper, _FAILED_LOG
            print("  Mode scraping : Playwright activé (fallback sur 403 et contenu JS-rendu)")
        else:
            from scraper import ArticleScraper, _FAILED_LOG
            print("  Mode scraping : cloudscraper uniquement")

        _FAILED_LOG.write_text("", encoding="utf-8")

        db = DatabaseManager()
        scraper = ArticleScraper()

        rows = db.fetch_articles()
        if not rows:
            print("[Error] No articles with a source URL found in the database.")
            sys.exit(1)

        request_delay = float(os.getenv("SCRAPER_REQUEST_DELAY", "1"))
        print(f"  Found {len(rows)} article(s) in database.")
        docs: list[Document] = []
        for title, url in rows:
            print(f"  Scraping: {url}")
            content = scraper.scrape(url)
            if content:
                docs.append(Document(
                    page_content=content,
                    metadata={"source": url, "title": title},
                ))
                print(f"    ✓ {len(content)} chars")
            else:
                print("    ✗ Could not extract content")
            time.sleep(request_delay)

        if not docs:
            print("[Error] No articles could be scraped.")
            sys.exit(1)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        print(f"  {len(chunks)} chunks created from {len(docs)} article(s).")

        print("  Computing embeddings (first run may take a few minutes)...")
        vectorstore = Chroma.from_documents(
            chunks, self.get_embeddings(), persist_directory=self.vector_store_path
        )
        print(f"  Vector store saved to '{self.vector_store_path}/'.")
        return vectorstore

    def build_retriever(self, vectorstore: Chroma):
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

    def get_source_urls(self, docs) -> list[str]:
        """Return deduplicated source URLs from retrieved document chunks."""
        seen: set[str] = set()
        urls: list[str] = []
        for doc in docs:
            url = doc.metadata.get("source", "")
            if url and url not in seen:
                seen.add(url)
                urls.append(url)
        return urls
