from pathlib import Path

_BASE = Path(__file__).parent
_ARTICLES_FILE = _BASE / "articles_rag.txt"


class DatabaseManager:
    def fetch_articles(self) -> list[tuple[str, str]]:
        """Return (title, source_url) for every URL listed in articles_rag.txt."""
        urls = [
            line.strip()
            for line in _ARTICLES_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        return [(url, url) for url in urls]
