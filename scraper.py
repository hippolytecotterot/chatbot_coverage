import os
from pathlib import Path

import requests
import trafilatura

_CA_CERT = Path(__file__).parent / "ca.pem"

# Make trafilatura (which uses requests internally) pick up the corporate CA cert.
if _CA_CERT.exists():
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(_CA_CERT))

_VERIFY = str(_CA_CERT) if _CA_CERT.exists() else True

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

_FAILED_LOG = Path(__file__).parent / "failed_urls.txt"


class ArticleScraper:
    def scrape(self, url: str) -> str | None:
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=False,
                    no_fallback=False,
                )
                if text and len(text) > 100:
                    return text
            resp = requests.get(url, headers=_HEADERS, timeout=15, verify=_VERIFY)
            if resp.status_code == 200:
                content = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
                if content and len(content) > 100:
                    return content
        except Exception:
            pass
        with open(_FAILED_LOG, "a", encoding="utf-8") as f:
            f.write(url + "\n")
        return None
