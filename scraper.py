import os
from pathlib import Path

import cloudscraper
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
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

_cloudscraper = cloudscraper.create_scraper()
_cloudscraper.verify = _VERIFY

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
            resp = _cloudscraper.get(url, headers=_HEADERS, timeout=15)
            if resp.status_code == 200:
                content = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
                if content and len(content) > 100:
                    return content
        except Exception:
            pass
        with open(_FAILED_LOG, "a", encoding="utf-8") as f:
            f.write(url + "\n")
        return None

    def scrape_debug(self, url: str) -> dict:
        """Scrape a URL and return a detailed status dict instead of swallowing errors."""
        result = {"url": url, "ok": False, "method": None, "chars": 0, "error": None}
        errors: list[str] = []
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
                    result.update(ok=True, method="trafilatura.fetch_url", chars=len(text))
                    return result
                errors.append(
                    f"trafilatura.fetch_url: contenu récupéré mais extract a retourné "
                    f"{'rien' if not text else f'seulement {len(text)} chars (<100)'}"
                )
            else:
                errors.append("trafilatura.fetch_url: retourné None (page inaccessible ou vide)")

            resp = requests.get(url, headers=_HEADERS, timeout=15, verify=_VERIFY)
            if resp.status_code == 200:
                content = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
                if content and len(content) > 100:
                    result.update(ok=True, method="requests+extract", chars=len(content))
                    return result
                errors.append(
                    f"requests (200): extract a retourné "
                    f"{'rien' if not content else f'seulement {len(content)} chars (<100)'}"
                )
            else:
                errors.append(f"requests: HTTP {resp.status_code}")

            resp = _cloudscraper.get(url, headers=_HEADERS, timeout=15)
            if resp.status_code == 200:
                content = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
                if content and len(content) > 100:
                    result.update(ok=True, method="cloudscraper+extract", chars=len(content))
                    return result
                errors.append(
                    f"cloudscraper (200): extract a retourné "
                    f"{'rien' if not content else f'seulement {len(content)} chars (<100)'}"
                )
            else:
                errors.append(f"cloudscraper: HTTP {resp.status_code}")

        except Exception as exc:
            errors.append(f"Exception : {type(exc).__name__}: {exc}")

        result["error"] = " | ".join(errors)
        return result


_DEBUG_FILE = Path(__file__).parent / "debug.txt"


def _run_debug() -> None:
    if not _DEBUG_FILE.exists():
        print(f"Fichier introuvable : {_DEBUG_FILE}")
        return

    urls = [line.strip() for line in _DEBUG_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not urls:
        print("debug.txt est vide.")
        return

    scraper = ArticleScraper()
    ok, failed = 0, 0

    print(f"Scraping de {len(urls)} URL(s)...\n{'─' * 60}")
    for url in urls:
        res = scraper.scrape_debug(url)
        if res["ok"]:
            print(f"  [OK]  {url}")
            print(f"        méthode={res['method']}  chars={res['chars']}")
            ok += 1
        else:
            print(f"  [KO]  {url}")
            print(f"        {res['error']}")
            failed += 1
        print()

    print("─" * 60)
    print(f"Résultat : {ok} OK / {failed} échec(s) sur {len(urls)} URL(s)")


if __name__ == "__main__":
    _run_debug()
