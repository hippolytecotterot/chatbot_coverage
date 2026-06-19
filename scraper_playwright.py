import io
import os
import ssl
import time
from pathlib import Path

import cloudscraper
import pypdf
import trafilatura
import urllib3
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from requests.adapters import HTTPAdapter

_CA_CERT = Path(__file__).parent / "ca.pem"
_VERIFY = str(_CA_CERT) if _CA_CERT.exists() else True

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class _LaxSSLAdapter(HTTPAdapter):
    """Fallback adapter for servers with outdated TLS cipher suites."""
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.set_ciphers("DEFAULT:@SECLEVEL=1")
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


_scraper = cloudscraper.create_scraper()
_scraper.verify = _VERIFY

_scraper_lax = cloudscraper.create_scraper()
_scraper_lax.mount("https://", _LaxSSLAdapter())

_FAILED_LOG = Path(__file__).parent / "failed_urls.txt"
_MAX_RETRIES = int(os.getenv("SCRAPER_MAX_RETRIES", "3"))
_RETRY_DELAY = float(os.getenv("SCRAPER_RETRY_DELAY", "2"))


def _is_pdf(url: str, content_type: str) -> bool:
    return url.lower().endswith(".pdf") or "pdf" in content_type.lower()


def _extract_pdf(content: bytes) -> str | None:
    try:
        reader = pypdf.PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text if len(text) > 100 else None
    except Exception:
        return None


def _parse_html(html: str) -> str | None:
    content = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
    )
    return content if content and len(content) > 100 else None


def _parse_html_bs4(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return text if len(text) > 100 else None


def _fetch_with_playwright(url: str) -> str | None:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(ignore_https_errors=True)
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
            html = page.content()
            browser.close()
            return html
    except Exception:
        return None


class ArticleScraper:
    def _try_scrape(self, url: str) -> str | None:
        try_playwright = False
        for scraper in (_scraper, _scraper_lax):
            try:
                resp = scraper.get(url, timeout=15)
                if resp.status_code == 200:
                    if _is_pdf(url, resp.headers.get("Content-Type", "")):
                        return _extract_pdf(resp.content)
                    content = _parse_html(resp.text) or _parse_html_bs4(resp.text)
                    if content:
                        return content
                    try_playwright = True
                elif resp.status_code in (403, 429, 503):
                    try_playwright = True
            except Exception as exc:
                if "SSL" in type(exc).__name__ or "SSL" in str(exc):
                    continue
                try_playwright = True

        if try_playwright:
            html = _fetch_with_playwright(url)
            if html:
                content = _parse_html(html) or _parse_html_bs4(html)
                if content:
                    return content
        return None

    def scrape(self, url: str) -> str | None:
        for attempt in range(_MAX_RETRIES):
            result = self._try_scrape(url)
            if result:
                return result
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY)
        with open(_FAILED_LOG, "a", encoding="utf-8") as f:
            f.write(url + "\n")
        return None

    def scrape_debug(self, url: str) -> dict:
        """Scrape a URL and return a detailed status dict instead of swallowing errors."""
        result = {"url": url, "ok": False, "method": None, "chars": 0, "error": None}
        errors: list[str] = []
        use_playwright = False

        for label, scraper in (("cloudscraper", _scraper), ("cloudscraper_lax", _scraper_lax)):
            try:
                resp = scraper.get(url, timeout=15)
                if resp.status_code == 200:
                    if _is_pdf(url, resp.headers.get("Content-Type", "")):
                        content = _extract_pdf(resp.content)
                        if content:
                            result.update(ok=True, method=f"{label}+pypdf", chars=len(content))
                            return result
                        errors.append(f"{label}+pypdf: extraction PDF échouée")
                    else:
                        content = _parse_html(resp.text)
                        if content:
                            result.update(ok=True, method=f"{label}+trafilatura", chars=len(content))
                            return result
                        content = _parse_html_bs4(resp.text)
                        if content:
                            result.update(ok=True, method=f"{label}+bs4", chars=len(content))
                            return result
                        errors.append(f"{label} (200): trafilatura et bs4 ont échoué (contenu probablement JS-rendu)")
                        use_playwright = True
                else:
                    errors.append(f"{label}: HTTP {resp.status_code}")
                    if resp.status_code in (403, 429, 503):
                        use_playwright = True
            except Exception as exc:
                errors.append(f"{label}: {type(exc).__name__}: {exc}")
                if "SSL" in type(exc).__name__ or "SSL" in str(exc):
                    continue
                use_playwright = True

        if use_playwright or not errors:
            html = _fetch_with_playwright(url)
            if html:
                content = _parse_html(html)
                if content:
                    result.update(ok=True, method="playwright+trafilatura", chars=len(content))
                    return result
                content = _parse_html_bs4(html)
                if content:
                    result.update(ok=True, method="playwright+bs4", chars=len(content))
                    return result
                errors.append("playwright: trafilatura et bs4 n'ont pas pu extraire le contenu")
            else:
                errors.append("playwright: échec du chargement de la page")

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
