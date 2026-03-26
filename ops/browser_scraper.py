#!/usr/bin/env python3
"""
Lightweight browser scraper -- Crawl4AI primary, requests+BeautifulSoup fallback.
=================================================================================

Provides a simple async interface for scraping web pages that don't have APIs.
Used by HF Space agents and VM scripts for odds pages and sports data.

Priority:
  1. Crawl4AI (full browser rendering, JS support, returns markdown)
  2. requests/urllib fallback (no JS, but works everywhere)

Usage:
    from ops.browser_scraper import scrape_page, scrape_sync

    # Async
    result = await scrape_page("https://example.com")
    print(result["markdown"])

    # Sync wrapper
    result = scrape_sync("https://example.com")
    print(result["markdown"])

    # With CSS selectors for structured extraction
    result = await scrape_page("https://example.com", selectors={
        "title": "h1.page-title",
        "odds": "table.odds-table",
    })
    print(result["extracted"]["title"])
"""

import asyncio
import random
import re
import ssl
import time
import urllib.request
from typing import Optional

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1",
]

# Minimum delay between requests to the same domain (seconds)
MIN_DELAY = 1.0
MAX_DELAY = 3.0

# Track last request time per domain to rate-limit
_last_request: dict[str, float] = {}


def _get_domain(url: str) -> str:
    """Extract domain from URL for rate limiting."""
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc
    except Exception:
        return url


def _random_delay(domain: str) -> None:
    """Enforce a random delay between requests to the same domain."""
    now = time.monotonic()
    last = _last_request.get(domain, 0)
    elapsed = now - last
    needed = random.uniform(MIN_DELAY, MAX_DELAY)
    if elapsed < needed:
        time.sleep(needed - elapsed)
    _last_request[domain] = time.monotonic()


def _html_to_markdown(html: str) -> str:
    """
    Convert HTML to basic markdown. Lightweight fallback when Crawl4AI is not available.
    Strips tags and preserves text structure.
    """
    # Try html2text if available
    try:
        import html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.body_width = 0
        return h.handle(html)
    except ImportError:
        pass

    # Manual fallback: strip tags, preserve some structure
    text = html
    # Convert common block elements to newlines
    text = re.sub(r'<br\s*/?\s*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?(p|div|tr|li|h[1-6])[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<td[^>]*>', ' | ', text, flags=re.IGNORECASE)
    text = re.sub(r'</td>', '', text, flags=re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r'<[^>]+>', '', text)
    # Decode HTML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&nbsp;', ' ').replace('&quot;', '"').replace('&#39;', "'")
    # Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _extract_with_selectors(html: str, selectors: dict) -> dict:
    """
    Extract content from HTML using CSS selectors.
    Uses BeautifulSoup if available, otherwise returns empty results.
    """
    extracted = {}
    if not selectors:
        return extracted

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for key, selector in selectors.items():
            elements = soup.select(selector)
            if len(elements) == 1:
                extracted[key] = elements[0].get_text(separator=" ", strip=True)
            elif elements:
                extracted[key] = [el.get_text(separator=" ", strip=True) for el in elements]
            else:
                extracted[key] = None
    except ImportError:
        # No BeautifulSoup -- cannot do CSS selector extraction
        for key in selectors:
            extracted[key] = None

    return extracted


async def scrape_page(
    url: str,
    selectors: Optional[dict] = None,
    timeout: int = 30,
) -> dict:
    """
    Scrape a page and return structured result.

    Args:
        url: The URL to scrape.
        selectors: Optional dict of {name: css_selector} for structured extraction.
        timeout: Request timeout in seconds.

    Returns:
        {
            "url": str,
            "markdown": str,          # Full page content as markdown
            "html": str,              # Raw HTML (fallback path only)
            "extracted": dict,         # CSS selector results (if selectors provided)
            "source": str,            # "crawl4ai" or "requests"
            "success": bool,
            "error": str | None,
        }
    """
    domain = _get_domain(url)

    # --- Strategy 1: Crawl4AI (full browser, JS rendering) ---
    try:
        from crawl4ai import AsyncWebCrawler

        _random_delay(domain)
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(url=url)
            html = result.html if hasattr(result, "html") else ""
            markdown = result.markdown if hasattr(result, "markdown") else ""

            extracted = {}
            if selectors and html:
                extracted = _extract_with_selectors(html, selectors)

            return {
                "url": url,
                "markdown": markdown,
                "html": html,
                "extracted": extracted,
                "source": "crawl4ai",
                "success": True,
                "error": None,
            }
    except ImportError:
        pass  # Crawl4AI not installed, fall through to requests
    except Exception as e:
        # Crawl4AI failed (browser crash, timeout, etc.) -- fall through
        print(f"[browser_scraper] Crawl4AI failed for {url}: {e}")

    # --- Strategy 2: urllib fallback (no JS, but universal) ---
    try:
        _random_delay(domain)

        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "identity",
            "Connection": "keep-alive",
        }
        req = urllib.request.Request(url, headers=headers)

        # Create SSL context that doesn't verify (some sports sites have cert issues)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        resp = urllib.request.urlopen(req, timeout=timeout, context=ctx)
        raw_bytes = resp.read()

        # Try to detect encoding
        content_type = resp.headers.get("Content-Type", "")
        encoding = "utf-8"
        if "charset=" in content_type:
            encoding = content_type.split("charset=")[-1].split(";")[0].strip()

        try:
            html = raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            html = raw_bytes.decode("utf-8", errors="replace")

        markdown = _html_to_markdown(html)

        extracted = {}
        if selectors:
            extracted = _extract_with_selectors(html, selectors)

        return {
            "url": url,
            "markdown": markdown,
            "html": html,
            "extracted": extracted,
            "source": "requests",
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "url": url,
            "markdown": "",
            "html": "",
            "extracted": {},
            "source": "requests",
            "success": False,
            "error": str(e),
        }


async def scrape_pages(urls: list[str], selectors: Optional[dict] = None) -> list[dict]:
    """
    Scrape multiple pages sequentially (with rate limiting between requests).

    Args:
        urls: List of URLs to scrape.
        selectors: Optional CSS selectors applied to all pages.

    Returns:
        List of scrape results (same format as scrape_page).
    """
    results = []
    for url in urls:
        result = await scrape_page(url, selectors=selectors)
        results.append(result)
    return results


def scrape_sync(url: str, selectors: Optional[dict] = None) -> dict:
    """
    Synchronous wrapper around scrape_page for use in non-async code.

    Usage:
        result = scrape_sync("https://example.com")
        print(result["markdown"])
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an async context -- use a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, scrape_page(url, selectors=selectors))
            return future.result(timeout=60)
    else:
        return asyncio.run(scrape_page(url, selectors=selectors))


# ---------------------------------------------------------------------------
# CLI for testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python browser_scraper.py <url> [selector_name:css_selector ...]")
        print("Example: python browser_scraper.py https://example.com title:h1 links:a.nav-link")
        sys.exit(1)

    target_url = sys.argv[1]
    cli_selectors = {}
    for arg in sys.argv[2:]:
        if ":" in arg:
            name, sel = arg.split(":", 1)
            cli_selectors[name] = sel

    result = scrape_sync(target_url, selectors=cli_selectors if cli_selectors else None)

    if result["success"]:
        print(f"[{result['source']}] Scraped {result['url']}")
        print(f"Markdown length: {len(result['markdown'])} chars")
        if result["extracted"]:
            print(f"\nExtracted data:")
            print(json.dumps(result["extracted"], indent=2, default=str))
        print(f"\n--- Markdown (first 2000 chars) ---")
        print(result["markdown"][:2000])
    else:
        print(f"FAILED: {result['error']}")
        sys.exit(1)
