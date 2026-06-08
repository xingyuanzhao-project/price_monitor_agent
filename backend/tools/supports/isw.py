"""
ISW (Institute for the Study of War) connector.

Fetches latest geopolitical and military assessments from understandingwar.org
by parsing their RSS feed. No authentication required.

Source: https://understandingwar.org/
"""

from typing import Any
from xml.etree import ElementTree

import httpx

FEED_URLS = [
    "https://understandingwar.org/feed/",
    "https://understandingwar.org/rss.xml",
    "https://understandingwar.org/feed.xml",
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}


async def fetch_latest(limit: int = 10) -> list[dict[str, Any]]:
    """Fetch latest ISW research assessments from their RSS feed.

    ISW (understandingwar.org) uses Cloudflare protection. If all feed
    URLs return 403, this function raises RuntimeError with a clear
    message rather than returning empty data.

    Args:
        limit: Max articles to return.

    Returns:
        List of assessment dicts with title, link, date, description.
    """
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=HEADERS) as client:
        for url in FEED_URLS:
            try:
                resp = await client.get(url)
                if resp.status_code == 200 and resp.text.strip():
                    break
            except httpx.HTTPError:
                continue
        else:
            raise RuntimeError(
                "ISW feed unavailable -- understandingwar.org uses Cloudflare "
                "protection that blocks programmatic HTTP access (403). "
                "Browser-based automation is required to access ISW content."
            )

        resp.raise_for_status()
        root = ElementTree.fromstring(resp.text)

        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        items = root.findall(f".//{ns}item") or root.findall(".//item")
        if not items:
            entries = root.findall(f".//{ns}entry") or root.findall(".//entry")
            return _parse_atom(entries[:limit])

        return _parse_rss(items[:limit])


def _parse_rss(items: list) -> list[dict[str, Any]]:
    results = []
    for item in items:
        title_el = item.find("title")
        link_el = item.find("link")
        date_el = item.find("pubDate")
        desc_el = item.find("description")
        results.append({
            "title": title_el.text.strip() if title_el is not None and title_el.text else "",
            "link": link_el.text.strip() if link_el is not None and link_el.text else "",
            "date": date_el.text.strip() if date_el is not None and date_el.text else "",
            "description": (desc_el.text.strip()[:500] if desc_el is not None and desc_el.text else ""),
        })
    return results


def _parse_atom(entries: list) -> list[dict[str, Any]]:
    results = []
    for entry in entries:
        ns = ""
        if entry.tag.startswith("{"):
            ns = entry.tag.split("}")[0] + "}"
        title_el = entry.find(f"{ns}title")
        link_el = entry.find(f"{ns}link")
        date_el = entry.find(f"{ns}updated") or entry.find(f"{ns}published")
        summary_el = entry.find(f"{ns}summary") or entry.find(f"{ns}content")
        link_href = ""
        if link_el is not None:
            link_href = link_el.get("href", "") or (link_el.text or "").strip()
        results.append({
            "title": title_el.text.strip() if title_el is not None and title_el.text else "",
            "link": link_href,
            "date": date_el.text.strip() if date_el is not None and date_el.text else "",
            "description": (summary_el.text.strip()[:500] if summary_el is not None and summary_el.text else ""),
        })
    return results
