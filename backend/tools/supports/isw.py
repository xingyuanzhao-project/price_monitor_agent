"""
ISW (Institute for the Study of War) request builders and response parsers.

What it does:
    Defines request specs and response parsers for the ISW RSS feed.
    Fetches latest geopolitical and military assessments from
    understandingwar.org by parsing their RSS/Atom feed.  No authentication
    required.  Uses multiple fallback feed URLs due to Cloudflare protection.

Entities in it:
    - BASE_URL: understandingwar.org root.
    - _normalize_limit: Coerces and clamps result-count limits to valid
      integer bounds.
    - _parse_rss: Helper to extract items from RSS XML.
    - _parse_atom: Helper to extract entries from Atom XML.
    - Request/parse pair for: latest.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw XML text to the parse function.

Source: https://understandingwar.org/
"""

from typing import Any
from xml.etree import ElementTree


BASE_URL = "https://understandingwar.org"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_limit(raw: Any, default: int = 10, lower: int = 1, upper: int = 50) -> int:
    """Coerce a limit value to an integer within bounds.

    Args:
        raw: Limit value from the LLM (int, float, or str).
        default: Fallback when raw is falsy or unparseable.
        lower: Minimum allowed value.
        upper: Maximum allowed value.

    Returns:
        Clamped integer limit.
    """
    try:
        value = int(str(raw).strip())
    except (ValueError, TypeError):
        return default
    return max(lower, min(value, upper))


# ---------------------------------------------------------------------------
# latest
# ---------------------------------------------------------------------------

def latest_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for the latest ISW assessments feed.

    Uses multiple fallback feed paths because understandingwar.org employs
    Cloudflare protection that may block some URLs.

    Args:
        **kwargs: Generic LLM params.  Uses ``limit`` (applied client-side
                  after parsing).

    Returns:
        Request spec dict for http.fetch().
    """
    limit = _normalize_limit(kwargs.get("limit", 10))
    return {
        "path": "/feed/",
        "fallback_paths": ["/rss.xml", "/feed.xml"],
        "response_format": "text",
        "limit": limit,
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        },
        "follow_redirects": True,
        "timeout": 20.0,
    }


def latest_parse(data: str) -> list[dict[str, Any]]:
    """Parse ISW RSS/Atom feed XML text into assessment dicts.

    Handles both RSS ``<item>`` and Atom ``<entry>`` formats depending on
    feed structure.

    Args:
        data: Raw XML text from the feed response.

    Returns:
        List of assessment dicts with title, link, date, description.

    Raises:
        RuntimeError: If the feed text is empty or unparseable.
    """
    if not data or not data.strip():
        raise RuntimeError(
            "ISW feed unavailable -- understandingwar.org uses Cloudflare "
            "protection that blocks programmatic HTTP access (403). "
            "Browser-based automation is required to access ISW content."
        )

    root = ElementTree.fromstring(data)

    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    items = root.findall(f".//{ns}item") or root.findall(".//item")
    if not items:
        entries = root.findall(f".//{ns}entry") or root.findall(".//entry")
        return _parse_atom(entries)

    return _parse_rss(items)


def _parse_rss(items: list) -> list[dict[str, Any]]:
    """Extract structured data from RSS <item> elements.

    Args:
        items: List of ElementTree item elements.

    Returns:
        List of assessment dicts.
    """
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
    """Extract structured data from Atom <entry> elements.

    Args:
        entries: List of ElementTree entry elements.

    Returns:
        List of assessment dicts.
    """
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
