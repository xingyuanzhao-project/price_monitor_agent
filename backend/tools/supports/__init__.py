"""
Data source support modules for exchange, macro, news, and social data.

What it does:
    Each support module defines pure request builders and response parsers
    for a single external API.  Modules contain NO I/O — no httpx, no
    async, no network calls.  All HTTP execution is handled by http.py
    via the Endpoint abstraction.

Entities in it:
    - One module per data source (e.g. okx.py, fred.py, gdelt.py).
    - http.py — central HTTP executor and Endpoint dataclass.
    - Each module exports: BASE_URL, *_request/**_parse function pairs,
      and internal _normalize_* helpers.

How used by other modules:
    - data_acquisition.py imports each module and builds the DISPATCH table
      from (source_type → Endpoint(request_fn, parse_fn)) mappings.
    - data_acquisition._execute_fetch() calls http.fetch(base_url, spec,
      parse_fn) which makes the HTTP call and traces it.

==========================================================================
HOW TO ADD A NEW SUPPORT MODULE
==========================================================================

Follow these steps when adding a new data source.

1. CREATE THE MODULE FILE
   ~~~~~~~~~~~~~~~~~~~~~~~~
   Create ``supports/<source_id>.py``.  The source_id should be a short,
   lowercase, no-hyphen identifier (e.g. ``ecb``, ``okx``, ``worldbank``).

   Required structure::

       \"\"\"
       <Source Name> (<full name or description>) request builders and
       response parsers.

       What it does:
           <What API it covers, whether auth is required.>

       Entities in it:
           - BASE_URL: <description>.
           - Normalization: <what _normalize_* functions do>.
           - Request/parse pairs for: <list of endpoints>.

       How used by other modules:
           - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
           - http.fetch() calls the request function, makes the HTTP call,
             then passes the raw response to the parse function.

       API docs: <URL>
       \"\"\"

       from typing import Any

       BASE_URL = "<api root url>"

2. DEFINE NORMALIZATION FUNCTIONS
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   For every input parameter where the LLM may send variant formats, add
   a ``_normalize_*`` function.  Common categories:

   - **Symbol/ticker format**: slash vs dash vs concatenated, case
     (e.g. ``BTC/USDT`` → ``BTC-USDT`` for OKX, ``BTCUSDT`` for Binance).
   - **Currency codes**: uppercase 3-letter ISO 4217 (``usd`` → ``USD``).
   - **Country codes**: uppercase 2-letter ISO 3166-1 (``us`` → ``US``)
     or lowercase for URL paths.
   - **Date formats**: ``YYYY-MM-DD`` canonical form; strip time parts,
     handle ``YYYY/MM/DD`` or ``MM-DD-YYYY`` variants.
   - **Interval/frequency**: map common aliases
     (``1h`` → ``1H``, ``daily`` → ``1d``).
   - **Slug/identifier**: lowercase, strip whitespace, replace spaces
     with hyphens (``"Hacker News"`` → ``"hacker-news"``).

   Example::

       def _normalize_symbol(raw: str) -> str:
           s = raw.strip().upper()
           s = s.replace("/", "-").replace("_", "-")
           return s

3. DEFINE REQUEST/PARSE FUNCTION PAIRS
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   For each API endpoint, create two pure functions:

   **Request function** ``<source_type>_request(**kwargs: Any) -> dict``

   - Accepts ``**kwargs`` (the generic LLM parameters: ``symbol``,
     ``interval``, ``limit``, ``query``, ``indicator``, etc.).
   - Extracts only the parameters it needs via ``kwargs.get("name", default)``.
   - Normalizes inputs using the ``_normalize_*`` helpers.
   - Returns a request spec dict for ``http.fetch()``.

   Required key::

       "path": str   # URL path appended to BASE_URL

   Optional keys::

       "params": dict          # query-string params (GET)
       "method": str           # "GET" (default) or "POST"
       "body": dict            # JSON body (POST)
       "headers": dict         # extra HTTP headers
       "timeout": float        # seconds (default 15.0)
       "follow_redirects": bool  # (default False)
       "response_format": str  # "json" (default) or "text" (CSV/XML/RSS)
       "fallback_paths": list  # alternative paths to try on failure

   Example::

       def ticker_request(**kwargs: Any) -> dict[str, Any]:
           inst_id = _normalize_symbol(kwargs.get("symbol", ""))
           return {"path": "/market/ticker", "params": {"instId": inst_id}}

   **Parse function** ``<source_type>_parse(data: Any) -> Any``

   - Receives the decoded response body (dict/list for JSON, str for text).
   - Extracts and normalizes the data into a clean structure.
   - Raises ``RuntimeError`` with a clear message on API-level errors.
   - Returns the final data (list of dicts, dict, etc.).

   Example::

       def ticker_parse(data: dict) -> dict[str, Any]:
           if data.get("code") != "0":
               raise RuntimeError(f"API error: {data.get('msg', 'unknown')}")
           return data["data"][0] if data.get("data") else {}

4. REGISTER IN data_acquisition.py
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   In ``data_acquisition.py``, do three things:

   a. Add the import::

       from backend.tools.supports import new_source

   b. Add the base URL to ``SOURCE_BASE_URLS``::

       "new_source": new_source.BASE_URL,

   c. Add endpoints to ``DISPATCH``::

       "new_source": {
           "endpoint_name": Endpoint(
               new_source.endpoint_name_request,
               new_source.endpoint_name_parse,
           ),
       },

   d. Add the source_id to the appropriate list in ``SOURCE_CATEGORIES``
      (``"exchange"``, ``"macro"``, ``"news"``, or ``"social"``).

   e. Add aliases to ``_SOURCE_ALIASES`` for natural-language parsing.

5. STANDARDS TO RESPECT
   ~~~~~~~~~~~~~~~~~~~~~~
   - Modules must be pure: no ``import httpx``, no ``async``, no I/O.
   - All HTTP execution goes through ``http.fetch()`` in ``http.py``.
   - Functions accept ``**kwargs`` and pick what they need — do not
     require the caller to rename parameters.
   - Normalization lives in the request function, not in the caller.
   - Parse functions raise on API errors — no silent fallbacks.
   - Follow the project's coding rules (``docs/coding rules.md``):
     full docstrings, descriptive names, no abbreviations.
   - For API-key-requiring sources, accept the key via
     ``kwargs.get("api_key", "")`` and include it in the request spec's
     ``params`` or ``headers`` as the API expects.
"""
