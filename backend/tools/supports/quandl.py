"""
Nasdaq Data Link (formerly Quandl) request builders and response parsers.

What it does:
    Defines request specs and response parsers for the Nasdaq Data Link
    (Quandl) REST API v3.  Covers time-series dataset retrieval and dataset
    metadata lookup.  Requires an API key.

Entities in it:
    - BASE_URL: Nasdaq Data Link API v3 root.
    - _normalize_database_code: Uppercases database identifiers.
    - _normalize_dataset_code: Uppercases dataset identifiers.
    - Request/parse pairs for: dataset, metadata.

How used by other modules:
    - data_acquisition.py registers these as Endpoint pairs in DISPATCH.
    - http.fetch() calls the request function, makes the HTTP call, then
      passes the raw JSON to the parse function.

API docs: https://docs.data.nasdaq.com/
"""

from typing import Any


BASE_URL = "https://data.nasdaq.com/api/v3"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_database_code(raw: str) -> str:
    """Uppercase and strip a Nasdaq Data Link database code.

    Args:
        raw: Database code from the LLM (e.g. "fred", " WIKI ").

    Returns:
        Uppercased, stripped database code.
    """
    return raw.strip().upper()


def _normalize_dataset_code(raw: str) -> str:
    """Uppercase and strip a Nasdaq Data Link dataset code.

    Args:
        raw: Dataset code from the LLM (e.g. "gdp", " AAPL ").

    Returns:
        Uppercased, stripped dataset code.
    """
    return raw.strip().upper()


# ---------------------------------------------------------------------------
# dataset (time-series data)
# ---------------------------------------------------------------------------

def dataset_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for a Nasdaq Data Link time-series dataset.

    Args:
        **kwargs: Generic LLM params.  Uses ``database_code``,
                  ``dataset_code``, ``api_key``, ``limit``, ``order``
                  (asc/desc).

    Returns:
        Request spec dict for http.fetch().
    """
    database_code = _normalize_database_code(kwargs.get("database_code", ""))
    dataset_code = _normalize_dataset_code(kwargs.get("dataset_code", ""))
    api_key = kwargs.get("api_key", "")
    limit = int(kwargs.get("limit", 100))
    order = kwargs.get("order", "desc")
    return {
        "path": f"/datasets/{database_code}/{dataset_code}/data.json",
        "params": {
            "api_key": api_key,
            "limit": limit,
            "order": order,
        },
    }


def dataset_parse(data: dict) -> list[dict[str, Any]]:
    """Parse Nasdaq Data Link dataset JSON response.

    Zips column names with each row to produce a list of named dicts.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        List of row dicts with column names from the dataset.
    """
    dataset_data = data.get("dataset_data", {})
    columns = dataset_data.get("column_names", [])
    rows = []
    for row in dataset_data.get("data", []):
        rows.append(dict(zip(columns, row)))
    return rows


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------

def metadata_request(**kwargs: Any) -> dict[str, Any]:
    """Build request spec for Nasdaq Data Link dataset metadata.

    Args:
        **kwargs: Generic LLM params.  Uses ``database_code``,
                  ``dataset_code``, ``api_key``.

    Returns:
        Request spec dict for http.fetch().
    """
    database_code = _normalize_database_code(kwargs.get("database_code", ""))
    dataset_code = _normalize_dataset_code(kwargs.get("dataset_code", ""))
    api_key = kwargs.get("api_key", "")
    return {
        "path": f"/datasets/{database_code}/{dataset_code}/metadata.json",
        "params": {"api_key": api_key},
    }


def metadata_parse(data: dict) -> dict[str, Any]:
    """Parse Nasdaq Data Link dataset metadata JSON response.

    Args:
        data: Raw JSON dict from the API.

    Returns:
        Dict with name, description, frequency, column_names, and date range.
    """
    dataset = data.get("dataset", {})
    return {
        "name": dataset.get("name"),
        "description": dataset.get("description", "")[:500],
        "frequency": dataset.get("frequency"),
        "column_names": dataset.get("column_names", []),
        "newest_available_date": dataset.get("newest_available_date"),
        "oldest_available_date": dataset.get("oldest_available_date"),
    }
