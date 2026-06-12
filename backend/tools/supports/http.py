"""
Central HTTP executor for all support module API calls.

What it does:
    Provides the single convergence point for every external HTTP request
    made by the data-acquisition tools.  Support modules describe WHAT to
    request (path, params, headers) and HOW to parse the response.  This
    module handles the actual HTTP call, status checking, response
    decoding, and event tracing.

Entities in it:
    - Endpoint: Frozen dataclass pairing a request builder with a response
      parser.  One instance per (source_id, source_type) combination.
    - fetch: Async function that executes a request spec and returns the
      parsed result.  All tracing happens here.

How used by other modules:
    - data_acquisition.py imports Endpoint and builds the DISPATCH table
      from the support modules' request/parse functions.
    - _execute_fetch() calls fetch() with a base_url, request spec, parse
      function, and the injected event callback.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

import httpx


@dataclass(frozen=True, slots=True)
class Endpoint:
    """Pairs a request builder with a response parser for one API endpoint.

    Attributes:
        request: Pure function (**kwargs -> dict) producing a request spec.
                 Receives the generic LLM parameters (symbol, interval,
                 limit, query, indicator, …) and returns a dict describing
                 the HTTP request.
        parse:   Pure function (raw_response_data -> normalized_result).
                 Receives the decoded response body (dict for JSON, str
                 for text/XML/CSV) and returns the final data structure.
    """

    request: Callable[..., dict[str, Any]]
    parse: Callable[[Any], Any]


async def fetch(
    base_url: str,
    spec: dict[str, Any],
    parse_fn: Callable[[Any], Any],
    emit_event: Callable[[dict[str, Any]], Any] | None = None,
) -> Any:
    """Execute an HTTP request described by *spec* and return parsed data.

    Args:
        base_url: API root (e.g. ``"https://www.okx.com/api/v5"``).
        spec:     Request specification dict produced by an Endpoint.request
                  function.  Keys:

                  ``path`` *(str, required)* —
                      URL path appended to *base_url*.

                  ``params`` *(dict, optional)* —
                      Query-string parameters for GET requests.

                  ``method`` *(str, optional, default ``"GET"``)* —
                      HTTP method.

                  ``body`` *(dict, optional)* —
                      JSON body for POST requests.

                  ``headers`` *(dict, optional)* —
                      Extra HTTP headers merged with the client defaults.

                  ``timeout`` *(float, optional, default ``15.0``)* —
                      Per-request timeout in seconds.

                  ``follow_redirects`` *(bool, optional, default ``False``)* —
                      Whether to follow HTTP redirects.

                  ``response_format`` *(str, optional, default ``"json"``)* —
                      ``"json"`` calls ``resp.json()``; ``"text"`` returns
                      ``resp.text`` (for CSV, XML, RSS, etc.).

                  ``base_url`` *(str, optional)* —
                      Overrides the *base_url* argument.  Used by
                      instance-based services (Mastodon, Lemmy) when the
                      user specifies a custom instance.

                  ``fallback_paths`` *(list[str], optional)* —
                      Alternative paths to try in order if the primary path
                      fails with an HTTP or connection error.

        parse_fn: Transforms the raw response into the caller's domain
                  objects.
        emit_event: Optional async-compatible callback for trace events.

    Returns:
        Whatever *parse_fn* returns.

    Raises:
        httpx.HTTPStatusError: On non-2xx HTTP status after all paths
            have been tried.
    """
    method = spec.get("method", "GET")
    params = spec.get("params")
    body = spec.get("body")
    headers = spec.get("headers", {})
    timeout = spec.get("timeout", 15.0)
    follow_redirects = spec.get("follow_redirects", False)
    response_format = spec.get("response_format", "json")

    effective_base_url = spec.get("base_url", base_url)
    paths = [spec["path"]] + spec.get("fallback_paths", [])

    primary_url = f"{effective_base_url}{paths[0]}"
    if emit_event is not None:
        event: dict[str, Any] = {
            "type": "tool_dispatch",
            "url": primary_url,
            "method": method,
        }
        if params:
            event["params"] = params
        if body:
            event["body"] = body
        cb_result = emit_event(event)
        if asyncio.iscoroutine(cb_result):
            await cb_result

    async with httpx.AsyncClient(
        timeout=timeout,
        headers=headers,
        follow_redirects=follow_redirects,
    ) as client:
        last_error: BaseException | None = None
        resp: httpx.Response | None = None

        for path in paths:
            url = f"{effective_base_url}{path}"
            try:
                if method == "POST":
                    resp = await client.post(url, json=body)
                else:
                    resp = await client.get(url, params=params)
                resp.raise_for_status()
                break
            except (httpx.HTTPStatusError, httpx.ConnectError) as exc:
                last_error = exc
                continue

        if resp is None or last_error is not None and resp.is_error:
            raise last_error  # type: ignore[misc]

        raw = resp.text if response_format == "text" else resp.json()

    return parse_fn(raw)
