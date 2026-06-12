"""E2E: Yahoo Finance -- quote + ohlcv via FetchDataTool + artifact storage."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.data_acquisition import FetchExchangeDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchExchangeDataTool()
store = ArtifactStore(run_id="yahoo_e2e", base_dir="tests/output")


async def main():
    print("\n### Yahoo Finance -- all source_types + artifact ###\n")

    # 1/2 quote (stock)
    print("=" * 60)
    print("yahoo / quote / AAPL")
    print("=" * 60)
    r = await tool.execute(source_id="yahoo", source_type="quote", symbol="AAPL")
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2))
    assert r.content.get("symbol") == "AAPL", r.content
    assert r.content.get("price") is not None, "price missing"
    print("PASS\n")

    # 2/2 ohlcv (index -- S&P 500)
    print("=" * 60)
    print("yahoo / ohlcv / ^GSPC / 1d / 5d range")
    print("=" * 60)
    r = await tool.execute(source_id="yahoo", source_type="ohlcv", symbol="^GSPC", interval="1d", range_period="5d")
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content[:3], indent=2, default=str))
    assert len(r.content) > 0, "no candles"
    assert "o" in r.content[0], "missing ohlcv fields"
    print(f"Got {len(r.content)} candles")
    print("PASS\n")

    # artifact -- 1 month AAPL daily
    print("=" * 60)
    print("ARTIFACT: yahoo / ohlcv / AAPL / 1d / 1mo -> test/output/")
    print("=" * 60)
    r = await tool.execute(source_id="yahoo", source_type="ohlcv", symbol="AAPL", interval="1d", range_period="1mo")
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved {len(r.content)} candles -> {path}  ({r.size_bytes}B)")
    print(f"Read back {len(readback)} candles -- first: {json.dumps(readback[0], default=str)}")
    assert len(readback) == len(r.content), "readback mismatch"
    print("ARTIFACT PASS\n")

    print("ALL YAHOO TESTS PASSED (2 calls + 1 artifact)")


asyncio.run(main())
