"""E2E: CoinGecko -- all 3 source_types via FetchDataTool + artifact storage."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.data_acquisition import FetchExchangeDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchExchangeDataTool()
store = ArtifactStore(run_id="coingecko_e2e", base_dir="tests/output")


async def main():
    print("\n### CoinGecko -- all source_types + artifact ###\n")

    # 1/3  ticker (simple price)
    print("=" * 60)
    print("coingecko / ticker / bitcoin")
    print("=" * 60)
    r = await tool.execute(source_id="coingecko", source_type="ticker", symbol="bitcoin")
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2))
    assert r.content["coin_id"] == "bitcoin" and r.content["price"] is not None, r.content
    print("PASS\n")

    # 2/3  ohlcv (market chart)
    print("=" * 60)
    print("coingecko / ohlcv / ethereum / 7 days")
    print("=" * 60)
    r = await tool.execute(source_id="coingecko", source_type="ohlcv", symbol="ethereum")
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(f"prices: {len(r.content['prices'])} pts  market_caps: {len(r.content['market_caps'])} pts")
    print(f"first price: {r.content['prices'][0]}")
    assert len(r.content["prices"]) > 50, f"too few points: {len(r.content['prices'])}"
    print("PASS\n")

    # 3/3  trending
    print("=" * 60)
    print("coingecko / trending")
    print("=" * 60)
    r = await tool.execute(source_id="coingecko", source_type="trending")
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content[:5], indent=2))
    assert len(r.content) > 0 and "symbol" in r.content[0], r.content
    print("PASS\n")

    # artifact -- 90-day bitcoin market chart
    print("=" * 60)
    print("ARTIFACT: coingecko / ohlcv / bitcoin / 90 days  ->  test/output/")
    print("=" * 60)
    r = await tool.execute(source_id="coingecko", source_type="ohlcv", symbol="bitcoin", days=90)
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved {len(r.content['prices'])} price points -> {path}  ({r.size_bytes}B)")
    print(f"Read back {len(readback['prices'])} price points -- first: {readback['prices'][0]}")
    assert len(readback["prices"]) == len(r.content["prices"]), "readback mismatch"
    print("ARTIFACT PASS\n")

    print("ALL COINGECKO TESTS PASSED (3 calls + 1 artifact)")


asyncio.run(main())
