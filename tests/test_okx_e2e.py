"""E2E: OKX -- all 4 source_types via FetchDataTool + artifact storage."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.data_acquisition import FetchExchangeDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchExchangeDataTool()
store = ArtifactStore(run_id="okx_e2e", base_dir="tests/output")


async def main():
    print("\n### OKX -- all source_types + artifact ###\n")

    # 1/4  ticker
    print("=" * 60)
    print("okx / ticker / BTC-USDT")
    print("=" * 60)
    r = await tool.execute(source_id="okx", source_type="ticker", symbol="BTC-USDT")
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2))
    assert r.content.get("instId") == "BTC-USDT", r.content
    print("PASS\n")

    # 2/4  ohlcv
    print("=" * 60)
    print("okx / ohlcv / BTC-USDT / 1H / limit=3")
    print("=" * 60)
    r = await tool.execute(source_id="okx", source_type="ohlcv", symbol="BTC-USDT", interval="1H", limit=3)
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2))
    assert len(r.content) == 3 and "o" in r.content[0], r.content
    print("PASS\n")

    # 3/4  orderbook
    print("=" * 60)
    print("okx / orderbook / BTC-USDT / limit=5")
    print("=" * 60)
    r = await tool.execute(source_id="okx", source_type="orderbook", symbol="BTC-USDT", limit=5)
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2, default=str))
    assert len(r.content["asks"]) > 0 and len(r.content["bids"]) > 0, r.content
    print("PASS\n")

    # 4/4  trades
    print("=" * 60)
    print("okx / trades / BTC-USDT / limit=5")
    print("=" * 60)
    r = await tool.execute(source_id="okx", source_type="trades", symbol="BTC-USDT", limit=5)
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2))
    assert len(r.content) == 5, f"expected 5 trades, got {len(r.content)}"
    print("PASS\n")

    # artifact -- large OHLCV (300 candles)
    print("=" * 60)
    print("ARTIFACT: okx / ohlcv / BTC-USDT / 1H / limit=300  ->  test/output/")
    print("=" * 60)
    r = await tool.execute(source_id="okx", source_type="ohlcv", symbol="BTC-USDT", interval="1H", limit=300)
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved {len(r.content)} candles -> {path}  ({r.size_bytes}B)")
    print(f"Read back {len(readback)} candles -- first: {json.dumps(readback[0])}")
    assert len(readback) == len(r.content), "readback mismatch"
    print("ARTIFACT PASS\n")

    print("ALL OKX TESTS PASSED (4 calls + 1 artifact)")


asyncio.run(main())
