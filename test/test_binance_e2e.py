"""E2E: Binance -- all 4 source_types via FetchDataTool + artifact storage."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import backend.tools.supports.binance as binance_mod
from backend.tools.data_acquisition import FetchDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchDataTool()
store = ArtifactStore(run_id="binance_e2e", base_dir="test/output")


async def main():
    print("\n### Binance -- all source_types + artifact ###\n")

    # probe geo-block
    try:
        await tool.execute(source_id="binance", source_type="ticker", symbol="BTCUSDT")
    except Exception as e:
        if "451" in str(e):
            print("Primary domain geo-blocked (451). Switching to data-api.binance.vision\n")
            binance_mod.BASE_URL = "https://data-api.binance.vision/api/v3"

    # 1/4  ticker
    print("=" * 60)
    print("binance / ticker / BTCUSDT")
    print("=" * 60)
    r = await tool.execute(source_id="binance", source_type="ticker", symbol="BTCUSDT")
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2))
    assert r.content["symbol"] == "BTCUSDT", r.content
    print("PASS\n")

    # 2/4  ohlcv
    print("=" * 60)
    print("binance / ohlcv / BTCUSDT / 1h / limit=3")
    print("=" * 60)
    r = await tool.execute(source_id="binance", source_type="ohlcv", symbol="BTCUSDT", interval="1h", limit=3)
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2))
    assert len(r.content) == 3 and "o" in r.content[0], r.content
    print("PASS\n")

    # 3/4  orderbook
    print("=" * 60)
    print("binance / orderbook / BTCUSDT / limit=5")
    print("=" * 60)
    r = await tool.execute(source_id="binance", source_type="orderbook", symbol="BTCUSDT", limit=5)
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2))
    assert len(r.content["asks"]) == 5, r.content
    print("PASS\n")

    # 4/4  trades
    print("=" * 60)
    print("binance / trades / BTCUSDT / limit=5")
    print("=" * 60)
    r = await tool.execute(source_id="binance", source_type="trades", symbol="BTCUSDT", limit=5)
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content[:3], indent=2))
    assert len(r.content) == 5 and "price" in r.content[0], r.content
    print("PASS\n")

    # artifact -- large OHLCV (1000 candles, 1m)
    print("=" * 60)
    print("ARTIFACT: binance / ohlcv / ETHUSDT / 1m / limit=1000  ->  test/output/")
    print("=" * 60)
    r = await tool.execute(source_id="binance", source_type="ohlcv", symbol="ETHUSDT", interval="1m", limit=1000)
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved {len(r.content)} candles -> {path}  ({r.size_bytes}B)")
    print(f"Read back {len(readback)} candles -- first: {json.dumps(readback[0])}")
    assert len(readback) == len(r.content), "readback mismatch"
    print("ARTIFACT PASS\n")

    print("ALL BINANCE TESTS PASSED (4 calls + 1 artifact)")


asyncio.run(main())
