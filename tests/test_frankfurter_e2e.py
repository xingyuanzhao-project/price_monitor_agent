"""E2E: Frankfurter -- latest + timeseries via FetchDataTool + artifact."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.data_acquisition import FetchExchangeDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchExchangeDataTool()
store = ArtifactStore(run_id="frankfurter_e2e", base_dir="tests/output")


async def main():
    print("\n### Frankfurter -- all source_types + artifact ###\n")

    # 1/2 latest rates
    print("=" * 60)
    print("frankfurter / latest / base=USD")
    print("=" * 60)
    r = await tool.execute(source_id="frankfurter", source_type="latest", base="USD", symbols="EUR,GBP,JPY")
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2))
    assert "rates" in r.content, r.content
    assert "EUR" in r.content["rates"], "EUR rate missing"
    print("PASS\n")

    # 2/2 timeseries
    print("=" * 60)
    print("frankfurter / timeseries / USD->EUR / last 30 days")
    print("=" * 60)
    r = await tool.execute(source_id="frankfurter", source_type="timeseries", base="USD", symbols="EUR")
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    rates = r.content.get("rates", {})
    print(f"Got {len(rates)} dates")
    first_date = sorted(rates.keys())[0] if rates else "N/A"
    print(f"First date: {first_date} -> {rates.get(first_date, {})}")
    assert len(rates) > 5, "too few dates"
    print("PASS\n")

    # artifact -- full timeseries
    print("=" * 60)
    print("ARTIFACT: frankfurter / timeseries / USD->EUR,GBP,JPY -> test/output/")
    print("=" * 60)
    r = await tool.execute(source_id="frankfurter", source_type="timeseries", base="USD", symbols="EUR,GBP,JPY")
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved -> {path}  ({r.size_bytes}B)")
    print(f"Read back {len(readback.get('rates', {}))} dates")
    assert len(readback.get("rates", {})) > 0, "readback empty"
    print("ARTIFACT PASS\n")

    print("ALL FRANKFURTER TESTS PASSED (2 calls + 1 artifact)")


asyncio.run(main())
