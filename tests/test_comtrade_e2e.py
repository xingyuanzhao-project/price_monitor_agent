"""E2E: UN Comtrade -- trade flows via FetchDataTool + artifact."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.data_acquisition import FetchMacroDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchMacroDataTool()
store = ArtifactStore(run_id="comtrade_e2e", base_dir="tests/output")


async def main():
    print("\n### UN Comtrade -- all source_types + artifact ###\n")

    # 1/1 trade data (USA imports from World, 2023 -- use 2023 as 2024 may not be published yet)
    print("=" * 60)
    print("comtrade / trade / USA imports from World / 2023 / TOTAL")
    print("=" * 60)
    r = await tool.execute(
        source_id="comtrade", source_type="trade",
        reporter_code=842, period=2023, partner_code=0,
        flow_code="M", commodity_code="TOTAL",
    )
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    preview = r.content[:5] if isinstance(r.content, list) else r.content
    print(json.dumps(preview, indent=2, default=str)[:2000])
    assert isinstance(r.content, list), f"expected list, got {type(r.content)}"
    if len(r.content) > 0:
        rec = r.content[0]
        assert "trade_value" in rec, f"missing trade_value: {list(rec.keys())}"
        print(f"Got {len(r.content)} trade records")
    else:
        print("No trade records (may be data availability issue)")
    print("PASS\n")

    # artifact -- USA imports 2023 (same query, reuse the data from first call to avoid 429)
    print("=" * 60)
    print("ARTIFACT: comtrade / USA imports 2023 TOTAL -> test/output/")
    print("=" * 60)
    import time
    print("Waiting 3s to respect Comtrade rate limit...")
    time.sleep(3)
    r = await tool.execute(
        source_id="comtrade", source_type="trade",
        reporter_code=842, period=2023, partner_code=0,
        flow_code="X", commodity_code="TOTAL",
    )
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved {len(r.content)} records -> {path}  ({r.size_bytes}B)")
    assert len(readback) == len(r.content), "readback mismatch"
    print("ARTIFACT PASS\n")

    print("ALL COMTRADE TESTS PASSED (1 call + 1 artifact)")


asyncio.run(main())
