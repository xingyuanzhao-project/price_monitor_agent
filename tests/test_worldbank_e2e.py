"""E2E: World Bank -- indicator + search via FetchDataTool + artifact."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.data_acquisition import FetchMacroDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchMacroDataTool()
store = ArtifactStore(run_id="worldbank_e2e", base_dir="tests/output")


async def main():
    print("\n### World Bank -- all source_types + artifact ###\n")

    # 1/2 indicator -- US GDP
    print("=" * 60)
    print("worldbank / indicator / NY.GDP.MKTP.CD (GDP) / US / 2015:2025")
    print("=" * 60)
    r = await tool.execute(
        source_id="worldbank", source_type="indicator",
        indicator="NY.GDP.MKTP.CD", country="US", date_range="2015:2025",
    )
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content[:3], indent=2, default=str))
    assert len(r.content) > 0, "no records"
    assert r.content[0].get("indicator") == "NY.GDP.MKTP.CD", r.content[0]
    print(f"Got {len(r.content)} observations")
    print("PASS\n")

    # 2/2 search -- find inflation indicators
    print("=" * 60)
    print("worldbank / search / 'GDP'")
    print("=" * 60)
    r = await tool.execute(source_id="worldbank", source_type="search", query="GDP")
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content[:5], indent=2, default=str))
    assert len(r.content) > 0, "no indicators found"
    print(f"Found {len(r.content)} matching indicators")
    print("PASS\n")

    # artifact -- multi-country GDP
    print("=" * 60)
    print("ARTIFACT: worldbank / indicator / GDP / all / 2000:2025 -> test/output/")
    print("=" * 60)
    r = await tool.execute(
        source_id="worldbank", source_type="indicator",
        indicator="NY.GDP.MKTP.CD", country="all", date_range="2000:2025", limit=500,
    )
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved {len(r.content)} records -> {path}  ({r.size_bytes}B)")
    print(f"Read back {len(readback)} records -- first: {json.dumps(readback[0], default=str)}")
    assert len(readback) == len(r.content), "readback mismatch"
    print("ARTIFACT PASS\n")

    print("ALL WORLD BANK TESTS PASSED (2 calls + 1 artifact)")


asyncio.run(main())
