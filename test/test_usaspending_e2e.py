"""E2E: USA Spending -- by_agency + over_time via FetchDataTool + artifact."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.data_acquisition import FetchDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchDataTool()
store = ArtifactStore(run_id="usaspending_e2e", base_dir="test/output")


async def main():
    print("\n### USA Spending -- all source_types + artifact ###\n")

    # 1/2 by agency (FY 2025)
    print("=" * 60)
    print("usaspending / by_agency / FY2025")
    print("=" * 60)
    r = await tool.execute(
        source_id="usaspending", source_type="by_agency",
        fiscal_year=2025, limit=10,
    )
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    preview = r.content[:3] if isinstance(r.content, list) else r.content
    print(json.dumps(preview, indent=2, default=str)[:2000])
    assert isinstance(r.content, list), f"expected list, got {type(r.content)}"
    print(f"Got {len(r.content)} agencies")
    print("PASS\n")

    # 2/2 over time
    print("=" * 60)
    print("usaspending / over_time / fiscal_year / 2020-2025")
    print("=" * 60)
    r = await tool.execute(
        source_id="usaspending", source_type="over_time",
        group="fiscal_year",
        time_period_start="2020-10-01", time_period_end="2025-09-30",
    )
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    preview = r.content[:5] if isinstance(r.content, list) else r.content
    print(json.dumps(preview, indent=2, default=str)[:2000])
    assert isinstance(r.content, list), f"expected list"
    print(f"Got {len(r.content)} periods")
    print("PASS\n")

    # artifact
    print("=" * 60)
    print("ARTIFACT: usaspending / by_agency / FY2025 top 20 -> test/output/")
    print("=" * 60)
    r = await tool.execute(
        source_id="usaspending", source_type="by_agency",
        fiscal_year=2025, limit=20,
    )
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved {len(r.content)} agencies -> {path}  ({r.size_bytes}B)")
    assert len(readback) == len(r.content), "readback mismatch"
    print("ARTIFACT PASS\n")

    print("ALL USASPENDING TESTS PASSED (2 calls + 1 artifact)")


asyncio.run(main())
