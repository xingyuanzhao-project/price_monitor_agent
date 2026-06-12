"""E2E: USGS Earthquake -- earthquakes via FetchDataTool + artifact."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.data_acquisition import FetchNewsDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchNewsDataTool()
store = ArtifactStore(run_id="usgs_e2e", base_dir="tests/output")


async def main():
    print("\n### USGS Earthquake -- all source_types + artifact ###\n")

    # 1/1 earthquakes M4.5+ past week
    print("=" * 60)
    print("usgs / earthquakes / M4.5 / week")
    print("=" * 60)
    r = await tool.execute(
        source_id="usgs", source_type="earthquakes",
        min_magnitude="4.5", period="week",
    )
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    preview = r.content[:5] if isinstance(r.content, list) else r.content
    print(json.dumps(preview, indent=2, default=str))
    assert isinstance(r.content, list), "expected list"
    if len(r.content) > 0:
        eq = r.content[0]
        assert "magnitude" in eq, f"missing mag: {list(eq.keys())}"
        assert "place" in eq, f"missing place"
        print(f"Got {len(r.content)} earthquakes M4.5+ this week")
    else:
        print("No M4.5+ earthquakes this week (rare but possible)")
    print("PASS\n")

    # artifact -- M4.5 past month
    print("=" * 60)
    print("ARTIFACT: usgs / earthquakes / M4.5 / month -> test/output/")
    print("=" * 60)
    r = await tool.execute(
        source_id="usgs", source_type="earthquakes",
        min_magnitude="4.5", period="month",
    )
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved {len(r.content)} earthquakes -> {path}  ({r.size_bytes}B)")
    assert len(readback) == len(r.content), "readback mismatch"
    print("ARTIFACT PASS\n")

    print("ALL USGS TESTS PASSED (1 call + 1 artifact)")


asyncio.run(main())
