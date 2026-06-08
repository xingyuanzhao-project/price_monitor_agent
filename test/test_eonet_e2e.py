"""E2E: NASA EONET -- events + categories via FetchDataTool + artifact."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.data_acquisition import FetchDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchDataTool()
store = ArtifactStore(run_id="eonet_e2e", base_dir="test/output")


async def main():
    print("\n### NASA EONET -- all source_types + artifact ###\n")

    # 1/2 events (open, last 30 days)
    print("=" * 60)
    print("eonet / events / open / 30 days")
    print("=" * 60)
    r = await tool.execute(
        source_id="eonet", source_type="events",
        days=30, status="open", limit=20,
    )
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    preview = r.content[:3] if isinstance(r.content, list) else r.content
    print(json.dumps(preview, indent=2, default=str).encode("ascii", "replace").decode())
    assert isinstance(r.content, list), f"expected list, got {type(r.content)}"
    if len(r.content) > 0:
        ev = r.content[0]
        assert "title" in ev, f"missing title: {list(ev.keys())}"
        assert "categories" in ev, f"missing categories: {list(ev.keys())}"
        print(f"Got {len(r.content)} natural events (open 30d)")
    else:
        print("No open EONET events in last 30 days (rare)")
    print("PASS\n")

    # 2/2 categories
    print("=" * 60)
    print("eonet / categories")
    print("=" * 60)
    r = await tool.execute(source_id="eonet", source_type="categories")
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content[:5] if isinstance(r.content, list) else r.content, indent=2, default=str))
    assert isinstance(r.content, list), "expected list of categories"
    assert len(r.content) > 0, "no categories returned"
    print(f"Got {len(r.content)} event categories")
    print("PASS\n")

    # artifact -- 60 day events
    print("=" * 60)
    print("ARTIFACT: eonet / events / 60d open -> test/output/")
    print("=" * 60)
    r = await tool.execute(
        source_id="eonet", source_type="events",
        days=60, status="open", limit=100,
    )
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved {len(r.content)} events -> {path}  ({r.size_bytes}B)")
    assert len(readback) == len(r.content), "readback mismatch"
    print("ARTIFACT PASS\n")

    print("ALL EONET TESTS PASSED (2 calls + 1 artifact)")


asyncio.run(main())
