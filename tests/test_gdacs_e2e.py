"""E2E: GDACS -- disaster events via FetchDataTool + artifact."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.data_acquisition import FetchNewsDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchNewsDataTool()
store = ArtifactStore(run_id="gdacs_e2e", base_dir="tests/output")


async def main():
    print("\n### GDACS -- all source_types + artifact ###\n")

    # 1/1 current disaster events
    print("=" * 60)
    print("gdacs / events / top 25")
    print("=" * 60)
    r = await tool.execute(source_id="gdacs", source_type="events", limit=25)
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    preview = r.content[:3] if isinstance(r.content, list) else r.content
    print(json.dumps(preview, indent=2, default=str).encode("ascii", "replace").decode())
    assert isinstance(r.content, list), f"expected list, got {type(r.content)}"
    if len(r.content) > 0:
        ev = r.content[0]
        assert "title" in ev or "event_type" in ev, f"bad shape: {list(ev.keys())[:5]}"
        print(f"Got {len(r.content)} disaster alerts")
    else:
        print("No active GDACS alerts (unlikely but possible)")
    print("PASS\n")

    # artifact
    print("=" * 60)
    print("ARTIFACT: gdacs / events / 50 -> test/output/")
    print("=" * 60)
    r = await tool.execute(source_id="gdacs", source_type="events", limit=50)
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved {len(r.content)} alerts -> {path}  ({r.size_bytes}B)")
    assert len(readback) == len(r.content), "readback mismatch"
    print("ARTIFACT PASS\n")

    print("ALL GDACS TESTS PASSED (1 call + 1 artifact)")


asyncio.run(main())
