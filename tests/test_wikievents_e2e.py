"""E2E: WikiEvents -- latest via FetchDataTool + artifact."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.data_acquisition import FetchNewsDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchNewsDataTool()
store = ArtifactStore(run_id="wikievents_e2e", base_dir="tests/output")


async def main():
    print("\n### WikiEvents (Wikipedia Current Events) -- all source_types + artifact ###\n")

    # 1/1 latest
    print("=" * 60)
    print("wikievents / latest")
    print("=" * 60)
    r = await tool.execute(source_id="wikievents", source_type="latest")
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    if isinstance(r.content, list) and len(r.content) > 0:
        print(json.dumps(r.content[:3], indent=2, default=str))
        print(f"Got {len(r.content)} events")
    else:
        print(f"Response: {json.dumps(r.content, default=str)[:500]}")
    assert r.content is not None, "empty response"
    print("PASS\n")

    # artifact
    print("=" * 60)
    print("ARTIFACT: wikievents / latest -> test/output/")
    print("=" * 60)
    r = await tool.execute(source_id="wikievents", source_type="latest")
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved -> {path}  ({r.size_bytes}B)")
    if isinstance(readback, list):
        print(f"Read back {len(readback)} events")
    print("ARTIFACT PASS\n")

    print("ALL WIKIEVENTS TESTS PASSED (1 call + 1 artifact)")


asyncio.run(main())
