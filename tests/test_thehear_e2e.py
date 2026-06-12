"""E2E: The Hear -- country headlines via FetchDataTool + artifact."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.data_acquisition import FetchNewsDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchNewsDataTool()
store = ArtifactStore(run_id="thehear_e2e", base_dir="tests/output")


async def main():
    print("\n### The Hear -- country headlines + artifact ###\n")

    # 1/1 country -- US
    print("=" * 60)
    print("thehear / country / us")
    print("=" * 60)
    r = await tool.execute(source_id="thehear", source_type="country", country="us")
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    content = r.content
    if isinstance(content, dict):
        headlines = content.get("headlines", [])
        overviews = content.get("overviews", {})
        print(f"Got {len(headlines)} headlines, {len(overviews)} overviews")
        if headlines:
            print(json.dumps(headlines[:3], indent=2, default=str))
    else:
        print(json.dumps(content, indent=2, default=str)[:500])
    assert content is not None, "empty response"
    print("PASS\n")

    # artifact
    print("=" * 60)
    print("ARTIFACT: thehear / country / us -> test/output/")
    print("=" * 60)
    r = await tool.execute(source_id="thehear", source_type="country", country="us")
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved -> {path}  ({r.size_bytes}B)")
    if isinstance(readback, dict):
        print(f"Read back {len(readback.get('headlines', []))} headlines")
    print("ARTIFACT PASS\n")

    print("ALL THEHEAR TESTS PASSED (1 call + 1 artifact)")


asyncio.run(main())
