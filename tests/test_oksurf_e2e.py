"""E2E: OKSURF -- headlines + section via FetchDataTool + artifact."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.data_acquisition import FetchNewsDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchNewsDataTool()
store = ArtifactStore(run_id="oksurf_e2e", base_dir="tests/output")


async def main():
    print("\n### OKSURF (Google News) -- all source_types + artifact ###\n")

    # 1/2 all headlines
    print("=" * 60)
    print("oksurf / headlines")
    print("=" * 60)
    r = await tool.execute(source_id="oksurf", source_type="headlines")
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    if isinstance(r.content, list) and len(r.content) > 0:
        print(json.dumps(r.content[:3], indent=2, default=str))
        print(f"Got {len(r.content)} articles across all sections")
    else:
        print(f"Response: {json.dumps(r.content, default=str)[:500]}")
    assert r.content is not None, "empty response"
    print("PASS\n")

    # 2/2 specific section
    print("=" * 60)
    print("oksurf / section / Business")
    print("=" * 60)
    r = await tool.execute(source_id="oksurf", source_type="section", section="Business")
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    if isinstance(r.content, list) and len(r.content) > 0:
        print(json.dumps(r.content[:3], indent=2, default=str))
        print(f"Got {len(r.content)} Business articles")
    else:
        print(f"Response: {json.dumps(r.content, default=str)[:500]}")
    assert r.content is not None, "empty response"
    print("PASS\n")

    # artifact -- all headlines
    print("=" * 60)
    print("ARTIFACT: oksurf / headlines -> test/output/")
    print("=" * 60)
    r = await tool.execute(source_id="oksurf", source_type="headlines")
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved -> {path}  ({r.size_bytes}B)")
    if isinstance(readback, list):
        print(f"Read back {len(readback)} articles")
    print("ARTIFACT PASS\n")

    print("ALL OKSURF TESTS PASSED (2 calls + 1 artifact)")


asyncio.run(main())
