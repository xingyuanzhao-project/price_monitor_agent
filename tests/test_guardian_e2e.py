"""E2E: Guardian -- all 2 source_types via FetchDataTool + artifact storage.

Uses the free 'test' key -- shared daily quota may be exhausted.
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
from backend.tools.data_acquisition import FetchNewsDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchNewsDataTool()
store = ArtifactStore(run_id="guardian_e2e", base_dir="tests/output")


async def main():
    print("\n### Guardian -- all source_types + artifact ###\n")

    # quota check
    async with httpx.AsyncClient() as c:
        probe = await c.get("https://content.guardianapis.com/search",
                            params={"q": "ping", "api-key": "test", "page-size": 1})
    if probe.status_code == 429:
        remaining = probe.headers.get("x-ratelimit-remaining-day", "?")
        reset = probe.headers.get("retry-after", "?")
        print(f"'test' key quota exhausted (remaining={remaining}, resets in {reset}s)")
        print("SKIP -- re-run after daily reset\n")
        return

    # 1/2  search
    print("=" * 60)
    print("guardian / search / query=inflation / limit=5")
    print("=" * 60)
    r = await tool.execute(source_id="guardian", source_type="search", query="inflation", limit=5)
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2))
    assert len(r.content) > 0 and "title" in r.content[0], r.content
    print("PASS\n")

    await asyncio.sleep(1)

    # also search a different topic
    print("=" * 60)
    print("guardian / search / query=bitcoin / limit=3")
    print("=" * 60)
    r = await tool.execute(source_id="guardian", source_type="search", query="bitcoin", limit=3)
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2))
    assert len(r.content) > 0, "no bitcoin articles"
    print("PASS\n")

    await asyncio.sleep(1)

    # 2/2  headlines
    print("=" * 60)
    print("guardian / headlines / section=business")
    print("=" * 60)
    r = await tool.execute(source_id="guardian", source_type="headlines")
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content[:3], indent=2))
    assert len(r.content) > 0 and "title" in r.content[0], r.content
    print("PASS\n")

    await asyncio.sleep(1)

    # also test technology section
    print("=" * 60)
    print("guardian / headlines / section=technology")
    print("=" * 60)
    r = await tool.execute(source_id="guardian", source_type="headlines", section="technology")
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content[:3], indent=2))
    assert len(r.content) > 0, "no tech headlines"
    print("PASS\n")

    # artifact -- 50 articles about economy
    print("=" * 60)
    print("ARTIFACT: guardian / search / query=economy / limit=50  ->  test/output/")
    print("=" * 60)
    r = await tool.execute(source_id="guardian", source_type="search", query="economy", limit=50)
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved {len(r.content)} articles -> {path}  ({r.size_bytes}B)")
    print(f"Read back {len(readback)} articles -- #1: {readback[0]['title'][:60]}")
    assert len(readback) == len(r.content), "readback mismatch"
    print("ARTIFACT PASS\n")

    print("ALL GUARDIAN TESTS PASSED (4 calls + 1 artifact)")


asyncio.run(main())
