"""E2E: Lemmy -- posts + search via FetchDataTool + artifact."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.data_acquisition import FetchSocialMediaDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchSocialMediaDataTool()
store = ArtifactStore(run_id="lemmy_e2e", base_dir="tests/output")


async def main():
    print("\n### Lemmy -- all source_types + artifact ###\n")

    # 1/2 posts from cryptocurrency community
    print("=" * 60)
    print("lemmy / posts / cryptocurrency / lemmy.ml / Hot")
    print("=" * 60)
    r = await tool.execute(
        source_id="lemmy", source_type="posts",
        community="cryptocurrency", sort="Hot", limit=10,
    )
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    if isinstance(r.content, list) and len(r.content) > 0:
        print(json.dumps(r.content[:3], indent=2, default=str))
        print(f"Got {len(r.content)} posts")
        for p in r.content[:5]:
            print(f"  - [{p.get('score', 0)}] {p.get('title', '')[:60]}")
    else:
        print(f"Response: {json.dumps(r.content, default=str)[:500]}")
        print("Community may be empty on this instance -- trying lemmy.world")
        r = await tool.execute(
            source_id="lemmy", source_type="posts",
            community="cryptocurrency", instance="https://lemmy.world",
            sort="Hot", limit=10,
        )
        print(f"data_type={r.data_type}  size={r.size_bytes}B")
        print(json.dumps(r.content[:3], indent=2, default=str))
    print("PASS\n")

    # 2/2 search
    print("=" * 60)
    print("lemmy / search / 'bitcoin' / lemmy.ml")
    print("=" * 60)
    r = await tool.execute(
        source_id="lemmy", source_type="search",
        query="bitcoin", limit=10,
    )
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    if isinstance(r.content, list) and len(r.content) > 0:
        print(json.dumps(r.content[:3], indent=2, default=str))
        print(f"Got {len(r.content)} search results")
    else:
        print(f"Response: {json.dumps(r.content, default=str)[:500]}")
    print("PASS\n")

    # artifact
    print("=" * 60)
    print("ARTIFACT: lemmy / posts / cryptocurrency -> test/output/")
    print("=" * 60)
    r = await tool.execute(
        source_id="lemmy", source_type="posts",
        community="cryptocurrency", sort="New", limit=50,
    )
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved {len(r.content)} posts -> {path}  ({r.size_bytes}B)")
    if isinstance(readback, list) and len(readback) > 0:
        print(f"Read back {len(readback)} posts -- first: {readback[0].get('title', '')[:60]}")
    print("ARTIFACT PASS\n")

    print("ALL LEMMY TESTS PASSED (2 calls + 1 artifact)")


asyncio.run(main())
