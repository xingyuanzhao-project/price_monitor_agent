"""E2E: Hacker News -- all 3 source_types via FetchDataTool + artifact storage."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.data_acquisition import FetchDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchDataTool()
store = ArtifactStore(run_id="hackernews_e2e", base_dir="test/output")


async def main():
    print("\n### Hacker News -- all source_types + artifact ###\n")

    # 1/3  top_stories (IDs only)
    print("=" * 60)
    print("hackernews / top_stories / limit=10")
    print("=" * 60)
    r = await tool.execute(source_id="hackernews", source_type="top_stories", limit=10)
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2))
    assert len(r.content) == 10, f"expected 10, got {len(r.content)}"
    story_id = r.content[0]
    print("PASS\n")

    # 2/3  story (single item by ID)
    print("=" * 60)
    print(f"hackernews / story / item_id={story_id}")
    print("=" * 60)
    r = await tool.execute(source_id="hackernews", source_type="story", item_id=story_id)
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2))
    assert r.content["id"] == story_id and r.content["title"], r.content
    print("PASS\n")

    # also fetch a second story
    story_id2 = (await tool.execute(source_id="hackernews", source_type="top_stories", limit=10)).content[4]
    print("=" * 60)
    print(f"hackernews / story / item_id={story_id2}")
    print("=" * 60)
    r = await tool.execute(source_id="hackernews", source_type="story", item_id=story_id2)
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2))
    assert r.content["id"] == story_id2, r.content
    print("PASS\n")

    # 3/3  top_stories_detail (full objects)
    print("=" * 60)
    print("hackernews / top_stories_detail / limit=5")
    print("=" * 60)
    r = await tool.execute(source_id="hackernews", source_type="top_stories_detail", limit=5)
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2))
    assert len(r.content) == 5 and "title" in r.content[0] and "score" in r.content[0], r.content
    print("PASS\n")

    # artifact -- top 30 stories with full detail
    print("=" * 60)
    print("ARTIFACT: hackernews / top_stories_detail / limit=30  ->  test/output/")
    print("=" * 60)
    r = await tool.execute(source_id="hackernews", source_type="top_stories_detail", limit=30)
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved {len(r.content)} stories -> {path}  ({r.size_bytes}B)")
    print(f"Read back {len(readback)} stories -- #1: {readback[0]['title'][:60]}")
    assert len(readback) == len(r.content), "readback mismatch"
    print("ARTIFACT PASS\n")

    print("ALL HACKERNEWS TESTS PASSED (5 calls + 1 artifact)")


asyncio.run(main())
