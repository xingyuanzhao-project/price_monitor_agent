"""E2E: Hacker News -- all source_types via FetchDataTool + artifact storage."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.data_acquisition import FetchNewsDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchNewsDataTool()
store = ArtifactStore(run_id="hackernews_e2e", base_dir="tests/output")


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

    # 3/3  composed: top_stories IDs → story detail for each
    print("=" * 60)
    print("hackernews / top_stories + story / limit=5")
    print("=" * 60)
    ids_result = await tool.execute(source_id="hackernews", source_type="top_stories", limit=5)
    stories = []
    for sid in ids_result.content:
        sr = await tool.execute(source_id="hackernews", source_type="story", item_id=sid)
        stories.append(sr.content)
    print(f"Fetched {len(stories)} full stories")
    print(json.dumps(stories, indent=2))
    assert len(stories) == 5 and "title" in stories[0] and "score" in stories[0], stories
    print("PASS\n")

    # artifact -- top 10 stories with full detail
    print("=" * 60)
    print("ARTIFACT: hackernews / top_stories + story / limit=10  ->  test/output/")
    print("=" * 60)
    ids_r = await tool.execute(source_id="hackernews", source_type="top_stories", limit=10)
    detail_stories = []
    for sid in ids_r.content:
        sr = await tool.execute(source_id="hackernews", source_type="story", item_id=sid)
        detail_stories.append(sr.content)
    path = store.write("hackernews_top_stories_detail", detail_stories)
    readback = store.read(path)
    print(f"Saved {len(detail_stories)} stories -> {path}")
    print(f"Read back {len(readback)} stories -- #1: {readback[0]['title'][:60]}")
    assert len(readback) == len(detail_stories), "readback mismatch"
    print("ARTIFACT PASS\n")

    print("ALL HACKERNEWS TESTS PASSED (5+ calls + 1 artifact)")


asyncio.run(main())
