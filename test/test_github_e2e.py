"""E2E: GitHub -- trending + search via FetchDataTool + artifact."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.data_acquisition import FetchDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchDataTool()
store = ArtifactStore(run_id="github_e2e", base_dir="test/output")


async def main():
    print("\n### GitHub -- all source_types + artifact ###\n")

    # 1/2 trending
    print("=" * 60)
    print("github / trending / python / last 7 days")
    print("=" * 60)
    r = await tool.execute(
        source_id="github", source_type="trending",
        language="python", since_days=7, limit=10,
    )
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content[:3], indent=2, default=str))
    assert len(r.content) > 0, "no trending repos"
    assert "name" in r.content[0], "missing name field"
    print(f"Got {len(r.content)} trending repos")
    for repo in r.content[:5]:
        desc = (repo.get('description', '') or '')[:60].encode('ascii', 'replace').decode()
        print(f"  - {repo['name']} ({repo.get('stars', 0)} stars) -- {desc}")
    print("PASS\n")

    # 2/2 search -- crypto/finance repos
    print("=" * 60)
    print("github / search / 'trading bot cryptocurrency'")
    print("=" * 60)
    r = await tool.execute(
        source_id="github", source_type="search",
        query="trading bot cryptocurrency", limit=10,
    )
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content[:3], indent=2, default=str))
    assert len(r.content) > 0, "no search results"
    print(f"Got {len(r.content)} repos")
    print("PASS\n")

    # artifact -- trending repos
    print("=" * 60)
    print("ARTIFACT: github / trending / all languages -> test/output/")
    print("=" * 60)
    r = await tool.execute(source_id="github", source_type="trending", since_days=7, limit=50)
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved {len(r.content)} repos -> {path}  ({r.size_bytes}B)")
    print(f"Read back {len(readback)} repos -- first: {readback[0].get('name', '')}")
    assert len(readback) == len(r.content), "readback mismatch"
    print("ARTIFACT PASS\n")

    print("ALL GITHUB TESTS PASSED (2 calls + 1 artifact)")


asyncio.run(main())
