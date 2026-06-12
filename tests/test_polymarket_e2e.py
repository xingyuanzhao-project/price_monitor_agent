"""E2E: Polymarket -- markets + events + search via FetchDataTool + artifact."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.data_acquisition import FetchExchangeDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchExchangeDataTool()
store = ArtifactStore(run_id="polymarket_e2e", base_dir="tests/output")


async def main():
    print("\n### Polymarket -- all source_types + artifact ###\n")

    # 1/3 markets (top by volume)
    print("=" * 60)
    print("polymarket / markets / top 10")
    print("=" * 60)
    r = await tool.execute(source_id="polymarket", source_type="markets", limit=10)
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    preview = r.content[:3] if isinstance(r.content, list) else r.content
    print(json.dumps(preview, indent=2, default=str).encode("ascii", "replace").decode())
    assert isinstance(r.content, list), "expected list of markets"
    assert len(r.content) > 0, "no markets returned"
    first = r.content[0]
    assert "question" in first or "title" in first, f"bad market shape: {list(first.keys())[:5]}"
    print(f"Got {len(r.content)} markets")
    print("PASS\n")

    # 2/3 events
    print("=" * 60)
    print("polymarket / events / top 10")
    print("=" * 60)
    r = await tool.execute(source_id="polymarket", source_type="events", limit=10)
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    preview = r.content[:2] if isinstance(r.content, list) else r.content
    print(json.dumps(preview, indent=2, default=str).encode("ascii", "replace").decode())
    assert isinstance(r.content, list), "expected list of events"
    assert len(r.content) > 0, "no events returned"
    print(f"Got {len(r.content)} events")
    print("PASS\n")

    # 3/3 search
    print("=" * 60)
    print("polymarket / search / 'bitcoin'")
    print("=" * 60)
    r = await tool.execute(source_id="polymarket", source_type="search", query="bitcoin", limit=5)
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    content_str = json.dumps(r.content, indent=2, default=str).encode("ascii", "replace").decode()
    print(content_str[:2000])
    assert r.content, "search returned empty"
    print("PASS\n")

    # artifact -- full markets dump
    print("=" * 60)
    print("ARTIFACT: polymarket / markets / top 50 -> test/output/")
    print("=" * 60)
    r = await tool.execute(source_id="polymarket", source_type="markets", limit=50)
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved {len(r.content)} markets -> {path}  ({r.size_bytes}B)")
    assert len(readback) == len(r.content), "readback mismatch"
    print("ARTIFACT PASS\n")

    print("ALL POLYMARKET TESTS PASSED (3 calls + 1 artifact)")


asyncio.run(main())
