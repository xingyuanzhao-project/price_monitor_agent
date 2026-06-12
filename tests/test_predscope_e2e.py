"""E2E: PredScope -- markets + resolved via FetchDataTool + artifact."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.data_acquisition import FetchExchangeDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchExchangeDataTool()
store = ArtifactStore(run_id="predscope_e2e", base_dir="tests/output")


async def main():
    print("\n### PredScope -- all source_types + artifact ###\n")

    # 1/2 markets (top 100)
    print("=" * 60)
    print("predscope / markets")
    print("=" * 60)
    r = await tool.execute(source_id="predscope", source_type="markets")
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    if isinstance(r.content, dict):
        markets = r.content.get("markets", [])
        meta = r.content.get("meta", {})
        print(f"meta: {json.dumps(meta, indent=2, default=str)}")
        preview = markets[:3]
    elif isinstance(r.content, list):
        markets = r.content
        preview = markets[:3]
    else:
        markets = []
        preview = r.content
    print(json.dumps(preview, indent=2, default=str).encode("ascii", "replace").decode()[:2000])
    assert len(markets) > 0 or r.content, "no markets returned"
    if markets:
        m = markets[0]
        assert "title" in m, f"missing title: {list(m.keys())[:8]}"
        assert "outcomes" in m, f"missing outcomes: {list(m.keys())[:8]}"
        print(f"Got {len(markets)} markets")
    print("PASS\n")

    # 2/2 resolved
    print("=" * 60)
    print("predscope / resolved")
    print("=" * 60)
    r = await tool.execute(source_id="predscope", source_type="resolved")
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    if isinstance(r.content, dict):
        resolved = r.content.get("markets", r.content.get("resolved", []))
        print(f"Keys: {list(r.content.keys())}")
        preview = resolved[:2] if isinstance(resolved, list) else resolved
    else:
        preview = r.content[:2] if isinstance(r.content, list) else r.content
    print(json.dumps(preview, indent=2, default=str).encode("ascii", "replace").decode()[:2000])
    assert r.content, "resolved returned empty"
    print("PASS\n")

    # artifact -- full markets dump
    print("=" * 60)
    print("ARTIFACT: predscope / markets -> test/output/")
    print("=" * 60)
    r = await tool.execute(source_id="predscope", source_type="markets")
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved -> {path}  ({r.size_bytes}B)")
    assert readback, "readback empty"
    print("ARTIFACT PASS\n")

    print("ALL PREDSCOPE TESTS PASSED (2 calls + 1 artifact)")


asyncio.run(main())
