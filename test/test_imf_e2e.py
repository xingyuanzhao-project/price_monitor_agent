"""E2E: IMF DataMapper -- indicator + list via FetchDataTool + artifact."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.data_acquisition import FetchDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchDataTool()
store = ArtifactStore(run_id="imf_e2e", base_dir="test/output")


async def main():
    print("\n### IMF DataMapper -- all source_types + artifact ###\n")

    # 1/2 indicator -- GDP growth for USA, GBR, CHN
    print("=" * 60)
    print("imf / indicator / NGDP_RPCH (GDP growth %) / USA,GBR,CHN")
    print("=" * 60)
    r = await tool.execute(
        source_id="imf", source_type="indicator",
        indicator="NGDP_RPCH", countries="USA,GBR,CHN",
    )
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2, default=str)[:2000])
    assert r.content.get("indicator") == "NGDP_RPCH", r.content
    assert "data" in r.content, "data key missing"
    print("PASS\n")

    # 2/2 list indicators
    print("=" * 60)
    print("imf / list (all available indicators)")
    print("=" * 60)
    r = await tool.execute(source_id="imf", source_type="list")
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content[:5], indent=2, default=str))
    assert len(r.content) > 0, "no indicators"
    print(f"Found {len(r.content)} available indicators")
    print("PASS\n")

    # artifact -- inflation data
    print("=" * 60)
    print("ARTIFACT: imf / indicator / PCPIPCH (inflation) / USA -> test/output/")
    print("=" * 60)
    r = await tool.execute(
        source_id="imf", source_type="indicator",
        indicator="PCPIPCH", countries="USA",
    )
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved -> {path}  ({r.size_bytes}B)")
    print(f"Read back indicator={readback.get('indicator')}")
    assert readback.get("indicator") == "PCPIPCH", "readback mismatch"
    print("ARTIFACT PASS\n")

    print("ALL IMF TESTS PASSED (2 calls + 1 artifact)")


asyncio.run(main())
