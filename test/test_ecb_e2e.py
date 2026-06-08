"""E2E: ECB -- all 2 source_types via FetchDataTool + artifact storage."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.data_acquisition import FetchDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchDataTool()
store = ArtifactStore(run_id="ecb_e2e", base_dir="test/output")


async def main():
    print("\n### ECB -- all source_types + artifact ###\n")

    # 1/2  exchange_rates (EUR/USD)
    print("=" * 60)
    print("ecb / exchange_rates / USD / limit=5")
    print("=" * 60)
    r = await tool.execute(source_id="ecb", source_type="exchange_rates", symbol="USD", limit=5)
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2))
    assert len(r.content) == 5 and "value" in r.content[0], r.content
    print("PASS\n")

    # 2/2  interest_rates (default MRR_FR)
    print("=" * 60)
    print("ecb / interest_rates / MRR_FR / limit=5")
    print("=" * 60)
    r = await tool.execute(source_id="ecb", source_type="interest_rates", limit=5)
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2))
    assert len(r.content) == 5, r.content
    print("PASS\n")

    # also test DFR and MLFR
    print("=" * 60)
    print("ecb / interest_rates / DFR / limit=3")
    print("=" * 60)
    r = await tool.execute(source_id="ecb", source_type="interest_rates", rate_type="DFR", limit=3)
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2))
    assert len(r.content) == 3, r.content
    print("PASS\n")

    print("=" * 60)
    print("ecb / interest_rates / MLFR / limit=3")
    print("=" * 60)
    r = await tool.execute(source_id="ecb", source_type="interest_rates", rate_type="MLFR", limit=3)
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2))
    assert len(r.content) == 3, r.content
    print("PASS\n")

    # also test GBP and JPY exchange rates
    print("=" * 60)
    print("ecb / exchange_rates / GBP / limit=3")
    print("=" * 60)
    r = await tool.execute(source_id="ecb", source_type="exchange_rates", symbol="GBP", limit=3)
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2))
    assert len(r.content) == 3, r.content
    print("PASS\n")

    # artifact -- 500 days of EUR/USD daily rates
    print("=" * 60)
    print("ARTIFACT: ecb / exchange_rates / USD / limit=500  ->  test/output/")
    print("=" * 60)
    r = await tool.execute(source_id="ecb", source_type="exchange_rates", symbol="USD", limit=500)
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved {len(r.content)} observations -> {path}  ({r.size_bytes}B)")
    print(f"Read back {len(readback)} observations -- first: {json.dumps(readback[0])}")
    assert len(readback) == len(r.content), "readback mismatch"
    print("ARTIFACT PASS\n")

    print("ALL ECB TESTS PASSED (6 calls + 1 artifact)")


asyncio.run(main())
