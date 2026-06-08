"""E2E: BIS Statistics -- policy_rates + exchange_rates via FetchDataTool + artifact."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.data_acquisition import FetchDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchDataTool()
store = ArtifactStore(run_id="bis_e2e", base_dir="test/output")


async def main():
    print("\n### BIS Statistics -- all source_types + artifact ###\n")

    # 1/2 policy rates (US Fed)
    print("=" * 60)
    print("bis / policy_rates / US / last 12 monthly")
    print("=" * 60)
    r = await tool.execute(
        source_id="bis", source_type="policy_rates",
        country="US", frequency="M", last_n=12,
    )
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2, default=str)[:2000])
    obs = r.content.get("observations", [])
    assert len(obs) > 0, f"no observations: {r.content}"
    print(f"Got {len(obs)} observations")
    print("PASS\n")

    # 2/2 exchange rates (REER broad, Japan)
    print("=" * 60)
    print("bis / exchange_rates / JP / real / broad / last 12")
    print("=" * 60)
    r = await tool.execute(
        source_id="bis", source_type="exchange_rates",
        country="JP", rate_type="R", basis="B", last_n=12,
    )
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content, indent=2, default=str)[:2000])
    obs = r.content.get("observations", [])
    assert len(obs) > 0, f"no exchange rate observations: {r.content}"
    print(f"Got {len(obs)} observations")
    print("PASS\n")

    # artifact -- US policy rates 24-month
    print("=" * 60)
    print("ARTIFACT: bis / policy_rates / US 24mo -> test/output/")
    print("=" * 60)
    r = await tool.execute(
        source_id="bis", source_type="policy_rates",
        country="US", frequency="M", last_n=24,
    )
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved -> {path}  ({r.size_bytes}B)")
    assert readback.get("observations"), "readback empty"
    print("ARTIFACT PASS\n")

    print("ALL BIS TESTS PASSED (2 calls + 1 artifact)")


asyncio.run(main())
