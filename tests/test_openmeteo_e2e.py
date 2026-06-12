"""E2E: Open-Meteo -- forecast + historical via FetchDataTool + artifact."""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.data_acquisition import FetchMacroDataTool
from backend.tools.artifact import ArtifactStore

tool = FetchMacroDataTool()
store = ArtifactStore(run_id="openmeteo_e2e", base_dir="tests/output")


async def main():
    print("\n### Open-Meteo -- all source_types + artifact ###\n")

    # 1/2 forecast (New York)
    print("=" * 60)
    print("openmeteo / forecast / NYC (40.71, -74.01)")
    print("=" * 60)
    r = await tool.execute(
        source_id="openmeteo", source_type="forecast",
        latitude=40.71, longitude=-74.01, forecast_days=7,
    )
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content.get("daily", {}), indent=2, default=str)[:2000])
    assert "daily" in r.content, f"missing daily key, got: {list(r.content.keys())}"
    daily = r.content["daily"]
    assert "temperature_2m_max" in daily, f"missing temp field: {list(daily.keys())}"
    print("PASS\n")

    # 2/2 historical (London, Jan 2025)
    print("=" * 60)
    print("openmeteo / historical / London (51.51, -0.13) / 2025-01-01 to 2025-01-31")
    print("=" * 60)
    r = await tool.execute(
        source_id="openmeteo", source_type="historical",
        latitude=51.51, longitude=-0.13,
        start_date="2025-01-01", end_date="2025-01-31",
    )
    print(f"data_type={r.data_type}  size={r.size_bytes}B")
    print(json.dumps(r.content.get("daily", {}), indent=2, default=str)[:2000])
    assert "daily" in r.content, "missing daily key"
    assert len(r.content["daily"].get("time", [])) > 0, "no historical data"
    print(f"Got {len(r.content['daily']['time'])} days")
    print("PASS\n")

    # artifact -- 7-day forecast
    print("=" * 60)
    print("ARTIFACT: openmeteo / forecast / NYC 7d -> test/output/")
    print("=" * 60)
    r = await tool.execute(
        source_id="openmeteo", source_type="forecast",
        latitude=40.71, longitude=-74.01, forecast_days=16,
    )
    path = store.write(r.data_type, r.content)
    readback = store.read(path)
    print(f"Saved forecast -> {path}  ({r.size_bytes}B)")
    assert readback, "readback empty"
    print("ARTIFACT PASS\n")

    print("ALL OPEN-METEO TESTS PASSED (2 calls + 1 artifact)")


asyncio.run(main())
