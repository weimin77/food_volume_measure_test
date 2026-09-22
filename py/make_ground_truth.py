#!/usr/bin/env python3
"""Generate ``test_data/ground_truth.json`` for the food volume dataset.

Ground truth for each food PCD is the **C++ interface** result
(``measure_from_pcd`` run with the reference/default configuration).  Run this
script whenever the PCDs in ``test_data/food/`` change; the trainer then reads
the generated JSON instead of re-computing volumes.

Run (conda ``volume_measure`` environment):

    python make_ground_truth.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vm_module import TEST_DATA, load_vm  # noqa: E402

BASELINE_PCD = TEST_DATA / "base.pcd"
FOOD_DIR = TEST_DATA / "food"
OUT_JSON = TEST_DATA / "ground_truth.json"


def main() -> int:
    vm = load_vm()
    ref_cfg = vm.MeasurementConfig()  # all volume_defaults == reference config

    samples = []
    for pcd in sorted(FOOD_DIR.glob("*.pcd")):
        est = vm.measure_from_pcd([str(BASELINE_PCD)], str(pcd), ref_cfg)
        if est.status != vm.MeasurementStatus.kSuccess:
            print(f"skip {pcd.name}: {vm.status_to_string(est.status)}")
            continue
        samples.append(
            {
                "pcd": str(pcd.relative_to(TEST_DATA)),
                "gt_volume_cm3": float(est.volume_cm3),
            }
        )
        print(f"{pcd.name}: {est.volume_cm3:.3f} cm3")

    doc = {
        "baseline_pcd": str(BASELINE_PCD.relative_to(TEST_DATA)),
        "volume_unit": "cm3",
        "samples": samples,
    }
    OUT_JSON.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_JSON} ({len(samples)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
