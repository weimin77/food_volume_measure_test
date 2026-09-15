#!/usr/bin/env python3
"""Wrap the food volume measurement Python library into the het-ai HPO flow.

Training data
-------------
``test_data/ground_truth.json`` maps each food PCD to its ground-truth volume in
cm³.  ``load_data()`` resolves each entry with ``load_pcd``, so a training
sample is the **loaded point cloud + its ground-truth volume**.  The ground
truth is the **C++ interface** result (``measure_from_pcd`` with the
reference/default config) and is generated offline by ``make_ground_truth.py``.

HPO objective
-------------
The search space explores *coarser* algorithm settings (larger voxel, larger
plane distance threshold, larger integration cell, higher min height) and the
objective is to **minimize the mean relative volume error** against the C++
ground truth.  This answers "which steps/parameters can be relaxed with little
impact on the result".

Run (conda ``volume_measure`` environment, Python 3.10):

    python het_ai_volume_trainer.py          # dry-run only (fast validation)
    python het_ai_volume_trainer.py --run    # full Optuna HPO
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from het_ai.studio import BaseTrainer, DataBundle, TrainConfig

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vm_module import TEST_DATA, load_vm  # noqa: E402


class VolumeCalibrationTrainer(BaseTrainer):
    """Calibrate/ablate the PCD-IM parameters against the C++ reference result."""

    objectives = {"relative_error": "minimize"}

    # Low bounds are the reference config values; the search widens toward
    # coarser settings. The dry-run therefore samples the reference config and
    # must produce a near-zero relative error by construction.
    @BaseTrainer.search(
        voxel_size_m=BaseTrainer.TunableFloat(0.003, 0.02, log=True),
        plane_distance_threshold_m=BaseTrainer.TunableFloat(0.003, 0.02, log=True),
        integration_resolution_m=BaseTrainer.TunableFloat(0.003, 0.02, log=True),
        min_height_m=BaseTrainer.TunableFloat(0.0015, 0.01, log=True),
    )
    def train(
        self,
        data: DataBundle,
        voxel_size_m: float,
        plane_distance_threshold_m: float,
        integration_resolution_m: float,
        min_height_m: float,
    ) -> float:
        vm = data.meta["vm"]
        baseline_cloud = data.meta["baseline_cloud"]
        samples = data.splits["train"]["samples"]

        cfg = vm.MeasurementConfig()
        cfg.voxel_size_m = float(voxel_size_m)
        cfg.plane_distance_threshold_m = float(plane_distance_threshold_m)
        cfg.integration_resolution_m = float(integration_resolution_m)
        cfg.min_height_m = float(min_height_m)

        measurer = vm.FoodVolumeMeasurer()
        rel_errors: list[float] = []
        for sample in samples:
            measurer.set_config(cfg).set_baseline([baseline_cloud]).set_food(sample["cloud"])
            est = measurer.run()
            if est.status != vm.MeasurementStatus.kSuccess:
                # A failed measurement is the worst possible outcome for this sample.
                rel_errors.append(1.0)
                continue
            gt = sample["gt_volume"]
            denom = abs(gt) if abs(gt) > 1.0e-6 else 1.0e-6
            rel_errors.append(abs(est.volume_cm3 - gt) / denom)

        mean_rel_error = float(sum(rel_errors) / len(rel_errors)) if rel_errors else 1.0
        self.report(0, mean_rel_error)
        return mean_rel_error

    def load_data(self, dvc_data_root: str) -> DataBundle:
        """Load the point clouds + ground-truth volumes from the JSON table."""
        vm = load_vm()
        gt_path = TEST_DATA / "ground_truth.json"
        if not gt_path.exists():
            raise RuntimeError(f"未找到真值表 {gt_path}，请先运行: python make_ground_truth.py")

        doc = json.loads(gt_path.read_text(encoding="utf-8"))
        baseline_pcd = TEST_DATA / doc["baseline_pcd"]
        baseline_cloud = vm.load_pcd(str(baseline_pcd))
        if len(baseline_cloud) == 0:
            raise RuntimeError(f"基线点云为空: {baseline_pcd}")

        samples = []
        for item in doc["samples"]:
            pcd = TEST_DATA / item["pcd"]
            if not pcd.exists():
                print(f"  [load_data] 缺失点云，跳过: {pcd}")
                continue
            cloud = vm.load_pcd(str(pcd))
            samples.append({"cloud": cloud, "gt_volume": float(item["gt_volume_cm3"]), "name": pcd.name})

        if not samples:
            raise RuntimeError("ground_truth.json 中没有可用样本")

        print(f"[load_data] 基线点云: {baseline_pcd.name} ({len(baseline_cloud)} points)")
        print(f"[load_data] 训练样本: {len(samples)} 个（点云 + 真值体积 cm³）:")
        for s in samples:
            print(f"    {s['name']}: {len(s['cloud'])} points -> {s['gt_volume']:.3f} cm3")

        return DataBundle(
            splits={"train": {"samples": samples}},
            meta={"vm": vm, "baseline_cloud": baseline_cloud},
        )

    def mock_data(self) -> DataBundle:
        """Fast subset for dry-run validation.

        Reuses the real JSON table but keeps only the first few samples, so the
        dry-run validates the whole pipeline (load -> train -> report) quickly
        without measuring every PCD.  The full ``run()`` still calls
        ``load_data()`` and uses all samples.
        """
        data = self.load_data(self.config.dvc_data_root)
        samples = data.splits["train"]["samples"][:3]
        print(f"[mock_data] 使用前 {len(samples)} 个样本作为 dry-run 子集")
        return DataBundle(splits={"train": {"samples": samples}}, meta=data.meta)


def main() -> int:
    config = TrainConfig(n_trials=20, direction="minimize")
    trainer = VolumeCalibrationTrainer(config)

    if "--run" in sys.argv:
        print("=== het-ai full HPO run ===")
        result = trainer.run()
        print("Best metric:", result.metric_dict)
        print("Best params:", result.params_dict)
    else:
        print("=== het-ai dry_run() ===")
        result = trainer.dry_run()
        print("\n[dry_run] score          =", result["score"])
        print("[dry_run] elapsed        =", f"{result['elapsed']:.2f}s")
        print("[dry_run] export_path    =", result["export_path"])
        print("[dry_run] ✅ 全流程验证通过")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
