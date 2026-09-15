#!/usr/bin/env python3
"""冒烟测试编译好的 ``food_volume_measure_python`` pybind 模块。

覆盖公开 API 的四条主线：端到端便捷函数、链式 ``FoodVolumeMeasurer``、
逐连通块明细、日志 sink。

数据取自 ``py/test_data/``：
  - ``d405_..._20260805_142739.pcd``  空炉基线
  - ``d405_..._20260819_180010.pcd``  食材帧

运行（conda ``volume_measure`` 环境，Python 3.10）：
    python py/test_pybind_module.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vm_module import TEST_DATA, load_vm  # noqa: E402

BASELINE_PCD = TEST_DATA / "d405_260322274982_20260805_142739.pcd"
FOOD_PCD = TEST_DATA / "d405_260322274982_20260819_180010.pcd"


def test_measure_from_pcd(vm) -> None:
    """便捷函数：measure_from_pcd(baseline_pcds, food_pcd, cfg)。"""
    print("\n=== 1. measure_from_pcd（端到端便捷函数） ===")
    cfg = vm.MeasurementConfig()
    est = vm.measure_from_pcd([str(BASELINE_PCD)], str(FOOD_PCD), cfg)
    print("status           :", est.status, "=", vm.status_to_string(est.status))
    if est.status != vm.MeasurementStatus.kSuccess:
        print("message          :", est.message)
        return
    print("volume_cm3       :", round(est.volume_cm3, 3))
    print("raw / interpolated:", round(est.raw_volume_cm3, 3), "/", round(est.interpolated_volume_cm3, 3))
    print("measured / interpolated cells:", est.measured_cells, "/", est.interpolated_cells)
    print("mean / max height:", round(est.mean_height_m, 6), "/", round(est.max_height_m, 6))
    print("footprint_area_m2:", round(est.footprint_area_m2, 6))


def test_measurer_class(vm) -> None:
    """链式 FoodVolumeMeasurer + 显式 load_pcd。"""
    print("\n=== 2. FoodVolumeMeasurer 链式调用 ===")
    baseline = vm.load_pcd(str(BASELINE_PCD))
    food = vm.load_pcd(str(FOOD_PCD))
    print("baseline points  :", len(baseline))
    print("food points      :", len(food))

    measurer = vm.FoodVolumeMeasurer()
    est = (
        measurer.set_input_unit(vm.LengthUnit.kMeter)
        .set_voxel_size(0.003)
        .set_integration_resolution(0.003)
        .set_height_range(0.0015, -1.0)
        .set_cluster_params(0.010, 8)
        .set_baseline([baseline])
        .set_food(food)
        .run()
    )
    print("status           :", est.status, "=", vm.status_to_string(est.status))
    ok = est.status == vm.MeasurementStatus.kSuccess
    print("volume_cm3       :", round(est.volume_cm3, 3) if ok else "-")

    cfg = measurer.config()
    print("cfg.voxel_size_m :", cfg.voxel_size_m)
    print("cfg.save_middle_cloud / dir:", cfg.save_middle_cloud, "/", cfg.middle_cloud_dir)


def test_component_estimates(vm) -> None:
    """逐连通块明细：与 selected_cluster_labels 一一对应。"""
    print("\n=== 3. 逐连通块明细 ===")
    est = vm.measure_from_pcd([str(BASELINE_PCD)], str(FOOD_PCD), vm.MeasurementConfig())
    if est.status != vm.MeasurementStatus.kSuccess:
        print("skip: status =", vm.status_to_string(est.status))
        return

    print("cluster_count    :", est.cluster_count)
    print("component_count  :", est.component_count)
    print("labels           :", list(est.selected_cluster_labels))
    total = 0.0
    for i, c in enumerate(est.component_estimates):
        label = est.selected_cluster_labels[i] if i < len(est.selected_cluster_labels) else -1
        print(f"  块[{label}] volume={c.volume_cm3:.3f} cm3 "
              f"cells(measured/interp)={c.measured_cells}/{c.interpolated_cells} "
              f"max_h={c.max_height_m:.4f} m")
        total += c.volume_cm3
    print("逐块求和         :", round(total, 3), "cm3 (总计", round(est.volume_cm3, 3), "cm3)")


def test_types_and_helpers(vm) -> None:
    """基础类型与自由函数。"""
    print("\n=== 4. 类型 / 自由函数 ===")
    print("1 mm ->", vm.length_unit_to_meter_scale(vm.LengthUnit.kMillimeter), "m")

    print("is_finite(Point3f):", vm.is_finite(vm.Point3f(1.0, 2.0, 3.0)))

    cloud = vm.PointCloud()
    cloud.append(vm.Point3f(0.0, 0.0, 0.0))
    cloud.append(vm.Point3f(1.0, 1.0, 1.0))
    print("PointCloud len   :", len(cloud))

    for i in range(int(vm.MeasurementStatus.kUnsupportedPlatform) + 1):
        status = vm.MeasurementStatus(i)
        assert vm.status_to_string(status), status
    print("status_to_string : 全部枚举可读")


def test_logging(vm) -> None:
    """日志：级别 + 控制台 + 文件 sink。"""
    print("\n=== 5. 日志 ===")
    log_path = SCRIPT_DIR / "pybind_test.log"
    vm.log_set_level(vm.LogLevel.kInfo)
    vm.log_set_console(False)
    vm.log_set_file(str(log_path), vm.LogFileMode.kTruncate)
    vm.log_info("pybind module logging works")
    vm.log_warning("this is a warning")
    vm.log_close_file()
    print("wrote", log_path.name, "| level =", vm.log_get_level(), "| console =", vm.log_get_console())


def main() -> int:
    vm = load_vm()
    print("module version   :", vm.__version__)
    print("baseline pcd     :", BASELINE_PCD.name)
    print("food pcd         :", FOOD_PCD.name)

    test_measure_from_pcd(vm)
    test_measurer_class(vm)
    test_component_estimates(vm)
    test_types_and_helpers(vm)
    test_logging(vm)

    print("\n=== 完成 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
