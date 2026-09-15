#!/usr/bin/env python3
"""触发库的中间阶段点云导出，便于逐阶段可视化排查。

测量流水线现在由 ``FoodVolumeMeasurer`` 封装，``run()`` 里一次完成所有阶段。
要拿到中间点云，打开库自带的导出开关即可：

    measurer.set_save_middle_cloud(True).set_middle_cloud_dir(out_dir)

每次成功测量会写出 6 个 PCD：

  | 文件 | 阶段 |
  |------|------|
  | ``0_input_food.pcd``       | 原始输入食材帧 |
  | ``1_downsampled.pcd``      | 体素降采样后 |
  | ``2_remaining.pcd``        | 移除背景平面后 |
  | ``3_baseline_surface.pcd`` | 空炉基线表面（由基线栅格重建到世界坐标） |
  | ``4_food_components.pcd``  | 选中的食材连通块（合并） |
  | ``5_top_surface.pcd``      | 顶表面（体积积分所用） |

用法：

    python py/dump_intermediate_pcd.py                       # 用内置多食材样例
    python py/dump_intermediate_pcd.py <baseline.pcd> <food.pcd>
    python py/dump_intermediate_pcd.py <baseline.pcd> <food.pcd> -o OUT_DIR
    python py/dump_intermediate_pcd.py ... --show            # 额外用 Open3D 显示

运行环境：conda ``volume_measure``（Python 3.10）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vm_module import PROJECT_ROOT, TEST_DATA, load_vm  # noqa: E402

DEFAULT_BASELINE = TEST_DATA / "d405_260322274982_20260805_142739.pcd"
DEFAULT_FOOD = TEST_DATA / "food" / "d405_260322274982_20260811_184514.pcd"
MIDDLE_DATA_ROOT = PROJECT_ROOT / "middle_data" / "python"

STAGE_FILES = (
    "0_input_food.pcd",
    "1_downsampled.pcd",
    "2_remaining.pcd",
    "3_baseline_surface.pcd",
    "4_food_components.pcd",
    "5_top_surface.pcd",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 IM 流水线各阶段中间点云")
    parser.add_argument("baseline", nargs="?", default=str(DEFAULT_BASELINE), help="空炉基线 PCD")
    parser.add_argument("food", nargs="?", default=str(DEFAULT_FOOD), help="待测食材 PCD")
    parser.add_argument("-o", "--out-dir", default=None, help="输出目录（默认 middle_data/python/<food 名>）")
    parser.add_argument("--config", default=None, help="可选的 MeasurementConfig JSON")
    parser.add_argument("--show", action="store_true", help="导出后额外用 Open3D 显示各阶段点云")
    return parser.parse_args(argv)


def show_with_open3d(out_dir: Path) -> None:
    """若装了 Open3D，就把导出结果依次显示出来。"""
    try:
        import open3d as o3d
    except ImportError:
        print("[skip] 未安装 open3d，跳过显示")
        return

    for name in STAGE_FILES:
        path = out_dir / name
        if not path.exists():
            continue
        pcd = o3d.io.read_point_cloud(str(path))
        print(f"  [show] {name}: {len(pcd.points)} 点")
        o3d.visualization.draw_geometries([pcd], window_name=name)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    vm = load_vm()

    baseline_path = Path(args.baseline).resolve()
    food_path = Path(args.food).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else MIDDLE_DATA_ROOT / food_path.stem

    for path in (baseline_path, food_path):
        if not path.exists():
            raise SystemExit(f"点云不存在: {path}")

    print(f"baseline : {baseline_path}")
    print(f"food     : {food_path}")
    print(f"输出目录 : {out_dir}\n")

    baseline = vm.load_pcd(str(baseline_path))
    food = vm.load_pcd(str(food_path))
    print(f"输入点数 : baseline={len(baseline)} food={len(food)}")

    measurer = vm.FoodVolumeMeasurer()
    if args.config:
        if not measurer.load_config_from_json(args.config):
            raise SystemExit(f"配置加载失败: {args.config}")
    measurer.set_save_middle_cloud(True).set_middle_cloud_dir(str(out_dir))

    est = measurer.set_baseline([baseline]).set_food(food).run()
    print(f"\nstatus   : {vm.status_to_string(est.status)}")
    if est.status != vm.MeasurementStatus.kSuccess:
        print(f"message  : {est.message}")
        return 1

    print(f"体积     : {est.volume_cm3:.3f} cm^3 "
          f"(实测 {est.raw_volume_cm3:.3f} + 补洞 {est.interpolated_volume_cm3:.3f})")
    print(f"选中块数 : {est.component_count} | 标签 {list(est.selected_cluster_labels)}")

    print("\n导出文件:")
    missing = []
    for name in STAGE_FILES:
        path = out_dir / name
        if path.exists():
            print(f"  [ok]   {name}")
        else:
            print(f"  [miss] {name}")
            missing.append(name)
    if missing:
        print(f"\n注意: {len(missing)} 个阶段未产出（该阶段点云可能为空或被裁剪掉）")

    if args.show:
        show_with_open3d(out_dir)

    print("\n=== 完成 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
