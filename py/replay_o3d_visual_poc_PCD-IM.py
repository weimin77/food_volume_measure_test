"""PCD-IM（基准面高度差积分）体积计算与四阶段回放。

固定烤箱场景中，空炉点云先被离散为基准面局部坐标系的高度图。食物点云在相同
栅格取最高表面，以 ``(食物高度 - 空炉高度) × 栅格面积`` 逐格累加体积。
反光导致的缺失深度仅可在单一前景组件内部、并通过面积/边界/曲面拟合等约束后
保守补全；补全格与原始实测格分开统计、着色和计入体积。所有几何量单位为米。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

# 限制数值库线程，避免树莓派 GUI/Open3D 回放与线性代数计算争抢 CPU。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

#from benchmark_metrics import BenchmarkMetrics, BenchmarkTimer, print_benchmark

from replay_o3d_visual_poc import (
    RAW_VIEW,
    SEGMENT_VIEW,
    VOLUME_VIEW,
    apply_uniform_color,
    compute_compact_minimum_volume_obb,
    draw_stage,
    format_plane,
    format_vector,
    load_point_cloud,
    parse_target_label,
    print_cluster_candidates,
    resolve_target_candidate,
    select_cluster,
    summarize_clusters,
)

import open3d as o3d


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PCD = SCRIPT_DIR / "test_data" / "d405_260322274982_20260819_180010.pcd"
DEFAULT_BASELINE_PCD = SCRIPT_DIR / "test_data" / "d405_260322274982_20260805_142739.pcd"
DEFAULT_BASELINE_STATE = REPO_ROOT / "captures" / "pcd_i_empty_baseline.json"
BASELINE_DISPLAY_COLOR = np.array([0.08, 0.82, 0.28], dtype=np.float64)
INTERPOLATED_DIFFERENCE_COLOR = np.array([1.00, 0.55, 0.00], dtype=np.float64)
UNFILLED_HOLE_COLOR = np.array([0.90, 0.10, 0.90], dtype=np.float64)


@dataclass
class PcdIntegralMetrics:
    """IM 一次运行的汇总指标；既包含正式积分体积，也保留参考几何与 Benchmark。"""
    input_points: int
    downsampled_points: int
    cluster_count: int
    selected_cluster_labels: list[int]
    selected_cluster_points: int
    component_count: int
    top_surface_points: int
    baseline_frame_count: int
    plane_1: np.ndarray
    plane_2: np.ndarray | None
    occupied_cell_count: int
    footprint_bbox_cell_count: int
    footprint_area_m2: float
    mean_height_m: float
    max_height_m: float
    integral_volume_m3: float
    raw_integral_volume_m3: float
    aabb_volume_m3: float
    obb_compact_volume_m3: float
    convex_hull_volume_m3: float
    missing_baseline_cell_count: int
    #benchmark: BenchmarkMetrics


@dataclass
class BaselineHeightMap:
    """空炉基线的局部平面坐标系及每个 ``(u,v)`` 栅格的稳健高度中位数。"""
    source_paths: list[str]
    frame_count: int
    plane: np.ndarray
    plane_origin: np.ndarray
    plane_u: np.ndarray
    plane_v: np.ndarray
    plane_n: np.ndarray
    cell_size_m: float
    height_by_cell: dict[tuple[int, int], float]
    point_count_by_cell: dict[tuple[int, int], int]
    cell_indices: np.ndarray
    heights_m: np.ndarray
    surface_points_xyz: np.ndarray


@dataclass
class HeightGrid:
    """食材差分积分网格；实测/插补单元通过 ``is_interpolated`` 明确区分。"""
    cell_indices: np.ndarray
    cell_centers_uv: np.ndarray
    base_points_xyz: np.ndarray
    top_points_xyz: np.ndarray
    top_points_rgb: np.ndarray
    baseline_heights_m: np.ndarray
    food_heights_m: np.ndarray
    heights_m: np.ndarray
    plane_origin: np.ndarray
    plane_u: np.ndarray
    plane_v: np.ndarray
    plane_n: np.ndarray
    cell_size_m: float
    occupied_cell_count: int
    bbox_cell_count: int
    footprint_area_m2: float
    bbox_area_m2: float
    occupancy_ratio: float
    mean_height_m: float
    max_height_m: float
    volume_m3: float
    missing_baseline_cell_count: int
    measured_cell_count: int
    interpolated_cell_count: int
    raw_volume_m3: float
    interpolated_volume_m3: float
    is_interpolated: np.ndarray
    component_labels: np.ndarray


@dataclass
class PlaneRoi:
    """基线平面坐标中的内缩 ROI，同时保留原始基线 bbox 以便追溯边界裁剪。"""
    u_min_m: float
    u_max_m: float
    v_min_m: float
    v_max_m: float
    bbox_u_min_m: float
    bbox_u_max_m: float
    bbox_v_min_m: float
    bbox_v_max_m: float
    border_margin_m: float


def parse_args() -> argparse.Namespace:
    """定义 IM 的采集重放、基线、分割、积分、补洞和可视化 CLI 参数。"""
    parser = argparse.ArgumentParser(
        description="Replay a plane-height point-cloud integral POC from a D405 PCD capture."
    )
    parser.add_argument("--pcd", default=str(DEFAULT_PCD), help="Point-cloud input used for the PCD integral POC.")
    parser.add_argument("--voxel-size", type=float, default=0.003, help="Voxel downsample size in meters.")
    parser.add_argument(
        "--plane1-distance-threshold",
        type=float,
        default=0.003,
        help="RANSAC distance threshold for the base-plane candidate.",
    )
    parser.add_argument(
        "--plane2-distance-threshold",
        type=float,
        default=0.003,
        help="RANSAC distance threshold for the optional secondary-plane removal.",
    )
    parser.add_argument(
        "--remove-secondary-plane",
        action="store_true",
        help="Remove the second dominant plane before clustering. Disabled by default to preserve planar food.",
    )
    parser.add_argument("--cluster-eps", type=float, default=0.02, help="DBSCAN eps parameter.")
    parser.add_argument(
        "--foreground-cluster-eps",
        type=float,
        default=0.010,
        help="DBSCAN eps in baseline-plane coordinates for separate food components.",
    )
    parser.add_argument("--cluster-min-points", type=int, default=8, help="DBSCAN min_points parameter.")
    parser.add_argument(
        "--target-label",
        default="auto",
        help="Foreground labels to retain (comma separated), or 'auto'/'all' to retain all eligible food components.",
    )
    parser.add_argument(
        "--baseline-pcd",
        action="append",
        default=[str(DEFAULT_BASELINE_PCD)],
        help="Empty-oven baseline PCD. Repeat this option to aggregate multiple empty frames.",
    )
    parser.add_argument(
        "--baseline-state",
        default=str(DEFAULT_BASELINE_STATE),
        help="Baseline state JSON used when --baseline-pcd is omitted.",
    )
    parser.add_argument(
        "--baseline-max-surface-height-m",
        type=float,
        default=0.03,
        help="Ignore empty-scene points higher than this above the baseline plane when building the baseline map.",
    )
    parser.add_argument(
        "--baseline-fill-radius-cells",
        type=int,
        default=2,
        help="Neighbor radius used to fill missing baseline cells under the food footprint.",
    )
    parser.add_argument(
        "--integration-resolution-m",
        type=float,
        default=0.003,
        help="Base-plane raster cell size used for height integration.",
    )
    parser.add_argument(
        "--roi-border-margin-m",
        type=float,
        default=0.02,
        help="Exclude target points within this border margin from the baseline-plane footprint bbox.",
    )
    parser.add_argument(
        "--min-height-m",
        type=float,
        default=0.0015,
        help="Ignore object points whose base-plane height is below this threshold.",
    )
    parser.add_argument(
        "--max-height-m",
        type=float,
        default=None,
        help="Optional maximum allowed height above the base plane.",
    )
    parser.add_argument(
        "--hole-fill-max-cells",
        type=int,
        default=9,
        help="Maximum size of one enclosed IM hole to interpolate; 0 disables food-surface hole filling.",
    )
    parser.add_argument(
        "--hole-fill-neighbor-radius-cells",
        type=int,
        default=2,
        help="Chebyshev neighbor radius used for local IM hole interpolation.",
    )
    parser.add_argument(
        "--hole-fill-max-neighbor-height-delta-m",
        type=float,
        default=0.012,
        help="Reject a hole when its local measured-neighbor height range exceeds this value.",
    )
    parser.add_argument(
        "--curve-fill-max-hole-area-cm2",
        type=float,
        default=8.0,
        help="Maximum area of one component-owned enclosed hole eligible for quadratic-surface completion.",
    )
    parser.add_argument(
        "--curve-fill-max-component-area-ratio",
        type=float,
        default=0.30,
        help="Maximum single curved-hole area / measured component footprint ratio.",
    )
    parser.add_argument(
        "--curve-fill-max-imputed-ratio",
        type=float,
        default=0.25,
        help="Maximum accumulated inferred-cell ratio for any one foreground component.",
    )
    parser.add_argument(
        "--curve-fill-rim-radius-cells",
        type=int,
        default=4,
        help="Chebyshev radius used to collect same-component rim samples for curved-hole fitting.",
    )
    parser.add_argument(
        "--curve-fill-min-rim-samples",
        type=int,
        default=16,
        help="Minimum measured same-component rim cells required for quadratic-surface completion.",
    )
    parser.add_argument(
        "--curve-fill-min-rim-coverage",
        type=float,
        default=0.75,
        help="Minimum occupied angular coverage of the curved-hole rim, from 0 to 1.",
    )
    parser.add_argument(
        "--curve-fill-max-fit-rmse-m",
        type=float,
        default=0.004,
        help="Maximum robust quadratic rim-fit RMSE in metres.",
    )
    parser.add_argument(
        "--curve-fill-max-prediction-rise-m",
        type=float,
        default=0.015,
        help="Maximum allowed predicted height excursion above/below the measured rim range in metres.",
    )
    parser.add_argument(
        "--max-obb-faces",
        type=int,
        default=500,
        help="Maximum convex-hull face count before compact-OBB normal deduplication.",
    )
    parser.add_argument("--seed", type=int, default=43, help="Deterministic random seed for Open3D and NumPy.")
    parser.add_argument("--headless", action="store_true", help="Skip desktop visualization windows.")
    parser.add_argument(
        "--auto-close-seconds",
        type=float,
        default=None,
        help="Automatically close each desktop stage after N seconds.",
    )
    parser.add_argument(
        "--max-visualization-cells",
        type=int,
        default=1200,
        help="Cap the number of integrated cells rendered in Stage 4 to protect the Pi desktop session.",
    )
    parser.add_argument(
        "--result-json",
        default=None,
        help="Optional JSON path for machine-readable validation results.",
    )
    return parser.parse_args()


def normalize_plane(plane: np.ndarray) -> np.ndarray:
    """归一化平面 ``[a,b,c,d]``，使点到平面的有符号距离可直接以米解释。"""
    normal = np.asarray(plane[:3], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        raise RuntimeError(f"平面法向量无效: {plane}")
    return np.array([normal[0] / norm, normal[1] / norm, normal[2] / norm, plane[3] / norm], dtype=np.float64)


def orient_plane_toward_object(plane: np.ndarray, object_points: np.ndarray) -> np.ndarray:
    """按食物候选点的中位有符号高度翻转法向，约定食物位于正高度一侧。"""
    signed_heights = object_points @ plane[:3] + plane[3]
    if np.median(signed_heights) < 0.0:
        return -plane
    return plane


def build_plane_frame(plane_normal: np.ndarray, plane_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从平面法向构造右手 ``(u,v,n)`` 坐标系；原点取平面内点的均值。"""
    origin = plane_points.mean(axis=0).astype(np.float64)
    for reference in (
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    ):
        tangent = reference - np.dot(reference, plane_normal) * plane_normal
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-8:
            plane_u = tangent / tangent_norm
            plane_v = np.cross(plane_normal, plane_u)
            plane_v /= np.linalg.norm(plane_v)
            return origin, plane_u.astype(np.float64), plane_v.astype(np.float64)
    raise RuntimeError("无法为积分底面建立稳定的局部坐标系。")


def resolve_path_from_repo(raw_path: str | Path) -> Path:
    """将相对仓库路径或用户家目录缩写规范化为绝对路径。"""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def resolve_baseline_paths(args: argparse.Namespace) -> list[Path]:
    """优先使用重复传入的 ``--baseline-pcd``，否则从 GUI 维护的基线状态文件读取。"""
    if args.baseline_pcd:
        return [resolve_path_from_repo(item) for item in args.baseline_pcd]

    state_path = resolve_path_from_repo(args.baseline_state)
    if not state_path.exists():
        raise RuntimeError(
            "未找到空炉 baseline。"
            " 请先通过 GUI 的“设最新为空炉基线”按钮设置，"
            "或在命令行中传入 --baseline-pcd captures/xxx_empty.pcd。"
        )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    baseline_value = state.get("baseline_pcd")
    if not baseline_value:
        raise RuntimeError(f"baseline state 文件缺少 baseline_pcd 字段: {state_path}")
    return [resolve_path_from_repo(baseline_value)]


def project_points_to_plane_frame(
    points_xyz: np.ndarray,
    plane_origin: np.ndarray,
    plane_u: np.ndarray,
    plane_v: np.ndarray,
    plane_n: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把相机三维点转换到基准面 ``(u,v,height)``；height 沿正法向，单位米。"""
    offsets = points_xyz - plane_origin[None, :]
    uv = np.column_stack(
        [
            offsets @ plane_u,
            offsets @ plane_v,
        ]
    )
    heights = offsets @ plane_n
    return uv.astype(np.float64), heights.astype(np.float64)


def uv_to_cell_indices(uv: np.ndarray, cell_size_m: float) -> np.ndarray:
    """用向下取整把连续平面坐标映射为稳定的二维整数栅格索引。"""
    return np.floor(uv / cell_size_m).astype(np.int32)


def build_plane_roi_from_baseline(baseline: BaselineHeightMap, border_margin_m: float) -> PlaneRoi:
    """根据空炉有效栅格 bbox 内缩边缘距离，排除炉壁/角落的易受干扰区域。"""
    if border_margin_m < 0.0:
        raise ValueError("--roi-border-margin-m 不能小于 0。")
    if len(baseline.cell_indices) == 0:
        raise RuntimeError("baseline 没有任何 cell，无法构建平面 ROI。")

    cell_centers_uv = (baseline.cell_indices.astype(np.float64) + 0.5) * baseline.cell_size_m
    bbox_u_min_m = float(np.min(cell_centers_uv[:, 0]))
    bbox_u_max_m = float(np.max(cell_centers_uv[:, 0]))
    bbox_v_min_m = float(np.min(cell_centers_uv[:, 1]))
    bbox_v_max_m = float(np.max(cell_centers_uv[:, 1]))

    u_min_m = bbox_u_min_m + border_margin_m
    u_max_m = bbox_u_max_m - border_margin_m
    v_min_m = bbox_v_min_m + border_margin_m
    v_max_m = bbox_v_max_m - border_margin_m
    if u_min_m >= u_max_m or v_min_m >= v_max_m:
        raise RuntimeError(
            "ROI 边界收缩后无有效区域。"
            f" current bbox=({bbox_u_min_m:.4f},{bbox_u_max_m:.4f},{bbox_v_min_m:.4f},{bbox_v_max_m:.4f}),"
            f" margin={border_margin_m:.4f}"
        )

    return PlaneRoi(
        u_min_m=u_min_m,
        u_max_m=u_max_m,
        v_min_m=v_min_m,
        v_max_m=v_max_m,
        bbox_u_min_m=bbox_u_min_m,
        bbox_u_max_m=bbox_u_max_m,
        bbox_v_min_m=bbox_v_min_m,
        bbox_v_max_m=bbox_v_max_m,
        border_margin_m=float(border_margin_m),
    )


def mask_points_inside_plane_roi(uv: np.ndarray, plane_roi: PlaneRoi | None) -> np.ndarray:
    """返回平面坐标是否位于 ROI；无 ROI 时所有点均可通过。"""
    if plane_roi is None:
        return np.ones(len(uv), dtype=bool)

    return (
        (uv[:, 0] >= plane_roi.u_min_m)
        & (uv[:, 0] <= plane_roi.u_max_m)
        & (uv[:, 1] >= plane_roi.v_min_m)
        & (uv[:, 1] <= plane_roi.v_max_m)
    )


def extract_points_and_colors(point_cloud: o3d.geometry.PointCloud) -> tuple[np.ndarray, np.ndarray]:
    """导出 Open3D 点与 RGB；无可靠颜色时提供白色，保证显示管线长度一致。"""
    points = np.asarray(point_cloud.points, dtype=np.float64)
    if point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors, dtype=np.float64)
        if len(colors) == len(points):
            return points, colors

    default_colors = np.tile(np.array([[1.0, 1.0, 1.0]], dtype=np.float64), (len(points), 1))
    return points, default_colors


def build_baseline_heightmap(
    baseline_paths: list[Path],
    cell_size_m: float,
    plane_distance_threshold: float,
    voxel_size_m: float,
    max_surface_height_m: float,
    orientation_points: np.ndarray | None = None,
) -> BaselineHeightMap:
    """从一帧或多帧空炉 PCD 建立基准面高度图。

    首帧 RANSAC 平面确定局部坐标系；每帧空炉点都投影到此坐标，并在单元内取
    高度中位数，抵抗噪点。只保留接近空炉表面的高度，避免把意外前景写入基线。
    """
    if not baseline_paths:
        raise RuntimeError("baseline_paths 为空。")
    if cell_size_m <= 0.0:
        raise ValueError("--integration-resolution-m 必须大于 0。")
    if max_surface_height_m <= 0.0:
        raise ValueError("--baseline-max-surface-height-m 必须大于 0。")

    # 坐标系只从首帧确定，多帧随后在同一坐标系聚合，才能逐格比较。
    first_baseline = load_point_cloud(baseline_paths[0]).voxel_down_sample(voxel_size=voxel_size_m)
    plane, inliers = first_baseline.segment_plane(
        distance_threshold=plane_distance_threshold,
        ransac_n=3,
        num_iterations=1000,
    )
    plane = normalize_plane(np.asarray(plane, dtype=np.float64))
    if orientation_points is not None and len(orientation_points) > 0:
        plane = orient_plane_toward_object(plane, np.asarray(orientation_points, dtype=np.float64))
    ground = first_baseline.select_by_index(inliers)
    plane_origin, plane_u, plane_v = build_plane_frame(plane[:3], np.asarray(ground.points, dtype=np.float64))
    plane_n = plane[:3]

    height_samples_by_cell: dict[tuple[int, int], list[float]] = {}
    sample_count_by_cell: dict[tuple[int, int], int] = {}

    # 每格取中位数而非均值，降低随机深度飞点与边缘混合像素的影响。
    for baseline_path in baseline_paths:
        baseline_pcd = load_point_cloud(baseline_path)
        baseline_points = np.asarray(baseline_pcd.points, dtype=np.float64)
        baseline_uv, baseline_heights = project_points_to_plane_frame(
            baseline_points,
            plane_origin=plane_origin,
            plane_u=plane_u,
            plane_v=plane_v,
            plane_n=plane_n,
        )
        mask = np.isfinite(baseline_heights)
        mask &= baseline_heights >= -0.003
        mask &= baseline_heights <= max_surface_height_m
        filtered_uv = baseline_uv[mask]
        filtered_heights = baseline_heights[mask]
        if len(filtered_heights) == 0:
            continue

        filtered_cells = uv_to_cell_indices(filtered_uv, cell_size_m)
        for cell_index, height in zip(filtered_cells, filtered_heights, strict=False):
            key = (int(cell_index[0]), int(cell_index[1]))
            height_samples_by_cell.setdefault(key, []).append(float(height))
            sample_count_by_cell[key] = sample_count_by_cell.get(key, 0) + 1

    if not height_samples_by_cell:
        raise RuntimeError("空炉 baseline 没有生成任何有效 heightmap cell。")

    sorted_cells = sorted(height_samples_by_cell.items())
    cell_indices = np.array([cell for cell, _ in sorted_cells], dtype=np.int32)
    heights_m = np.array([float(np.median(samples)) for _, samples in sorted_cells], dtype=np.float64)
    cell_centers_uv = (cell_indices.astype(np.float64) + 0.5) * cell_size_m
    surface_points_xyz = (
        plane_origin[None, :]
        + cell_centers_uv[:, 0:1] * plane_u[None, :]
        + cell_centers_uv[:, 1:2] * plane_v[None, :]
        + heights_m[:, None] * plane_n[None, :]
    )
    height_by_cell = {
        (int(cell_indices[idx, 0]), int(cell_indices[idx, 1])): float(heights_m[idx])
        for idx in range(len(cell_indices))
    }

    return BaselineHeightMap(
        source_paths=[str(path) for path in baseline_paths],
        frame_count=len(baseline_paths),
        plane=plane,
        plane_origin=plane_origin,
        plane_u=plane_u,
        plane_v=plane_v,
        plane_n=plane_n,
        cell_size_m=float(cell_size_m),
        height_by_cell=height_by_cell,
        point_count_by_cell=sample_count_by_cell,
        cell_indices=cell_indices,
        heights_m=heights_m,
        surface_points_xyz=surface_points_xyz.astype(np.float64),
    )


def build_food_surface_map(
    obj: o3d.geometry.PointCloud,
    baseline: BaselineHeightMap,
    plane_roi: PlaneRoi | None = None,
) -> tuple[dict[tuple[int, int], dict[str, np.ndarray | float]], dict[str, int]]:
    """按基准面栅格提取食材最高表面点。

    一个格只保留最高点，防止同一深度柱的侧壁/内部点被重复积分；同时记录 RGB，
    便于 Stage 3/4 回放与采集图像对应。
    """
    points, colors = extract_points_and_colors(obj)
    uv, heights = project_points_to_plane_frame(
        points,
        plane_origin=baseline.plane_origin,
        plane_u=baseline.plane_u,
        plane_v=baseline.plane_v,
        plane_n=baseline.plane_n,
    )
    cells = uv_to_cell_indices(uv, baseline.cell_size_m)

    surface_by_cell: dict[tuple[int, int], dict[str, np.ndarray | float]] = {}
    discarded_negative = 0
    discarded_outside_roi = 0
    for cell_index, height, point_xyz, color_rgb in zip(cells, heights, points, colors, strict=False):
        if not np.isfinite(height) or height < -0.003:
            discarded_negative += 1
            continue
        if plane_roi is not None:
            point_u = (int(cell_index[0]) + 0.5) * baseline.cell_size_m
            point_v = (int(cell_index[1]) + 0.5) * baseline.cell_size_m
            if not (
                plane_roi.u_min_m <= point_u <= plane_roi.u_max_m
                and plane_roi.v_min_m <= point_v <= plane_roi.v_max_m
            ):
                discarded_outside_roi += 1
                continue

        key = (int(cell_index[0]), int(cell_index[1]))
        # 顶表面由最大正向高度定义，低于它的同列点不会重复进入面积积分。
        current = surface_by_cell.get(key)
        if current is None or float(height) > float(current["height_m"]):
            surface_by_cell[key] = {
                "height_m": float(height),
                "point_xyz": point_xyz.astype(np.float64),
                "color_rgb": color_rgb.astype(np.float64),
            }

    if not surface_by_cell:
        raise RuntimeError("目标簇没有生成任何可用的顶部 surface cell。")

    return surface_by_cell, {
        "input_points": int(len(points)),
        "discarded_negative_points": int(discarded_negative),
        "discarded_outside_roi_points": int(discarded_outside_roi),
        "surface_cell_count": int(len(surface_by_cell)),
    }


def build_valid_difference_point_cloud(
    obj: o3d.geometry.PointCloud,
    baseline: BaselineHeightMap,
    min_height_m: float,
    max_height_m: float | None,
    fill_radius_cells: int,
    plane_roi: PlaneRoi | None = None,
) -> tuple[o3d.geometry.PointCloud, dict[str, int]]:
    """对密集点逐点做空炉高度差过滤，生成仅用于前景对象聚类的点云。

    它与最终“每格最高点”的积分图不同：保留所有有效点可提高 DBSCAN 将相邻物体
    分开的稳定性。ROI、最小高度和基线缺失过滤与正式积分保持同一判据。
    """
    if fill_radius_cells < 0:
        raise ValueError("--baseline-fill-radius-cells 不能小于 0。")

    points, colors = extract_points_and_colors(obj)
    uv, heights = project_points_to_plane_frame(
        points,
        plane_origin=baseline.plane_origin,
        plane_u=baseline.plane_u,
        plane_v=baseline.plane_v,
        plane_n=baseline.plane_n,
    )
    cells = uv_to_cell_indices(uv, baseline.cell_size_m)

    kept_points: list[np.ndarray] = []
    kept_colors: list[np.ndarray] = []
    discarded_negative = 0
    discarded_outside_roi = 0
    rejected_below_min_height = 0
    missing_baseline = 0
    direct_baseline = 0
    neighbor_filled_baseline = 0

    for cell_index, point_xyz, color_rgb, food_height in zip(cells, points, colors, heights, strict=False):
        if not np.isfinite(food_height) or food_height < -0.003:
            discarded_negative += 1
            continue

        cell_key = (int(cell_index[0]), int(cell_index[1]))
        if plane_roi is not None:
            cell_center_u = (cell_key[0] + 0.5) * baseline.cell_size_m
            cell_center_v = (cell_key[1] + 0.5) * baseline.cell_size_m
            if not (
                plane_roi.u_min_m <= cell_center_u <= plane_roi.u_max_m
                and plane_roi.v_min_m <= cell_center_v <= plane_roi.v_max_m
            ):
                discarded_outside_roi += 1
                continue
        # 基线点不存在时宁可丢弃，不把无基准的深度点误认作食材。
        baseline_height, lookup_mode = lookup_baseline_height(
            cell_key,
            baseline_height_by_cell=baseline.height_by_cell,
            fill_radius_cells=fill_radius_cells,
        )
        if baseline_height is None:
            missing_baseline += 1
            continue

        if lookup_mode == "direct":
            direct_baseline += 1
        else:
            neighbor_filled_baseline += 1

        diff_height = float(food_height) - baseline_height
        if diff_height < min_height_m:
            rejected_below_min_height += 1
            continue
        if max_height_m is not None:
            diff_height = min(diff_height, max_height_m)

        kept_points.append(point_xyz.astype(np.float64))
        kept_colors.append(color_rgb.astype(np.float64))

    if kept_points:
        dense_points = np.asarray(kept_points, dtype=np.float64)
        dense_colors = np.asarray(kept_colors, dtype=np.float64)
        dense_cloud = make_point_cloud(dense_points, dense_colors)
    else:
        dense_cloud = o3d.geometry.PointCloud()

    return dense_cloud, {
        "input_points": int(len(points)),
        "valid_points": int(len(kept_points)),
        "discarded_negative_points": int(discarded_negative),
        "discarded_outside_roi_points": int(discarded_outside_roi),
        "rejected_below_min_height_points": int(rejected_below_min_height),
        "missing_baseline_points": int(missing_baseline),
        "baseline_direct_points": int(direct_baseline),
        "baseline_neighbor_filled_points": int(neighbor_filled_baseline),
    }


def parse_target_labels(raw_value: str) -> set[int] | None:
    """解析多组件标签：``auto/all`` 返回 None，逗号列表返回用户指定的标签集合。"""
    normalized = raw_value.strip().lower()
    if normalized in {"", "auto", "all"}:
        return None
    try:
        labels = {int(item.strip()) for item in raw_value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError("--target-label 必须是 auto、all 或以逗号分隔的整数标签。") from exc
    if not labels:
        raise ValueError("--target-label 未包含任何标签。")
    return labels


def cluster_foreground_components(
    foreground: o3d.geometry.PointCloud,
    baseline: BaselineHeightMap,
    cluster_eps: float,
    cluster_min_points: int,
) -> tuple[np.ndarray, list[object]]:
    """在基准面二维坐标而非相机三维坐标中对有效高度差点执行 DBSCAN。

    这样对象分割由炉腔平面内距离决定，避免不同高度或倾斜视角把同一食材拆散；
    返回的标签仍与原前景点顺序逐一对应。
    """
    points, colors = extract_points_and_colors(foreground)
    if len(points) == 0:
        raise RuntimeError("高度差前景为空，无法进行对象聚类。")
    uv, _ = project_points_to_plane_frame(
        points,
        plane_origin=baseline.plane_origin,
        plane_u=baseline.plane_u,
        plane_v=baseline.plane_v,
        plane_n=baseline.plane_n,
    )
    # 将 (u,v) 嵌入 z=0 的临时点云，只借用 Open3D 的 DBSCAN 实现。
    projected = make_point_cloud(
        np.column_stack([uv, np.zeros(len(uv), dtype=np.float64)]),
        colors,
    )
    labels = np.asarray(
        projected.cluster_dbscan(
            eps=cluster_eps,
            min_points=cluster_min_points,
            print_progress=False,
        )
    )
    if not np.any(labels >= 0):
        raise RuntimeError("高度差前景 DBSCAN 没有找到任何非噪声对象。请调整 eps 或 min_pts。")
    return labels, summarize_clusters(foreground, labels)


def resolve_foreground_component_labels(
    candidates: list[object],
    cluster_min_points: int,
    requested_labels: set[int] | None,
) -> tuple[list[int], str, int]:
    """默认保留所有达到最小尺寸的前景组件；也支持手动指定组件标签。"""
    min_candidate_size = max(32, cluster_min_points * 4)
    eligible = [candidate for candidate in candidates if int(candidate.point_count) >= min_candidate_size]
    if requested_labels is None:
        labels = [int(candidate.label) for candidate in eligible]
        mode = "all-foreground-components"
    else:
        available = {int(candidate.label) for candidate in eligible}
        missing = sorted(requested_labels - available)
        if missing:
            raise RuntimeError(f"指定的前景标签不存在或过小: {missing}。可用标签: {sorted(available)}")
        labels = sorted(requested_labels)
        mode = "manual-foreground-labels"
    if not labels:
        raise RuntimeError(
            "高度差前景中没有达到最小对象尺寸的食物组件。"
            f" min_candidate_size={min_candidate_size}"
        )
    return labels, mode, min_candidate_size


def merge_point_clouds(point_clouds: list[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
    """保序合并多个已选前景组件；合并仅用于统一积分，不抹除组件标签记录。"""
    if not point_clouds:
        raise ValueError("point_clouds 不能为空。")
    points_list: list[np.ndarray] = []
    colors_list: list[np.ndarray] = []
    for cloud in point_clouds:
        points, colors = extract_points_and_colors(cloud)
        points_list.append(points)
        colors_list.append(colors)
    return make_point_cloud(np.vstack(points_list), np.vstack(colors_list))


def lookup_baseline_height(
    cell_key: tuple[int, int],
    baseline_height_by_cell: dict[tuple[int, int], float],
    fill_radius_cells: int,
) -> tuple[float | None, str]:
    """查询空炉栅格高度：直接命中优先，缺失时在有限切比雪夫半径内取中位数。"""
    direct = baseline_height_by_cell.get(cell_key)
    if direct is not None:
        return float(direct), "direct"

    for radius in range(1, fill_radius_cells + 1):
        neighbors = []
        for du in range(-radius, radius + 1):
            for dv in range(-radius, radius + 1):
                sample = baseline_height_by_cell.get((cell_key[0] + du, cell_key[1] + dv))
                if sample is not None:
                    neighbors.append(float(sample))
        if neighbors:
            return float(np.median(neighbors)), f"neighbor-r{radius}"

    return None, "missing"


def make_height_grid(
    cell_indices: np.ndarray,
    baseline_heights_m: np.ndarray,
    food_heights_m: np.ndarray,
    heights_m: np.ndarray,
    top_points_rgb: np.ndarray,
    baseline: BaselineHeightMap,
    missing_baseline_cell_count: int,
    is_interpolated: np.ndarray | None = None,
    component_labels: np.ndarray | None = None,
) -> HeightGrid:
    """从逐格高度构造完整积分几何与统计。

    体积严格为 ``sum(height) * cell_size²``。``is_interpolated`` 将实测体积和
    补洞增量拆开，既供可信度评估，也供回放使用不同颜色显示。
    """
    if len(cell_indices) == 0:
        raise RuntimeError("无法构建空的积分格网。")
    if is_interpolated is None:
        is_interpolated = np.zeros(len(cell_indices), dtype=bool)
    if component_labels is None:
        component_labels = np.zeros(len(cell_indices), dtype=np.int32)
    cell_indices = np.asarray(cell_indices, dtype=np.int32)
    baseline_heights_m = np.asarray(baseline_heights_m, dtype=np.float64)
    food_heights_m = np.asarray(food_heights_m, dtype=np.float64)
    heights_m = np.asarray(heights_m, dtype=np.float64)
    top_points_rgb = np.asarray(top_points_rgb, dtype=np.float64)
    is_interpolated = np.asarray(is_interpolated, dtype=bool)
    component_labels = np.asarray(component_labels, dtype=np.int32)
    if len(component_labels) != len(cell_indices):
        raise ValueError("component_labels 数量必须与积分 cell 数量一致。")
    # 每个积分柱以栅格中心为位置，底点使用空炉高度而非理想平面 z=0。
    cell_centers_uv = (cell_indices.astype(np.float64) + 0.5) * baseline.cell_size_m
    base_points_xyz = (
        baseline.plane_origin[None, :]
        + cell_centers_uv[:, 0:1] * baseline.plane_u[None, :]
        + cell_centers_uv[:, 1:2] * baseline.plane_v[None, :]
        + baseline_heights_m[:, None] * baseline.plane_n[None, :]
    )
    top_points_xyz = base_points_xyz + heights_m[:, None] * baseline.plane_n[None, :]
    min_index = cell_indices.min(axis=0)
    max_index = cell_indices.max(axis=0)
    bbox_cell_count = int((max_index[0] - min_index[0] + 1) * (max_index[1] - min_index[1] + 1))
    cell_area_m2 = baseline.cell_size_m * baseline.cell_size_m
    occupied_cell_count = int(len(heights_m))
    measured_cell_count = int(np.count_nonzero(~is_interpolated))
    interpolated_cell_count = int(np.count_nonzero(is_interpolated))
    # 原始结果仅统计实测高度；补全格只累加到 repaired/总积分结果。
    raw_volume_m3 = float(np.sum(heights_m[~is_interpolated]) * cell_area_m2)
    interpolated_volume_m3 = float(np.sum(heights_m[is_interpolated]) * cell_area_m2)
    volume_m3 = raw_volume_m3 + interpolated_volume_m3
    return HeightGrid(
        cell_indices=cell_indices,
        cell_centers_uv=cell_centers_uv,
        base_points_xyz=base_points_xyz.astype(np.float64),
        top_points_xyz=top_points_xyz.astype(np.float64),
        top_points_rgb=top_points_rgb,
        baseline_heights_m=baseline_heights_m,
        food_heights_m=food_heights_m,
        heights_m=heights_m,
        plane_origin=baseline.plane_origin,
        plane_u=baseline.plane_u,
        plane_v=baseline.plane_v,
        plane_n=baseline.plane_n,
        cell_size_m=float(baseline.cell_size_m),
        occupied_cell_count=occupied_cell_count,
        bbox_cell_count=bbox_cell_count,
        footprint_area_m2=float(occupied_cell_count * cell_area_m2),
        bbox_area_m2=float(bbox_cell_count * cell_area_m2),
        occupancy_ratio=float(occupied_cell_count / bbox_cell_count) if bbox_cell_count else 0.0,
        mean_height_m=float(np.mean(heights_m)),
        max_height_m=float(np.max(heights_m)),
        volume_m3=volume_m3,
        missing_baseline_cell_count=int(missing_baseline_cell_count),
        measured_cell_count=measured_cell_count,
        interpolated_cell_count=interpolated_cell_count,
        raw_volume_m3=raw_volume_m3,
        interpolated_volume_m3=interpolated_volume_m3,
        is_interpolated=is_interpolated,
        component_labels=component_labels,
    )


def build_height_grid_from_baseline_difference(
    food_surface_by_cell: dict[tuple[int, int], dict[str, np.ndarray | float]],
    baseline: BaselineHeightMap,
    min_height_m: float,
    max_height_m: float | None,
    fill_radius_cells: int,
    component_label_by_cell: dict[tuple[int, int], int] | None = None,
) -> tuple[HeightGrid, dict[str, int]]:
    """将食物顶表面与空炉图逐格相减，得到未经补洞的原始 IM 积分网格。"""
    if fill_radius_cells < 0:
        raise ValueError("--baseline-fill-radius-cells 不能小于 0。")

    used_cell_indices: list[np.ndarray] = []
    used_cell_centers_uv: list[np.ndarray] = []
    base_points_xyz: list[np.ndarray] = []
    top_points_xyz: list[np.ndarray] = []
    top_points_rgb: list[np.ndarray] = []
    baseline_heights_m: list[float] = []
    food_heights_m: list[float] = []
    diff_heights_m: list[float] = []
    component_labels: list[int] = []
    baseline_mode_histogram = {"direct": 0, "neighbor-filled": 0, "missing": 0}

    # 排序保证 JSON/回放在相同输入下拥有稳定单元顺序，方便版本间核对。
    for cell_key, payload in sorted(food_surface_by_cell.items()):
        baseline_height, lookup_mode = lookup_baseline_height(
            cell_key,
            baseline_height_by_cell=baseline.height_by_cell,
            fill_radius_cells=fill_radius_cells,
        )
        if baseline_height is None:
            baseline_mode_histogram["missing"] += 1
            continue

        if lookup_mode == "direct":
            baseline_mode_histogram["direct"] += 1
        else:
            baseline_mode_histogram["neighbor-filled"] += 1

        food_height = float(payload["height_m"])
        diff_height = food_height - baseline_height
        if diff_height < min_height_m:
            continue
        if max_height_m is not None:
            diff_height = min(diff_height, max_height_m)

        cell_index = np.array([cell_key[0], cell_key[1]], dtype=np.int32)
        cell_center_uv = (cell_index.astype(np.float64) + 0.5) * baseline.cell_size_m
        base_xyz = (
            baseline.plane_origin
            + cell_center_uv[0] * baseline.plane_u
            + cell_center_uv[1] * baseline.plane_v
            + baseline_height * baseline.plane_n
        )
        top_xyz = base_xyz + diff_height * baseline.plane_n

        used_cell_indices.append(cell_index)
        used_cell_centers_uv.append(cell_center_uv)
        base_points_xyz.append(base_xyz.astype(np.float64))
        top_points_xyz.append(top_xyz.astype(np.float64))
        top_points_rgb.append(np.asarray(payload["color_rgb"], dtype=np.float64))
        baseline_heights_m.append(float(baseline_height))
        food_heights_m.append(float(food_height))
        diff_heights_m.append(float(diff_height))
        component_labels.append(int(component_label_by_cell.get(cell_key, 0)) if component_label_by_cell else 0)

    if not diff_heights_m:
        raise RuntimeError("baseline 差分后没有任何有效的积分 cell。请检查空炉 baseline 或高度阈值。")

    cell_indices = np.asarray(used_cell_indices, dtype=np.int32)
    top_points_rgb_arr = np.asarray(top_points_rgb, dtype=np.float64)
    baseline_heights_arr = np.asarray(baseline_heights_m, dtype=np.float64)
    food_heights_arr = np.asarray(food_heights_m, dtype=np.float64)
    diff_heights_arr = np.asarray(diff_heights_m, dtype=np.float64)
    return (
        make_height_grid(
            cell_indices=cell_indices,
            baseline_heights_m=baseline_heights_arr,
            food_heights_m=food_heights_arr,
            heights_m=diff_heights_arr,
            top_points_rgb=top_points_rgb_arr,
            baseline=baseline,
            missing_baseline_cell_count=int(baseline_mode_histogram["missing"]),
            component_labels=np.asarray(component_labels, dtype=np.int32),
        ),
        baseline_mode_histogram,
    )


def find_enclosed_hole_components(
    cell_indices: np.ndarray,
    blocked_cell_indices: np.ndarray | None = None,
) -> list[list[tuple[int, int]]]:
    """查找不连通到外边界的四连通空格组件，即候选深度孔洞。

    先在带一圈 padding 的 bbox 边缘泛洪标记“外部空气”，剩余空格才是封闭孔。
    对单一食材组件检测时，其他组件被视为阻塞格，因而二者之间的空隙不会被当作
    反光缺失深度。此函数只识别几何孔，是否补全仍由后续安全条件决定。
    """
    if len(cell_indices) == 0:
        return []
    minimum = cell_indices.min(axis=0) - 1
    maximum = cell_indices.max(axis=0) + 1
    shape = tuple((maximum - minimum + 1).tolist())
    occupied = np.zeros(shape, dtype=bool)
    occupancy_cells = [np.asarray(cell_indices, dtype=np.int32)]
    if blocked_cell_indices is not None and len(blocked_cell_indices):
        occupancy_cells.append(np.asarray(blocked_cell_indices, dtype=np.int32))
    for cell in np.vstack(occupancy_cells):
        if np.any(cell < minimum) or np.any(cell > maximum):
            continue
        occupied[int(cell[0] - minimum[0]), int(cell[1] - minimum[1])] = True
    # 从边界泛洪到达的空格属于外部，绝不能被插补为食物。
    exterior = np.zeros_like(occupied)
    pending: deque[tuple[int, int]] = deque()
    rows, cols = occupied.shape
    for row in range(rows):
        for col in (0, cols - 1):
            if not occupied[row, col] and not exterior[row, col]:
                exterior[row, col] = True
                pending.append((row, col))
    for col in range(cols):
        for row in (0, rows - 1):
            if not occupied[row, col] and not exterior[row, col]:
                exterior[row, col] = True
                pending.append((row, col))
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while pending:
        row, col = pending.popleft()
        for dr, dc in neighbors:
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols and not occupied[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                pending.append((nr, nc))
    internal = (~occupied) & (~exterior)
    holes: list[list[tuple[int, int]]] = []
    while np.any(internal):
        row, col = np.argwhere(internal)[0]
        internal[row, col] = False
        component = [(int(row + minimum[0]), int(col + minimum[1]))]
        pending = deque([(int(row), int(col))])
        while pending:
            current_row, current_col = pending.popleft()
            for dr, dc in neighbors:
                nr, nc = current_row + dr, current_col + dc
                if 0 <= nr < rows and 0 <= nc < cols and internal[nr, nc]:
                    internal[nr, nc] = False
                    pending.append((nr, nc))
                    component.append((int(nr + minimum[0]), int(nc + minimum[1])))
        holes.append(component)
    return holes


def collect_component_hole_rim_cells(
    hole: list[tuple[int, int]],
    label_by_cell: dict[tuple[int, int], int],
    component_label: int,
    radius_cells: int,
) -> list[tuple[int, int]]:
    """收集孔洞周围指定半径内、且确属同一组件的已测量边缘栅格。"""
    hole_cells = set(hole)
    rim_cells: set[tuple[int, int]] = set()
    for cell in hole_cells:
        for du in range(-radius_cells, radius_cells + 1):
            for dv in range(-radius_cells, radius_cells + 1):
                if du == 0 and dv == 0:
                    continue
                candidate = (cell[0] + du, cell[1] + dv)
                if candidate in hole_cells or label_by_cell.get(candidate) != component_label:
                    continue
                rim_cells.add(candidate)
    return sorted(rim_cells)


def hole_boundary_component_labels(
    hole: list[tuple[int, int]],
    label_by_cell: dict[tuple[int, int], int],
) -> set[int]:
    """统计孔洞 8 邻域归属的组件标签；只有单标签边界才可能安全补洞。"""
    hole_cells = set(hole)
    labels: set[int] = set()
    for cell in hole_cells:
        for du in range(-1, 2):
            for dv in range(-1, 2):
                if du == 0 and dv == 0:
                    continue
                candidate = (cell[0] + du, cell[1] + dv)
                if candidate not in hole_cells and candidate in label_by_cell:
                    labels.add(int(label_by_cell[candidate]))
    return labels


def interpolate_small_component_hole(
    hole: list[tuple[int, int]],
    height_by_cell: dict[tuple[int, int], float],
    label_by_cell: dict[tuple[int, int], int],
    component_label: int,
    neighbor_radius_cells: int,
    max_neighbor_height_delta_m: float,
    baseline: BaselineHeightMap,
    baseline_fill_radius_cells: int,
) -> list[tuple[tuple[int, int], float, float]] | None:
    """对小而平滑的同组件孔洞作距离加权高度插值。

    每个待填格至少需要 3 个同组件邻居，且邻居高度极差不能超过阈值；任一格
    不满足条件即整体拒绝，避免把边缘、褶皱或两个物品间隙平滑填满。
    """
    additions: list[tuple[tuple[int, int], float, float]] = []
    for cell in hole:
        samples: list[tuple[float, float]] = []
        for du in range(-neighbor_radius_cells, neighbor_radius_cells + 1):
            for dv in range(-neighbor_radius_cells, neighbor_radius_cells + 1):
                if du == 0 and dv == 0:
                    continue
                candidate = (cell[0] + du, cell[1] + dv)
                if label_by_cell.get(candidate) != component_label:
                    continue
                neighbor_height = height_by_cell.get(candidate)
                if neighbor_height is None:
                    continue
                distance = float(np.hypot(du, dv))
                samples.append((neighbor_height, 1.0 / max(distance, 1e-6)))
        if len(samples) < 3:
            return None
        sample_heights = np.asarray([sample[0] for sample in samples], dtype=np.float64)
        if float(np.ptp(sample_heights)) > max_neighbor_height_delta_m:
            return None
        # 距离倒数加权，使相邻表面点比较远的点对孔中心预测影响更大。
        weights = np.asarray([sample[1] for sample in samples], dtype=np.float64)
        baseline_height, _ = lookup_baseline_height(
            cell,
            baseline_height_by_cell=baseline.height_by_cell,
            fill_radius_cells=baseline_fill_radius_cells,
        )
        if baseline_height is None:
            return None
        additions.append((cell, float(baseline_height), float(np.average(sample_heights, weights=weights))))
    return additions


def fit_quadratic_component_hole(
    hole: list[tuple[int, int]],
    height_by_cell: dict[tuple[int, int], float],
    label_by_cell: dict[tuple[int, int], int],
    component_label: int,
    cell_size_m: float,
    rim_radius_cells: int,
    min_rim_samples: int,
    min_rim_coverage: float,
    max_fit_rmse_m: float,
    max_prediction_rise_m: float,
    min_height_m: float,
    max_height_m: float | None,
) -> tuple[np.ndarray | None, dict[str, int | float | str]]:
    """为单组件孔洞拟合带安全约束的局部二次高度曲面。

    模型为 ``a+bx+cy+dx²+exy+fy²``。只有边缘样本数量/环向覆盖足够、拟合残差
    足够小、预测值不越出边缘允许范围且仍符合 min/max 高度时才接受；否则返回
    ``None`` 和拒绝原因，调用方会将该孔标为紫色且不计体积。
    """
    rim_cells = collect_component_hole_rim_cells(
        hole=hole,
        label_by_cell=label_by_cell,
        component_label=component_label,
        radius_cells=rim_radius_cells,
    )
    details: dict[str, int | float | str] = {
        "rim_sample_count": int(len(rim_cells)),
        "rim_coverage_ratio": 0.0,
        "fit_rmse_m": 0.0,
        "outlier_count": 0,
    }
    if len(rim_cells) < min_rim_samples:
        details["reason"] = "insufficient-rim-samples"
        return None, details

    hole_center = np.mean(np.asarray(hole, dtype=np.float64) + 0.5, axis=0) * cell_size_m
    rim_uv = (np.asarray(rim_cells, dtype=np.float64) + 0.5) * cell_size_m
    relative_uv = rim_uv - hole_center[None, :]
    angles = np.arctan2(relative_uv[:, 1], relative_uv[:, 0])
    # 以八个方位桶检查边缘是否环绕孔洞，防止只取到单侧边缘仍外推曲面。
    occupied_angle_bins = np.unique(np.floor((angles + np.pi) / (2.0 * np.pi) * 8.0).astype(np.int32) % 8)
    coverage_ratio = float(len(occupied_angle_bins) / 8.0)
    details["rim_coverage_ratio"] = coverage_ratio
    if coverage_ratio < min_rim_coverage:
        details["reason"] = "insufficient-rim-coverage"
        return None, details

    def design_matrix(relative: np.ndarray) -> np.ndarray:
        """构造二次曲面最小二乘的六个基函数列。"""
        x = relative[:, 0]
        y = relative[:, 1]
        return np.column_stack([np.ones(len(relative)), x, y, x * x, x * y, y * y])

    design = design_matrix(relative_uv)
    observed_heights = np.asarray([height_by_cell[cell] for cell in rim_cells], dtype=np.float64)
    coefficients, _, rank, _ = np.linalg.lstsq(design, observed_heights, rcond=None)
    if rank < 6:
        details["reason"] = "rank-deficient-fit"
        return None, details

    # 首次拟合后以 MAD 剔除高度飞点，再在保留的内点上重拟合一次。
    residuals = observed_heights - design @ coefficients
    median_residual = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median_residual)))
    robust_limit = max(0.0015, 3.0 * 1.4826 * mad)
    inlier_mask = np.abs(residuals - median_residual) <= robust_limit
    if int(np.count_nonzero(inlier_mask)) >= min_rim_samples and not np.all(inlier_mask):
        coefficients, _, rank, _ = np.linalg.lstsq(design[inlier_mask], observed_heights[inlier_mask], rcond=None)
        if rank < 6:
            details["reason"] = "rank-deficient-refit"
            return None, details
        residuals = observed_heights[inlier_mask] - design[inlier_mask] @ coefficients
    details["outlier_count"] = int(len(observed_heights) - np.count_nonzero(inlier_mask))
    fit_rmse_m = float(np.sqrt(np.mean(np.square(residuals))))
    details["fit_rmse_m"] = fit_rmse_m
    if fit_rmse_m > max_fit_rmse_m:
        details["reason"] = "fit-rmse-too-large"
        return None, details

    hole_uv = (np.asarray(hole, dtype=np.float64) + 0.5) * cell_size_m
    predictions = design_matrix(hole_uv - hole_center[None, :]) @ coefficients
    # 预测不能明显超出已测边缘高度范围，限制凸起/凹陷外推带来的体积虚增。
    lower_bound = max(0.0, float(np.min(observed_heights)) - max_prediction_rise_m)
    upper_bound = float(np.max(observed_heights)) + max_prediction_rise_m
    if np.any(predictions < lower_bound) or np.any(predictions > upper_bound):
        details["reason"] = "prediction-outside-rim-bound"
        return None, details
    if np.any(predictions < min_height_m):
        details["reason"] = "prediction-below-min-height"
        return None, details
    if max_height_m is not None and np.any(predictions > max_height_m):
        details["reason"] = "prediction-above-max-height"
        return None, details
    details["reason"] = "accepted"
    return predictions.astype(np.float64), details


def complete_component_aware_enclosed_holes(
    grid: HeightGrid,
    baseline: BaselineHeightMap,
    max_small_hole_cells: int,
    neighbor_radius_cells: int,
    max_neighbor_height_delta_m: float,
    baseline_fill_radius_cells: int,
    curve_max_hole_area_cm2: float,
    curve_max_component_area_ratio: float,
    curve_max_imputed_ratio: float,
    curve_rim_radius_cells: int,
    curve_min_rim_samples: int,
    curve_min_rim_coverage: float,
    curve_max_fit_rmse_m: float,
    curve_max_prediction_rise_m: float,
    min_height_m: float,
    max_height_m: float | None,
) -> tuple[HeightGrid, dict[str, object], set[tuple[int, int]]]:
    """执行组件感知的保守补洞，并返回新网格、审计统计和未填孔集合。

    小平滑孔优先用邻域加权插值；更大的反光孔可尝试局部二次曲面。两种方式都须
    满足“边界仅属于一个组件”，曲面方式还必须通过物理面积、组件面积比、边缘
    覆盖、拟合误差、预测范围和每组件推断比例上限。多物品间空隙会被拒绝而非补满。
    """
    if max_small_hole_cells < 0 or neighbor_radius_cells < 1 or max_neighbor_height_delta_m <= 0.0:
        raise ValueError("hole-fill 参数无效。")
    if (
        curve_max_hole_area_cm2 < 0.0
        or not 0.0 < curve_max_component_area_ratio <= 1.0
        or not 0.0 < curve_max_imputed_ratio <= 1.0
        or curve_rim_radius_cells < 1
        or curve_min_rim_samples < 6
        or not 0.0 < curve_min_rim_coverage <= 1.0
        or curve_max_fit_rmse_m <= 0.0
        or curve_max_prediction_rise_m <= 0.0
    ):
        raise ValueError("curve-fill 参数无效。")

    stats: dict[str, int | float | bool | str] = {
        "enabled": bool(max_small_hole_cells > 0 or curve_max_hole_area_cm2 > 0.0),
        "mode": "component-aware-small-or-quadratic",
        "candidate_hole_count": 0,
        "candidate_hole_cell_count": 0,
        "filled_hole_count": 0,
        "filled_cell_count": 0,
        "small_filled_hole_count": 0,
        "small_filled_cell_count": 0,
        "small_fallback_to_curve_hole_count": 0,
        "curve_filled_hole_count": 0,
        "curve_filled_cell_count": 0,
        "skipped_large_hole_count": 0,
        "skipped_non_smooth_hole_count": 0,
        "skipped_ambiguous_component_hole_count": 0,
        "skipped_curve_area_count": 0,
        "skipped_curve_component_area_ratio_count": 0,
        "skipped_curve_imputed_ratio_count": 0,
        "skipped_curve_insufficient_rim_count": 0,
        "skipped_curve_rim_coverage_count": 0,
        "skipped_curve_fit_count": 0,
        "skipped_curve_prediction_count": 0,
        "skipped_missing_baseline_count": 0,
        "max_hole_cells": int(max_small_hole_cells),
        "neighbor_radius_cells": int(neighbor_radius_cells),
        "max_neighbor_height_delta_m": float(max_neighbor_height_delta_m),
        "curve_max_hole_area_cm2": float(curve_max_hole_area_cm2),
        "curve_max_component_area_ratio": float(curve_max_component_area_ratio),
        "curve_max_imputed_ratio": float(curve_max_imputed_ratio),
        "curve_rim_radius_cells": int(curve_rim_radius_cells),
        "curve_min_rim_samples": int(curve_min_rim_samples),
        "curve_min_rim_coverage": float(curve_min_rim_coverage),
        "curve_max_fit_rmse_m": float(curve_max_fit_rmse_m),
        "curve_max_prediction_rise_m": float(curve_max_prediction_rise_m),
    }
    height_by_cell = {
        (int(cell[0]), int(cell[1])): float(grid.heights_m[index])
        for index, cell in enumerate(grid.cell_indices)
    }
    label_by_cell = {
        (int(cell[0]), int(cell[1])): int(grid.component_labels[index])
        for index, cell in enumerate(grid.cell_indices)
    }
    all_cells = np.asarray(grid.cell_indices, dtype=np.int32)
    cell_area_m2 = grid.cell_size_m * grid.cell_size_m
    additions: list[tuple[tuple[int, int], float, float]] = []
    addition_labels: list[int] = []
    unfilled_hole_cells: set[tuple[int, int]] = set()
    processed_holes: set[tuple[tuple[int, int], ...]] = set()
    component_summaries: list[dict[str, int | float]] = []

    # 每个组件独立找洞/补洞，正是多食材情况下避免跨物品误填的核心。
    for component_label in sorted({int(label) for label in grid.component_labels}):
        component_mask = grid.component_labels == component_label
        component_cells = all_cells[component_mask]
        blocked_cells = all_cells[~component_mask]
        component_measured_cells = int(np.count_nonzero(component_mask & ~grid.is_interpolated))
        component_existing_inferred = int(np.count_nonzero(component_mask & grid.is_interpolated))
        component_added_cells = 0
        component_hole_count = 0
        component_curve_hole_count = 0
        component_holes = find_enclosed_hole_components(component_cells, blocked_cells)
        for hole in component_holes:
            hole_key = tuple(sorted(hole))
            if hole_key in processed_holes:
                continue
            processed_holes.add(hole_key)
            component_hole_count += 1
            stats["candidate_hole_count"] = int(stats["candidate_hole_count"]) + 1
            stats["candidate_hole_cell_count"] = int(stats["candidate_hole_cell_count"]) + len(hole)

            # 孔洞边界若触及多个组件，优先视为物体间气隙：记录但永不纳入补洞体积。
            if hole_boundary_component_labels(hole, label_by_cell) != {component_label}:
                stats["skipped_ambiguous_component_hole_count"] = int(
                    stats["skipped_ambiguous_component_hole_count"]
                ) + 1
                unfilled_hole_cells.update(hole)
                continue

            local_additions: list[tuple[tuple[int, int], float, float]] | None = None
            completion_kind = ""
            small_rejected = False
            # 小孔先走低成本、低风险的局部加权插值；失败后才评估曲面拟合资格。
            if max_small_hole_cells > 0 and len(hole) <= max_small_hole_cells:
                local_additions = interpolate_small_component_hole(
                    hole=hole,
                    height_by_cell=height_by_cell,
                    label_by_cell=label_by_cell,
                    component_label=component_label,
                    neighbor_radius_cells=neighbor_radius_cells,
                    max_neighbor_height_delta_m=max_neighbor_height_delta_m,
                    baseline=baseline,
                    baseline_fill_radius_cells=baseline_fill_radius_cells,
                )
                completion_kind = "small"
                if local_additions is None:
                    small_rejected = True
            if local_additions is None:
                hole_area_cm2 = float(len(hole) * cell_area_m2 * 1e4)
                component_area_m2 = float(len(component_cells) * cell_area_m2)
                if curve_max_hole_area_cm2 <= 0.0 or hole_area_cm2 > curve_max_hole_area_cm2:
                    stats["skipped_curve_area_count"] = int(stats["skipped_curve_area_count"]) + 1
                    stats["skipped_large_hole_count"] = int(stats["skipped_large_hole_count"]) + 1
                elif len(hole) * cell_area_m2 > component_area_m2 * curve_max_component_area_ratio:
                    stats["skipped_curve_component_area_ratio_count"] = int(
                        stats["skipped_curve_component_area_ratio_count"]
                    ) + 1
                else:
                    predictions, fit_details = fit_quadratic_component_hole(
                        hole=hole,
                        height_by_cell=height_by_cell,
                        label_by_cell=label_by_cell,
                        component_label=component_label,
                        cell_size_m=grid.cell_size_m,
                        rim_radius_cells=curve_rim_radius_cells,
                        min_rim_samples=curve_min_rim_samples,
                        min_rim_coverage=curve_min_rim_coverage,
                        max_fit_rmse_m=curve_max_fit_rmse_m,
                        max_prediction_rise_m=curve_max_prediction_rise_m,
                        min_height_m=min_height_m,
                        max_height_m=max_height_m,
                    )
                    if predictions is None:
                        reason = str(fit_details["reason"])
                        if reason == "insufficient-rim-samples":
                            stats["skipped_curve_insufficient_rim_count"] = int(
                                stats["skipped_curve_insufficient_rim_count"]
                            ) + 1
                        elif reason == "insufficient-rim-coverage":
                            stats["skipped_curve_rim_coverage_count"] = int(
                                stats["skipped_curve_rim_coverage_count"]
                            ) + 1
                        elif reason.startswith("prediction-"):
                            stats["skipped_curve_prediction_count"] = int(
                                stats["skipped_curve_prediction_count"]
                            ) + 1
                        else:
                            stats["skipped_curve_fit_count"] = int(stats["skipped_curve_fit_count"]) + 1
                    else:
                        local_additions = []
                        for cell, predicted_height in zip(hole, predictions, strict=True):
                            baseline_height, _ = lookup_baseline_height(
                                cell,
                                baseline_height_by_cell=baseline.height_by_cell,
                                fill_radius_cells=baseline_fill_radius_cells,
                            )
                            if baseline_height is None:
                                local_additions = None
                                stats["skipped_missing_baseline_count"] = int(
                                    stats["skipped_missing_baseline_count"]
                                ) + 1
                                break
                            local_additions.append((cell, float(baseline_height), float(predicted_height)))
                        completion_kind = "curve"

            if local_additions is None:
                if small_rejected:
                    stats["skipped_non_smooth_hole_count"] = int(stats["skipped_non_smooth_hole_count"]) + 1
                unfilled_hole_cells.update(hole)
                continue

            # 即便每个孔都合理，也限制单一组件累计推断比例，维持 IM 结果可信度。
            projected_inferred = component_existing_inferred + component_added_cells + len(local_additions)
            projected_total = component_measured_cells + projected_inferred
            if projected_total == 0 or projected_inferred / projected_total > curve_max_imputed_ratio:
                stats["skipped_curve_imputed_ratio_count"] = int(stats["skipped_curve_imputed_ratio_count"]) + 1
                unfilled_hole_cells.update(hole)
                continue

            additions.extend(local_additions)
            addition_labels.extend([component_label] * len(local_additions))
            component_added_cells += len(local_additions)
            stats["filled_hole_count"] = int(stats["filled_hole_count"]) + 1
            stats["filled_cell_count"] = int(stats["filled_cell_count"]) + len(local_additions)
            if completion_kind == "small":
                stats["small_filled_hole_count"] = int(stats["small_filled_hole_count"]) + 1
                stats["small_filled_cell_count"] = int(stats["small_filled_cell_count"]) + len(local_additions)
            else:
                if small_rejected:
                    stats["small_fallback_to_curve_hole_count"] = int(
                        stats["small_fallback_to_curve_hole_count"]
                    ) + 1
                component_curve_hole_count += 1
                stats["curve_filled_hole_count"] = int(stats["curve_filled_hole_count"]) + 1
                stats["curve_filled_cell_count"] = int(stats["curve_filled_cell_count"]) + len(local_additions)

        inferred_ratio = float(
            (component_existing_inferred + component_added_cells)
            / max(1, component_measured_cells + component_existing_inferred + component_added_cells)
        )
        component_summaries.append(
            {
                "label": int(component_label),
                "measured_cell_count": component_measured_cells,
                "inferred_cell_count": int(component_existing_inferred + component_added_cells),
                "inferred_ratio": inferred_ratio,
                "footprint_area_cm2": float(len(component_cells) * cell_area_m2 * 1e4),
                "candidate_hole_count": component_hole_count,
                "curve_filled_hole_count": component_curve_hole_count,
            }
        )

    stats["component_count"] = int(len(component_summaries))
    stats["component_summaries"] = component_summaries
    stats["unfilled_hole_count"] = int(len(processed_holes) - int(stats["filled_hole_count"]))
    stats["unfilled_hole_cell_count"] = int(len(unfilled_hole_cells))

    if not additions:
        # 无合法补洞时保持原始实测网格；质量状态明确标记 raw-only。
        stats["quality_status"] = "raw-only"
        return grid, stats, unfilled_hole_cells
    addition_cells = np.asarray([item[0] for item in additions], dtype=np.int32)
    addition_baseline_heights = np.asarray([item[1] for item in additions], dtype=np.float64)
    addition_heights = np.asarray([item[2] for item in additions], dtype=np.float64)
    # 补洞格的颜色和 is_interpolated 标记将在 Stage 3/4 中显示为橙色。
    completed = make_height_grid(
        cell_indices=np.vstack([grid.cell_indices, addition_cells]),
        baseline_heights_m=np.concatenate([grid.baseline_heights_m, addition_baseline_heights]),
        food_heights_m=np.concatenate([grid.food_heights_m, addition_baseline_heights + addition_heights]),
        heights_m=np.concatenate([grid.heights_m, addition_heights]),
        top_points_rgb=np.vstack([grid.top_points_rgb, np.tile(np.array([[1.0, 0.55, 0.0]]), (len(additions), 1))]),
        baseline=baseline,
        missing_baseline_cell_count=grid.missing_baseline_cell_count,
        is_interpolated=np.concatenate([grid.is_interpolated, np.ones(len(additions), dtype=bool)]),
        component_labels=np.concatenate([grid.component_labels, np.asarray(addition_labels, dtype=np.int32)]),
    )
    max_component_inferred_ratio = max(
        (float(summary["inferred_ratio"]) for summary in component_summaries),
        default=0.0,
    )
    stats["max_component_inferred_ratio"] = max_component_inferred_ratio
    stats["quality_status"] = "reliable" if max_component_inferred_ratio <= 0.05 else "warning"
    return completed, stats, unfilled_hole_cells


def make_height_colors(heights_m: np.ndarray) -> np.ndarray:
    """把高度线性映射为蓝到红的 RGB；仅用于实测差值的可视化，不参与计算。"""
    if len(heights_m) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    min_height = float(np.min(heights_m))
    max_height = float(np.max(heights_m))
    if max_height - min_height < 1e-12:
        normalized = np.full(len(heights_m), 0.5, dtype=np.float64)
    else:
        normalized = (heights_m - min_height) / (max_height - min_height)

    return np.column_stack(
        [
            0.10 + 0.90 * normalized,
            0.25 + 0.55 * (1.0 - normalized),
            1.00 - 0.70 * normalized,
        ]
    ).astype(np.float64)


def make_point_cloud(points_xyz: np.ndarray, colors_rgb: np.ndarray) -> o3d.geometry.PointCloud:
    """由同长度的 NumPy 坐标/颜色数组构造 Open3D 点云。"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb.astype(np.float64))
    return pcd


def compute_mesh_volume(mesh: o3d.geometry.TriangleMesh) -> float:
    """计算闭合网格体积；Open3D 拒绝时用平移后的有向四面体和作为兼容回退。"""
    try:
        return float(mesh.get_volume())
    except RuntimeError:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        triangles = np.asarray(mesh.triangles, dtype=np.int32)
        if len(vertices) == 0 or len(triangles) == 0:
            raise

        reference = vertices.mean(axis=0)
        shifted = vertices - reference[None, :]
        triangle_vertices = shifted[triangles]
        signed_volumes = np.einsum(
            "ij,ij->i",
            triangle_vertices[:, 0, :],
            np.cross(triangle_vertices[:, 1, :], triangle_vertices[:, 2, :]),
        ) / 6.0
        volume = float(abs(np.sum(signed_volumes)))
        if not np.isfinite(volume) or volume <= 0.0:
            raise
        return volume


def build_plane_roi_lineset(
    plane_roi: PlaneRoi,
    plane_origin: np.ndarray,
    plane_u: np.ndarray,
    plane_v: np.ndarray,
    plane_n: np.ndarray,
    color_rgb: list[float],
    lift_m: float = 0.001,
) -> o3d.geometry.LineSet:
    """将平面 ROI 四条边转换为略高于基准面的线框，供 Stage 3/4 标示有效区域。"""
    corners_uv = np.array(
        [
            [plane_roi.u_min_m, plane_roi.v_min_m],
            [plane_roi.u_max_m, plane_roi.v_min_m],
            [plane_roi.u_max_m, plane_roi.v_max_m],
            [plane_roi.u_min_m, plane_roi.v_max_m],
        ],
        dtype=np.float64,
    )
    corners_xyz = (
        plane_origin[None, :]
        + corners_uv[:, 0:1] * plane_u[None, :]
        + corners_uv[:, 1:2] * plane_v[None, :]
        + lift_m * plane_n[None, :]
    )
    lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_xyz.astype(np.float64))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color_rgb, dtype=np.float64), (len(lines), 1))
    )
    return line_set


def make_centerline_lineset(base_points_xyz: np.ndarray, top_points_xyz: np.ndarray, colors_rgb: np.ndarray) -> o3d.geometry.LineSet:
    """为每个积分格构造基准点到顶部差值点的竖直连线。"""
    if len(base_points_xyz) != len(top_points_xyz):
        raise ValueError("base_points_xyz and top_points_xyz must have the same length.")

    point_pairs = np.empty((len(base_points_xyz) * 2, 3), dtype=np.float64)
    point_pairs[0::2] = base_points_xyz
    point_pairs[1::2] = top_points_xyz
    lines = np.array([[index * 2, index * 2 + 1] for index in range(len(base_points_xyz))], dtype=np.int32)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(point_pairs)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors_rgb.astype(np.float64))
    return line_set


def build_column_prism_lineset(grid: HeightGrid, colors_rgb: np.ndarray, max_cells: int) -> o3d.geometry.LineSet:
    """为 Stage 4 构造积分柱体线框；对显示单元抽样，绝不影响真实积分结果。"""
    if len(grid.heights_m) == 0:
        raise RuntimeError("没有可视化的积分单元。")

    if max_cells <= 0:
        raise ValueError("--max-visualization-cells 必须大于 0。")

    if len(grid.heights_m) > max_cells:
        # 每隔若干格渲染一个柱体，保护 Pi 桌面；体积仍使用全部格子。
        step = int(np.ceil(len(grid.heights_m) / max_cells))
        selection = np.arange(0, len(grid.heights_m), step, dtype=np.int32)
    else:
        selection = np.arange(len(grid.heights_m), dtype=np.int32)

    half_u = grid.plane_u * (grid.cell_size_m / 2.0)
    half_v = grid.plane_v * (grid.cell_size_m / 2.0)

    points: list[np.ndarray] = []
    lines: list[list[int]] = []
    line_colors: list[np.ndarray] = []

    for render_idx in selection.tolist():
        base_center = grid.base_points_xyz[render_idx]
        height = grid.heights_m[render_idx]
        color = colors_rgb[render_idx]

        base_corners = np.array(
            [
                base_center - half_u - half_v,
                base_center + half_u - half_v,
                base_center + half_u + half_v,
                base_center - half_u + half_v,
            ],
            dtype=np.float64,
        )
        top_corners = base_corners + height * grid.plane_n[None, :]

        offset = len(points)
        points.extend(np.vstack([base_corners, top_corners]))
        lines.extend(
            [
                [offset + 0, offset + 1],
                [offset + 1, offset + 2],
                [offset + 2, offset + 3],
                [offset + 3, offset + 0],
                [offset + 4, offset + 5],
                [offset + 5, offset + 6],
                [offset + 6, offset + 7],
                [offset + 7, offset + 4],
                [offset + 0, offset + 4],
                [offset + 1, offset + 5],
                [offset + 2, offset + 6],
                [offset + 3, offset + 7],
            ]
        )
        line_colors.extend([color] * 12)

    prism_lineset = o3d.geometry.LineSet()
    prism_lineset.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    prism_lineset.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    prism_lineset.colors = o3d.utility.Vector3dVector(np.asarray(line_colors, dtype=np.float64))
    return prism_lineset


def build_unfilled_hole_marker_cloud(
    unfilled_hole_cells: set[tuple[int, int]],
    grid: HeightGrid,
    baseline: BaselineHeightMap,
    fill_radius_cells: int,
    lift_m: float = 0.0015,
) -> tuple[o3d.geometry.PointCloud | None, int]:
    """构造紫色标记点显示已识别但被保守策略拒绝的组件内孔洞。

    这些点轻微抬离空炉基准面以避免 Z-fighting，且不属于 ``HeightGrid``，因此
    绝不会被纳入 IM 体积。
    """
    marker_points: list[np.ndarray] = []
    for cell_key in sorted(unfilled_hole_cells):
        baseline_height, _ = lookup_baseline_height(
            cell_key=cell_key,
            baseline_height_by_cell=baseline.height_by_cell,
            fill_radius_cells=fill_radius_cells,
        )
        if baseline_height is None:
            continue
        cell_u = (float(cell_key[0]) + 0.5) * grid.cell_size_m
        cell_v = (float(cell_key[1]) + 0.5) * grid.cell_size_m
        point_xyz = (
            grid.plane_origin
            + cell_u * grid.plane_u
            + cell_v * grid.plane_v
            + (float(baseline_height) + lift_m) * grid.plane_n
        )
        marker_points.append(point_xyz.astype(np.float64))

    if not marker_points:
        return None, 0

    points = np.asarray(marker_points, dtype=np.float64)
    colors = np.tile(UNFILLED_HOLE_COLOR, (len(points), 1))
    return make_point_cloud(points, colors), int(len(points))


def run_poc(args: argparse.Namespace) -> tuple[PcdIntegralMetrics, dict[str, object]]:
    """运行完整 IM：空炉建图、差分前景分割、逐格积分、保守补洞和四阶段回放。"""
    #benchmark_timer = BenchmarkTimer()
    np.random.seed(args.seed)
    o3d.utility.random.seed(args.seed)

    pcd_path = Path(args.pcd).expanduser().resolve()
    print("\n=== 加载 PCD ===", flush=True)
    print(f"PCD path: {pcd_path}", flush=True)
    pcd = load_point_cloud(pcd_path)
    input_points = len(pcd.points)
    print(f"初始点云数量: {input_points}", flush=True)
    has_rgb = bool(pcd.has_colors() and len(np.asarray(pcd.colors)) == input_points)
    print(f"RGB colors loaded: {'yes' if has_rgb else 'no'}", flush=True)

    # Stage 1 是原始采集质量检查点：用于人工对照 RGB/ROI 与点云视场。
    draw_stage([pcd], args.headless, auto_close_seconds=args.auto_close_seconds, **RAW_VIEW)

    print("\n=== 体素下采样 ===", flush=True)
    pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
    downsampled_points = len(pcd.points)
    print(f"体素下采样后点云数量: {downsampled_points}", flush=True)

    print("\n=== RANSAC 底面积分平面 ===", flush=True)
    plane_1, inliers_1 = pcd.segment_plane(
        distance_threshold=args.plane1_distance_threshold,
        ransac_n=3,
        num_iterations=1000,
    )
    plane_1 = np.asarray(plane_1, dtype=np.float64)
    print(f"Plane 1 (base): {format_plane(plane_1)}", flush=True)
    # 此处平面仅供去背景与对齐法向；正式积分零点来自下方空炉 heightmap。
    ground = pcd.select_by_index(inliers_1)
    apply_uniform_color(ground, [0.86, 0.22, 0.22])
    remaining = pcd.select_by_index(inliers_1, invert=True)

    plane_2: np.ndarray | None = None
    wall: o3d.geometry.PointCloud | None = None
    if args.remove_secondary_plane:
        print("\n=== RANSAC 次级背景平面（手动启用） ===", flush=True)
        plane_2, inliers_2 = remaining.segment_plane(
            distance_threshold=args.plane2_distance_threshold,
            ransac_n=3,
            num_iterations=1000,
        )
        plane_2 = np.asarray(plane_2, dtype=np.float64)
        print(f"Plane 2 (secondary): {format_plane(plane_2)}", flush=True)
        wall = remaining.select_by_index(inliers_2)
        apply_uniform_color(wall, [0.12, 0.36, 0.92])
        remaining2 = remaining.select_by_index(inliers_2, invert=True)
    else:
        print("\n=== 保留次级平面 ===", flush=True)
        print("Secondary-plane removal is disabled; planar food remains available for clustering.", flush=True)
        remaining2 = remaining

    print("\n=== 初始 DBSCAN 聚类（仅用于基线法向定向） ===", flush=True)
    labels = np.array(
        remaining2.cluster_dbscan(
            eps=args.cluster_eps,
            min_points=args.cluster_min_points,
            print_progress=False,
        )
    )
    if np.any(labels >= 0):
        initial_cluster_candidates = summarize_clusters(remaining2, labels)
        print(f"initial DBSCAN cluster count: {len(initial_cluster_candidates)}", flush=True)
        print_cluster_candidates(initial_cluster_candidates)
        orientation_candidate, _, _ = resolve_target_candidate(
            initial_cluster_candidates,
            cluster_min_points=args.cluster_min_points,
            requested_label=None,
        )
        orientation_obj = select_cluster(remaining2, labels, orientation_candidate.label)
    else:
        # 初始三维 DBSCAN 只帮助确定基准面正法向；真正前景分割在差分后完成，
        # 无初始簇时使用剩余点定向即可，不应提前失败。
        print("initial DBSCAN found no cluster; orienting baseline from all remaining points.", flush=True)
        orientation_obj = remaining2

    print("\n=== 空炉 baseline 差分积分 ===", flush=True)
    # 先建空炉 map，再以同一局部坐标框架处理食材；这是 IM 可重放、可比较的基石。
    baseline_paths = resolve_baseline_paths(args)
    baseline = build_baseline_heightmap(
        baseline_paths=baseline_paths,
        cell_size_m=args.integration_resolution_m,
        plane_distance_threshold=args.plane1_distance_threshold,
        voxel_size_m=args.voxel_size,
        max_surface_height_m=args.baseline_max_surface_height_m,
        orientation_points=np.asarray(orientation_obj.points, dtype=np.float64),
    )
    print(f"baseline frame count: {baseline.frame_count}", flush=True)
    for source_path in baseline.source_paths:
        print(f"  baseline: {source_path}", flush=True)
    print(f"baseline plane: {format_plane(baseline.plane)}", flush=True)
    print(f"baseline plane origin: {format_vector(baseline.plane_origin)}", flush=True)
    print(f"baseline plane normal: {format_vector(baseline.plane_n)}", flush=True)
    print(f"baseline cell count: {len(baseline.height_by_cell)}", flush=True)
    plane_roi = build_plane_roi_from_baseline(baseline, args.roi_border_margin_m)
    print(
        "plane ROI: "
        f"margin={plane_roi.border_margin_m:.4f} m, "
        f"u=[{plane_roi.u_min_m:.4f}, {plane_roi.u_max_m:.4f}], "
        f"v=[{plane_roi.v_min_m:.4f}, {plane_roi.v_max_m:.4f}]",
        flush=True,
    )

    # 按空炉高度差得到候选食材点，炉壁/底面和无基准区域此时已被排除。
    dense_valid_points, dense_valid_filter = build_valid_difference_point_cloud(
        obj=remaining2,
        baseline=baseline,
        min_height_m=args.min_height_m,
        max_height_m=args.max_height_m,
        fill_radius_cells=args.baseline_fill_radius_cells,
        plane_roi=plane_roi,
    )
    print(
        "dense valid diff points: "
        f"input_points={dense_valid_filter['input_points']}, "
        f"valid_points={dense_valid_filter['valid_points']}, "
        f"outside_roi={dense_valid_filter['discarded_outside_roi_points']}, "
        f"below_min_height={dense_valid_filter['rejected_below_min_height_points']}, "
        f"missing_baseline={dense_valid_filter['missing_baseline_points']}",
        flush=True,
    )

    print("\n=== 基线高度差前景对象聚类 ===", flush=True)
    # 多物品核心：在基准面二维距离中聚类，默认选中所有达标组件而非只选“中心”一个。
    foreground_labels, cluster_candidates = cluster_foreground_components(
        dense_valid_points,
        baseline=baseline,
        cluster_eps=args.foreground_cluster_eps,
        cluster_min_points=args.cluster_min_points,
    )
    print(f"foreground cluster count: {len(cluster_candidates)}", flush=True)
    print_cluster_candidates(cluster_candidates)
    selected_labels, target_mode, min_candidate_size = resolve_foreground_component_labels(
        cluster_candidates,
        cluster_min_points=args.cluster_min_points,
        requested_labels=parse_target_labels(args.target_label),
    )
    component_clouds = [select_cluster(dense_valid_points, foreground_labels, label) for label in selected_labels]
    selected_candidates = [candidate for candidate in cluster_candidates if int(candidate.label) in selected_labels]
    obj = merge_point_clouds(component_clouds)
    print(
        f"Selected food labels: {selected_labels} "
        f"(mode={target_mode}, components={len(component_clouds)}, points={len(obj.points)}, "
        f"min_candidate_size={min_candidate_size})",
        flush=True,
    )
    component_palette = ([0.92, 0.24, 0.24], [0.18, 0.66, 0.95], [0.94, 0.70, 0.16], [0.55, 0.33, 0.85])
    stage2_geometries: list[object] = [ground]
    if wall is not None:
        stage2_geometries.append(wall)
    for component_index, component in enumerate(component_clouds):
        display_component = o3d.geometry.PointCloud(component)
        color = component_palette[component_index % len(component_palette)]
        apply_uniform_color(display_component, color)
        outline = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(component.get_axis_aligned_bounding_box())
        apply_uniform_color(outline, color)
        stage2_geometries.extend([display_component, outline])
    draw_stage(
        stage2_geometries,
        args.headless,
        auto_close_seconds=args.auto_close_seconds,
        window_name="PCD-IM Stage 2 - Base Plane + All Foreground Food Components",
        **SEGMENT_VIEW,
    )

    # 将 Stage 2 前景组件标签保留到积分单元分辨率。只有孔洞边缘完全属于一个组件时
    # 才可能补全，这可阻止把多个食材之间的空气间隙误补为反光深度缺失。
    component_surface_candidates: dict[tuple[int, int], tuple[float, int]] = {}
    for component_label, component in zip(selected_labels, component_clouds, strict=True):
        component_surface_by_cell, _ = build_food_surface_map(
            obj=component,
            baseline=baseline,
            plane_roi=plane_roi,
        )
        for cell_key, payload in component_surface_by_cell.items():
            candidate_height = float(payload["height_m"])
            current = component_surface_candidates.get(cell_key)
            if current is None or candidate_height > current[0]:
                component_surface_candidates[cell_key] = (candidate_height, int(component_label))

    food_surface_by_cell, surface_filter = build_food_surface_map(obj=obj, baseline=baseline, plane_roi=plane_roi)
    component_label_by_cell = {
        cell_key: component_surface_candidates[cell_key][1]
        for cell_key in food_surface_by_cell
        if cell_key in component_surface_candidates
    }
    print(
        "food top-surface map: "
        f"input_points={surface_filter['input_points']}, "
        f"discarded_negative={surface_filter['discarded_negative_points']}, "
        f"outside_roi={surface_filter['discarded_outside_roi_points']}, "
        f"surface_cells={surface_filter['surface_cell_count']}",
        flush=True,
    )

    # 原始 grid 只由实测顶表面构成；补洞函数会返回标记为 interpolated 的新 grid。
    grid, baseline_lookup = build_height_grid_from_baseline_difference(
        food_surface_by_cell=food_surface_by_cell,
        baseline=baseline,
        min_height_m=args.min_height_m,
        max_height_m=args.max_height_m,
        fill_radius_cells=args.baseline_fill_radius_cells,
        component_label_by_cell=component_label_by_cell,
    )
    grid, hole_completion, unfilled_hole_cells = complete_component_aware_enclosed_holes(
        grid=grid,
        baseline=baseline,
        max_small_hole_cells=args.hole_fill_max_cells,
        neighbor_radius_cells=args.hole_fill_neighbor_radius_cells,
        max_neighbor_height_delta_m=args.hole_fill_max_neighbor_height_delta_m,
        baseline_fill_radius_cells=args.baseline_fill_radius_cells,
        curve_max_hole_area_cm2=args.curve_fill_max_hole_area_cm2,
        curve_max_component_area_ratio=args.curve_fill_max_component_area_ratio,
        curve_max_imputed_ratio=args.curve_fill_max_imputed_ratio,
        curve_rim_radius_cells=args.curve_fill_rim_radius_cells,
        curve_min_rim_samples=args.curve_fill_min_rim_samples,
        curve_min_rim_coverage=args.curve_fill_min_rim_coverage,
        curve_max_fit_rmse_m=args.curve_fill_max_fit_rmse_m,
        curve_max_prediction_rise_m=args.curve_fill_max_prediction_rise_m,
        min_height_m=args.min_height_m,
        max_height_m=args.max_height_m,
    )
    unfilled_hole_markers, unfilled_hole_cell_count = build_unfilled_hole_marker_cloud(
        unfilled_hole_cells=unfilled_hole_cells,
        grid=grid,
        baseline=baseline,
        fill_radius_cells=args.baseline_fill_radius_cells,
    )
    print(f"积分网格分辨率: {grid.cell_size_m:.4f} m", flush=True)
    print(f"占用网格数: {grid.occupied_cell_count}", flush=True)
    print(f"外接网格数: {grid.bbox_cell_count}", flush=True)
    print(f"Footprint 占用面积: {grid.footprint_area_m2:.6f} m^2", flush=True)
    print(f"Footprint 覆盖率: {grid.occupancy_ratio:.4f}", flush=True)
    print(
        "baseline lookup: "
        f"direct={baseline_lookup['direct']}, "
        f"neighbor-filled={baseline_lookup['neighbor-filled']}, "
        f"missing={baseline_lookup['missing']}",
        flush=True,
    )
    print(f"平均高度差: {grid.mean_height_m:.6f} m", flush=True)
    print(f"最大高度差: {grid.max_height_m:.6f} m", flush=True)
    print(f"PCD 原始积分体积: {grid.raw_volume_m3:.6f} m^3", flush=True)
    print(f"PCD 补洞后积分体积: {grid.volume_m3:.6f} m^3", flush=True)
    print(
        "hole completion: "
        f"measured={grid.measured_cell_count}, interpolated={grid.interpolated_cell_count}, "
        f"small={hole_completion['small_filled_cell_count']}, "
        f"curve={hole_completion['curve_filled_cell_count']}, "
        f"filled_cells={hole_completion['filled_cell_count']}, quality={hole_completion['quality_status']}",
        flush=True,
    )
    print(
        "IM 差值可视化/积分: "
        f"measured={grid.measured_cell_count}（蓝→红，计入原始积分）, "
        f"interpolated={grid.interpolated_cell_count}（橙色，计入补洞增量）, "
        f"unfilled_enclosed_holes={unfilled_hole_cell_count}（紫色，不计入体积）",
        flush=True,
    )

    # 回放图例：蓝→红实测差值；橙色已计入 repaired IM 的补洞；紫色仅提示未填孔。
    height_colors = make_height_colors(grid.heights_m)
    difference_colors = height_colors.copy()
    difference_colors[grid.is_interpolated] = INTERPOLATED_DIFFERENCE_COLOR
    measured_mask = ~grid.is_interpolated
    roi_outline = build_plane_roi_lineset(
        plane_roi=plane_roi,
        plane_origin=baseline.plane_origin,
        plane_u=baseline.plane_u,
        plane_v=baseline.plane_v,
        plane_n=baseline.plane_n,
        color_rgb=[0.96, 0.76, 0.12],
    )
    baseline_surface = make_point_cloud(
        grid.base_points_xyz,
        np.tile(BASELINE_DISPLAY_COLOR, (len(grid.base_points_xyz), 1)),
    )
    measured_difference_surface = make_point_cloud(
        grid.top_points_xyz[measured_mask],
        difference_colors[measured_mask],
    )
    interpolated_difference_surface = None
    if grid.interpolated_cell_count:
        interpolated_difference_surface = make_point_cloud(
            grid.top_points_xyz[grid.is_interpolated],
            difference_colors[grid.is_interpolated],
        )
    center_lines = make_centerline_lineset(grid.base_points_xyz, grid.top_points_xyz, difference_colors)

    stage3_geometries: list[o3d.geometry.Geometry] = [
        ground,
        roi_outline,
        baseline_surface,
        measured_difference_surface,
        center_lines,
    ]
    if interpolated_difference_surface is not None:
        stage3_geometries.append(interpolated_difference_surface)
    if unfilled_hole_markers is not None:
        stage3_geometries.append(unfilled_hole_markers)

    draw_stage(
        stage3_geometries,
        args.headless,
        auto_close_seconds=args.auto_close_seconds,
        window_name="PCD-IM Stage 3 - Difference Heatmap (Orange=Included Fill, Magenta=Excluded Hole)",
        point_size=5.0,
        line_width=2.0,
        **VOLUME_VIEW,
    )

    print("\n=== 参考体积对比 ===", flush=True)
    # AABB/OBB/凸包为参考几何，不能替代基准面差值积分结果。
    analysis_obj = obj
    aabb = analysis_obj.get_axis_aligned_bounding_box()
    obb_compact = compute_compact_minimum_volume_obb(
        analysis_obj,
        max_faces=args.max_obb_faces,
        seed=args.seed,
    )
    hull_mesh, _ = analysis_obj.compute_convex_hull(joggle_inputs=True)

    hull_volume_m3 = compute_mesh_volume(hull_mesh)
    print(f"AABB 体积: {aabb.volume():.6f} m^3", flush=True)
    print(f"OBB-Compact 体积: {obb_compact.volume:.6f} m^3", flush=True)
    print(f"Convex Hull 体积: {hull_volume_m3:.6f} m^3", flush=True)

    prism_lineset = build_column_prism_lineset(
        grid=grid,
        colors_rgb=difference_colors,
        max_cells=args.max_visualization_cells,
    )
    stage4_geometries: list[o3d.geometry.Geometry] = [
        ground,
        roi_outline,
        baseline_surface,
        measured_difference_surface,
        center_lines,
        prism_lineset,
    ]
    if interpolated_difference_surface is not None:
        stage4_geometries.append(interpolated_difference_surface)
    if unfilled_hole_markers is not None:
        stage4_geometries.append(unfilled_hole_markers)
    draw_stage(
        stage4_geometries,
        args.headless,
        auto_close_seconds=args.auto_close_seconds,
        window_name="PCD-IM Stage 4 - Difference Integral (Orange=Included Fill, Magenta=Excluded Hole)",
        point_size=5.0,
        line_width=2.0,
        **VOLUME_VIEW,
    )

    #benchmark = benchmark_timer.finish()
    metrics = PcdIntegralMetrics(
        input_points=input_points,
        downsampled_points=downsampled_points,
        cluster_count=len(cluster_candidates),
        selected_cluster_labels=selected_labels,
        selected_cluster_points=len(obj.points),
        component_count=len(component_clouds),
        top_surface_points=int(surface_filter["surface_cell_count"]),
        baseline_frame_count=baseline.frame_count,
        plane_1=plane_1,
        plane_2=plane_2,
        occupied_cell_count=grid.occupied_cell_count,
        footprint_bbox_cell_count=grid.bbox_cell_count,
        footprint_area_m2=grid.footprint_area_m2,
        mean_height_m=grid.mean_height_m,
        max_height_m=grid.max_height_m,
        integral_volume_m3=grid.volume_m3,
        raw_integral_volume_m3=grid.raw_volume_m3,
        aabb_volume_m3=float(aabb.volume()),
        obb_compact_volume_m3=float(obb_compact.volume),
        convex_hull_volume_m3=hull_volume_m3,
        missing_baseline_cell_count=grid.missing_baseline_cell_count,
        #benchmark=benchmark,
    )
    # JSON 同时输出输入、筛选、补洞和颜色语义，GUI 无需解析终端日志即可审计结果。
    payload = {
        "pcd_path": str(pcd_path),
        "pcd_has_rgb": has_rgb,
        "voxel_size_m": args.voxel_size,
        "plane_1": plane_1.tolist(),
        "plane_2": plane_2.tolist() if plane_2 is not None else None,
        "secondary_plane_removed": bool(args.remove_secondary_plane),
        "cluster_count": len(cluster_candidates),
        "selected_cluster_labels": selected_labels,
        "target_selection_mode": target_mode,
        "target_min_candidate_size": min_candidate_size,
        "selected_clusters": [candidate.to_dict() for candidate in selected_candidates],
        "cluster_candidates": [candidate.to_dict() for candidate in cluster_candidates],
        "baseline": {
            "source_paths": baseline.source_paths,
            "frame_count": baseline.frame_count,
            "plane": baseline.plane.tolist(),
            "plane_origin_m": baseline.plane_origin.tolist(),
            "plane_axes": {
                "u": baseline.plane_u.tolist(),
                "v": baseline.plane_v.tolist(),
                "n": baseline.plane_n.tolist(),
            },
            "cell_size_m": baseline.cell_size_m,
            "cell_count": len(baseline.height_by_cell),
            "max_surface_height_m": args.baseline_max_surface_height_m,
        },
        "plane_roi": {
            "bbox_u_min_m": plane_roi.bbox_u_min_m,
            "bbox_u_max_m": plane_roi.bbox_u_max_m,
            "bbox_v_min_m": plane_roi.bbox_v_min_m,
            "bbox_v_max_m": plane_roi.bbox_v_max_m,
            "u_min_m": plane_roi.u_min_m,
            "u_max_m": plane_roi.u_max_m,
            "v_min_m": plane_roi.v_min_m,
            "v_max_m": plane_roi.v_max_m,
            "border_margin_m": plane_roi.border_margin_m,
        },
        "surface_filter": surface_filter,
        "valid_point_filter": dense_valid_filter,
        "baseline_lookup": baseline_lookup,
        "height_grid": {
            "integration_resolution_m": grid.cell_size_m,
            "plane_origin_m": grid.plane_origin.tolist(),
            "plane_axes": {
                "u": grid.plane_u.tolist(),
                "v": grid.plane_v.tolist(),
                "n": grid.plane_n.tolist(),
            },
            "occupied_cell_count": grid.occupied_cell_count,
            "bbox_cell_count": grid.bbox_cell_count,
            "footprint_area_m2": grid.footprint_area_m2,
            "bbox_area_m2": grid.bbox_area_m2,
            "occupancy_ratio": grid.occupancy_ratio,
            "mean_height_m": grid.mean_height_m,
            "max_height_m": grid.max_height_m,
            "volume_m3": grid.volume_m3,
            "raw_volume_m3": grid.raw_volume_m3,
            "interpolated_volume_m3": grid.interpolated_volume_m3,
            "measured_cell_count": grid.measured_cell_count,
            "interpolated_cell_count": grid.interpolated_cell_count,
            "measured_coverage_ratio": float(grid.measured_cell_count / grid.occupied_cell_count),
            "missing_baseline_cell_count": grid.missing_baseline_cell_count,
        },
        "difference_visualization": {
            "measured_difference_cell_count": grid.measured_cell_count,
            "interpolated_difference_cell_count": grid.interpolated_cell_count,
            "unfilled_enclosed_hole_cell_count": unfilled_hole_cell_count,
            "legend": {
                "blue_to_red": "measured baseline-height difference; included in raw and repaired IM volume",
                "orange": "component-owned small or quadratic-surface fill; included only in repaired IM volume",
                "magenta": "component-owned enclosed depth hole left unfilled; excluded from IM volume",
                "green": "empty-oven baseline point",
            },
        },
        "hole_completion": hole_completion,
        #"benchmark": benchmark.to_dict(),
    }
    return metrics, payload


def write_result_json(path: str | Path, metrics: PcdIntegralMetrics, payload: dict[str, object]) -> Path:
    """将 IM 结果及补洞可追溯信息写为 UTF-8 JSON，供 GUI 状态栏和回放复用。"""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "pcd_path": payload["pcd_path"],
        "pcd_has_rgb": payload["pcd_has_rgb"],
        "input_points": metrics.input_points,
        "downsampled_points": metrics.downsampled_points,
        "cluster_count": metrics.cluster_count,
        "baseline": payload["baseline"],
        "plane_roi": payload["plane_roi"],
        "plane_1": payload["plane_1"],
        "plane_2": payload["plane_2"],
        "secondary_plane_removed": payload["secondary_plane_removed"],
        "target": {
            "mode": payload["target_selection_mode"],
            "labels": metrics.selected_cluster_labels,
            "label": metrics.selected_cluster_labels[0],
            "point_count": metrics.selected_cluster_points,
            "component_count": metrics.component_count,
            "top_surface_points": metrics.top_surface_points,
            "baseline_frame_count": metrics.baseline_frame_count,
            "min_candidate_size": payload["target_min_candidate_size"],
            "summaries": payload["selected_clusters"],
        },
        "cluster_candidates": payload["cluster_candidates"],
        "surface_filter": payload["surface_filter"],
        "valid_point_filter": payload["valid_point_filter"],
        "baseline_lookup": payload["baseline_lookup"],
        "height_grid": payload["height_grid"],
        "difference_visualization": payload["difference_visualization"],
        "hole_completion": payload["hole_completion"],
        #"benchmark": payload["benchmark"],
        "volumes_m3": {
            "pcd_integral": metrics.integral_volume_m3,
            "pcd_integral_raw": metrics.raw_integral_volume_m3,
            "aabb": metrics.aabb_volume_m3,
            "obb_compact": metrics.obb_compact_volume_m3,
            "convex_hull": metrics.convex_hull_volume_m3,
        },
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    """CLI 入口：失败返回非零退出码，成功打印实测/补洞体积及 Benchmark。"""
    args = parse_args()
    try:
        metrics, payload = run_poc(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.result_json:
        result_path = write_result_json(args.result_json, metrics, payload)
        print(f"Saved result JSON: {result_path}", flush=True)

    print("\n=== PCD-IM 复现完成 ===", flush=True)
    print(f"输入点数: {metrics.input_points}", flush=True)
    print(f"下采样点数: {metrics.downsampled_points}", flush=True)
    print(f"聚类数: {metrics.cluster_count}", flush=True)
    print(f"目标簇标签: {metrics.selected_cluster_labels}", flush=True)
    print(f"目标组件数: {metrics.component_count}", flush=True)
    print(f"目标簇点数: {metrics.selected_cluster_points}", flush=True)
    print(f"顶部 surface cell 数: {metrics.top_surface_points}", flush=True)
    print(f"baseline 帧数: {metrics.baseline_frame_count}", flush=True)
    print(f"占用网格数: {metrics.occupied_cell_count}", flush=True)
    print(f"Footprint 占用面积: {metrics.footprint_area_m2:.6f} m^2", flush=True)
    print(f"平均高度: {metrics.mean_height_m:.6f} m", flush=True)
    print(f"最大高度: {metrics.max_height_m:.6f} m", flush=True)
    print(f"PCD 原始积分体积: {metrics.raw_integral_volume_m3:.6f} m^3", flush=True)
    print(f"PCD 补洞后积分体积: {metrics.integral_volume_m3:.6f} m^3", flush=True)
    print(f"AABB 体积: {metrics.aabb_volume_m3:.6f} m^3", flush=True)
    print(f"OBB-Compact 体积: {metrics.obb_compact_volume_m3:.6f} m^3", flush=True)
    print(f"Convex Hull 体积: {metrics.convex_hull_volume_m3:.6f} m^3", flush=True)
    print(f"未匹配 baseline cell 数: {metrics.missing_baseline_cell_count}", flush=True)
    #print_benchmark("PCD-IM", metrics.benchmark)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
