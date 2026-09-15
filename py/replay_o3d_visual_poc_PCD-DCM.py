"""PCD-DCM（深度柱映射）体积计算与四阶段回放。

算法以空炉基准面为局部坐标系：把每个食物表面点投影到同一 ``(u, v)`` 栅格，
向基准面查找对应高度，再形成“基准点—食物顶点”的深度柱。正式 DCM 体积对每个
前景组件分别对柱体上下端点求凸包后求和，避免多个分离物品被一个整体凸包桥接。
本文件复用 IM 的基线、高度图和补洞工具以保证两条算法的输入口径一致。单位为米。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# 树莓派上的小规模点云无需多线程 BLAS；限制线程使 Benchmark 可重复且避免抢占 GUI。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from benchmark_metrics import BenchmarkMetrics, BenchmarkTimer, print_benchmark


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
DEFAULT_PCD_FALLBACK = REPO_ROOT / "ZC_O3D_poc" / "cloud_bin_0.pcd"
DEFAULT_BASELINE_STATE = REPO_ROOT / "captures" / "pcd_i_empty_baseline.json"
COMPAT_DIR = REPO_ROOT / "compat"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(COMPAT_DIR) not in sys.path:
    sys.path.insert(0, str(COMPAT_DIR))

import open3d as o3d

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


def load_pcd_im_module():
    """按文件路径加载 IM 公共工具，避免连字符文件名无法用普通 import 导入。"""
    module_path = Path(__file__).resolve().with_name("replay_o3d_visual_poc_PCD-IM.py")
    if str(module_path.parent) not in sys.path:
        sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("replay_o3d_visual_poc_pcd_im_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 PCD-IM 依赖模块: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# DCM 的基线栅格、前景组件与补洞规则刻意直接复用 IM，防止两算法比较时口径不同。
PCD_IM = load_pcd_im_module()

resolve_baseline_paths = PCD_IM.resolve_baseline_paths
resolve_path_from_repo = PCD_IM.resolve_path_from_repo
build_baseline_heightmap = PCD_IM.build_baseline_heightmap
extract_points_and_colors = PCD_IM.extract_points_and_colors
project_points_to_plane_frame = PCD_IM.project_points_to_plane_frame
uv_to_cell_indices = PCD_IM.uv_to_cell_indices
lookup_baseline_height = PCD_IM.lookup_baseline_height
build_food_surface_map = PCD_IM.build_food_surface_map
build_height_grid_from_baseline_difference = PCD_IM.build_height_grid_from_baseline_difference
build_plane_roi_from_baseline = PCD_IM.build_plane_roi_from_baseline
build_plane_roi_lineset = PCD_IM.build_plane_roi_lineset
build_valid_difference_point_cloud = PCD_IM.build_valid_difference_point_cloud
cluster_foreground_components = PCD_IM.cluster_foreground_components
resolve_foreground_component_labels = PCD_IM.resolve_foreground_component_labels
parse_target_labels = PCD_IM.parse_target_labels
merge_point_clouds = PCD_IM.merge_point_clouds
complete_component_aware_enclosed_holes = PCD_IM.complete_component_aware_enclosed_holes
compute_mesh_volume = PCD_IM.compute_mesh_volume
make_height_colors = PCD_IM.make_height_colors
make_point_cloud = PCD_IM.make_point_cloud
PlaneRoi = PCD_IM.PlaneRoi


BASE_COLOR = np.array([0.08, 0.82, 0.28], dtype=np.float64)
COMPONENT_COLORS = (
    np.array([0.92, 0.24, 0.24], dtype=np.float64),
    np.array([0.18, 0.66, 0.95], dtype=np.float64),
    np.array([0.94, 0.70, 0.16], dtype=np.float64),
    np.array([0.55, 0.33, 0.85], dtype=np.float64),
)


@dataclass
class DepthColumnMapping:
    """有效深度柱的上下端点、颜色、高度和所在基准面栅格索引。"""
    top_points_xyz: np.ndarray
    top_points_rgb: np.ndarray
    base_points_xyz: np.ndarray
    base_points_rgb: np.ndarray
    heights_m: np.ndarray
    cell_indices: np.ndarray
    base_lookup_histogram: dict[str, int]


@dataclass
class DepthColumnConvexMetrics:
    """一次 DCM 运行的核心计量、参考值和性能统计，最终写入 result JSON。"""
    input_points: int
    downsampled_points: int
    cluster_count: int
    selected_cluster_labels: list[int]
    selected_cluster_points: int
    component_count: int
    valid_point_count: int
    mapped_base_point_count: int
    dcm_hull_input_point_count: int
    dcm_hull_vertex_count: int
    dcm_hull_triangle_count: int
    dcm_hull_volume_m3: float
    integral_volume_m3: float
    raw_integral_volume_m3: float
    reference_aabb_volume_m3: float
    reference_obb_compact_volume_m3: float
    reference_convex_hull_volume_m3: float
    footprint_area_m2: float
    mean_height_m: float
    max_height_m: float
    missing_baseline_point_count: int
    benchmark: BenchmarkMetrics


def parse_args() -> argparse.Namespace:
    """声明 DCM 命令行参数；与 IM 共享的阈值保持相同含义和单位。"""
    parser = argparse.ArgumentParser(
        description="Replay a depth-column mapping convex-hull POC from a D405 PCD capture."
    )
    parser.add_argument(
        "--pcd",
        default=None,
        help="Point-cloud input used for the PCD-DCM POC. Omit it to auto-pick the latest capture.",
    )
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
        default=None,
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
        help="Base-plane raster cell size used for baseline lookup and reference integration.",
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
        help="Maximum size of one enclosed IM reference-grid hole to interpolate; 0 disables it.",
    )
    parser.add_argument(
        "--hole-fill-neighbor-radius-cells",
        type=int,
        default=2,
        help="Chebyshev neighbor radius used for local IM reference-grid hole interpolation.",
    )
    parser.add_argument(
        "--hole-fill-max-neighbor-height-delta-m",
        type=float,
        default=0.012,
        help="Reject a reference-grid hole when local measured-neighbor height range exceeds this value.",
    )
    parser.add_argument("--curve-fill-max-hole-area-cm2", type=float, default=8.0)
    parser.add_argument("--curve-fill-max-component-area-ratio", type=float, default=0.30)
    parser.add_argument("--curve-fill-max-imputed-ratio", type=float, default=0.25)
    parser.add_argument("--curve-fill-rim-radius-cells", type=int, default=4)
    parser.add_argument("--curve-fill-min-rim-samples", type=int, default=16)
    parser.add_argument("--curve-fill-min-rim-coverage", type=float, default=0.75)
    parser.add_argument("--curve-fill-max-fit-rmse-m", type=float, default=0.004)
    parser.add_argument("--curve-fill-max-prediction-rise-m", type=float, default=0.015)
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
        "--max-visualization-pairs",
        type=int,
        default=2000,
        help="Cap the number of depth-to-base line pairs rendered in Stage 3.",
    )
    parser.add_argument(
        "--result-json",
        default=None,
        help="Optional JSON path for machine-readable validation results.",
    )
    return parser.parse_args()


def find_latest_capture() -> Path | None:
    """在 captures 中按修改时间查找当前相机序列号对应的最近 PCD。"""
    capture_dir = REPO_ROOT / "captures"
    if not capture_dir.exists():
        return None

    latest_files = sorted(
        capture_dir.glob("d405_260322274982_*.pcd"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not latest_files:
        return None
    return latest_files[0].resolve()


def resolve_input_pcd(raw_pcd: str | None) -> tuple[Path, str]:
    """按显式路径、最近采集、仓库样例的优先级解析输入，并返回来源标识。"""
    if raw_pcd:
        return resolve_path_from_repo(raw_pcd), "explicit"

    latest_capture = find_latest_capture()
    if latest_capture is not None:
        return latest_capture, "latest-capture"

    return DEFAULT_PCD_FALLBACK.resolve(), "fallback-sample"


def build_depth_column_mapping(
    obj: o3d.geometry.PointCloud,
    baseline,
    min_height_m: float,
    max_height_m: float | None,
    fill_radius_cells: int,
    plane_roi: PlaneRoi | None = None,
) -> tuple[DepthColumnMapping, dict[str, int]]:
    """将对象点映射为深度柱，并按高度、ROI 和空炉栅格可用性过滤。

    每个输入点首先投影到空炉基准面坐标系，随后在相同栅格查找空炉高度。只有
    ``food_height - baseline_height`` 位于有效高度范围的点才保留；对应的基准点和
    食物顶点组成一根柱。返回的统计值能区分 ROI 剔除、阈值剔除和基线缺失。
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

    top_points_xyz: list[np.ndarray] = []
    top_points_rgb: list[np.ndarray] = []
    base_points_xyz: list[np.ndarray] = []
    base_points_rgb: list[np.ndarray] = []
    kept_heights_m: list[float] = []
    kept_cell_indices: list[np.ndarray] = []
    stats = {
        "input_points": int(len(points)),
        "valid_points": 0,
        "discarded_negative_points": 0,
        "discarded_outside_roi_points": 0,
        "rejected_below_min_height_points": 0,
        "missing_baseline_points": 0,
        "baseline_direct_points": 0,
        "baseline_neighbor_filled_points": 0,
    }

    # 逐点而不是仅按单元保留：DCM 凸包需要表面点和其基准面投影点形成完整柱体边界。
    for cell_index, point_xyz, color_rgb, point_uv, food_height in zip(cells, points, colors, uv, heights, strict=False):
        if not np.isfinite(food_height) or food_height < -0.003:
            stats["discarded_negative_points"] += 1
            continue

        cell_key = (int(cell_index[0]), int(cell_index[1]))
        if plane_roi is not None:
            cell_center_u = (cell_key[0] + 0.5) * baseline.cell_size_m
            cell_center_v = (cell_key[1] + 0.5) * baseline.cell_size_m
            if not (
                plane_roi.u_min_m <= cell_center_u <= plane_roi.u_max_m
                and plane_roi.v_min_m <= cell_center_v <= plane_roi.v_max_m
            ):
                stats["discarded_outside_roi_points"] += 1
                continue

        # 空炉图直接命中优先；允许有限邻域插补以容忍基线采集中的稀疏深度缺失。
        baseline_height, lookup_mode = lookup_baseline_height(
            cell_key,
            baseline_height_by_cell=baseline.height_by_cell,
            fill_radius_cells=fill_radius_cells,
        )
        if baseline_height is None:
            stats["missing_baseline_points"] += 1
            continue

        if lookup_mode == "direct":
            stats["baseline_direct_points"] += 1
        else:
            stats["baseline_neighbor_filled_points"] += 1

        diff_height = float(food_height) - baseline_height
        if diff_height < min_height_m:
            stats["rejected_below_min_height_points"] += 1
            continue
        if max_height_m is not None:
            diff_height = min(diff_height, max_height_m)

        # 用空炉高度重建同一 (u,v) 的基准点；它不是简单的平面投影，包含炉腔底面形变。
        base_point_xyz = (
            baseline.plane_origin
            + point_uv[0] * baseline.plane_u
            + point_uv[1] * baseline.plane_v
            + baseline_height * baseline.plane_n
        )

        top_points_xyz.append(point_xyz.astype(np.float64))
        top_points_rgb.append(color_rgb.astype(np.float64))
        base_points_xyz.append(base_point_xyz.astype(np.float64))
        base_points_rgb.append(BASE_COLOR.copy())
        kept_heights_m.append(float(diff_height))
        kept_cell_indices.append(np.array([cell_key[0], cell_key[1]], dtype=np.int32))
        stats["valid_points"] += 1

    if not top_points_xyz:
        raise RuntimeError("PCD-DCM 没有生成任何有效深度柱。请检查空炉 baseline 或高度阈值。")

    mapping = DepthColumnMapping(
        top_points_xyz=np.asarray(top_points_xyz, dtype=np.float64),
        top_points_rgb=np.asarray(top_points_rgb, dtype=np.float64),
        base_points_xyz=np.asarray(base_points_xyz, dtype=np.float64),
        base_points_rgb=np.asarray(base_points_rgb, dtype=np.float64),
        heights_m=np.asarray(kept_heights_m, dtype=np.float64),
        cell_indices=np.asarray(kept_cell_indices, dtype=np.int32),
        base_lookup_histogram={
            "direct": int(stats["baseline_direct_points"]),
            "neighbor-filled": int(stats["baseline_neighbor_filled_points"]),
            "missing": int(stats["missing_baseline_points"]),
        },
    )
    return mapping, stats


def make_wireframe_from_mesh(mesh: o3d.geometry.TriangleMesh, color_rgb: np.ndarray) -> o3d.geometry.LineSet:
    """将组件凸包三角网格转为同色线框，便于 Stage 4 观察而不遮挡点云。"""
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    apply_uniform_color(wireframe, color_rgb.tolist())
    return wireframe


def build_projection_lineset(mapping: DepthColumnMapping, max_pairs: int) -> o3d.geometry.LineSet:
    """构造 Stage 3 深度柱连线；对大量点按步长抽样，仅影响显示不影响体积。"""
    if max_pairs <= 0:
        raise ValueError("--max-visualization-pairs 必须大于 0。")

    pair_count = len(mapping.top_points_xyz)
    if pair_count > max_pairs:
        step = int(np.ceil(pair_count / max_pairs))
        selection = np.arange(0, pair_count, step, dtype=np.int32)
    else:
        selection = np.arange(pair_count, dtype=np.int32)

    point_pairs = np.empty((len(selection) * 2, 3), dtype=np.float64)
    point_pairs[0::2] = mapping.base_points_xyz[selection]
    point_pairs[1::2] = mapping.top_points_xyz[selection]
    lines = np.array([[index * 2, index * 2 + 1] for index in range(len(selection))], dtype=np.int32)
    colors = make_height_colors(mapping.heights_m[selection])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(point_pairs)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return line_set


def build_dcm_hull_input_cloud(mapping: DepthColumnMapping) -> o3d.geometry.PointCloud:
    """合并深度柱所有上下端点，作为单一组件 DCM 凸包的输入边界点集。"""
    merged_points_xyz = np.vstack([mapping.top_points_xyz, mapping.base_points_xyz])
    merged_points_rgb = np.vstack([mapping.top_points_rgb, mapping.base_points_rgb])
    return make_point_cloud(merged_points_xyz, merged_points_rgb)


def run_poc(args: argparse.Namespace) -> tuple[DepthColumnConvexMetrics, dict[str, object]]:
    """执行完整 DCM 流程：预处理、基线差分分割、组件凸包、参考积分和四阶段渲染。"""
    benchmark_timer = BenchmarkTimer()
    np.random.seed(args.seed)
    o3d.utility.random.seed(args.seed)

    pcd_path, pcd_source = resolve_input_pcd(args.pcd)
    print("\n=== 加载 PCD ===", flush=True)
    print(f"PCD path: {pcd_path}", flush=True)
    print(f"PCD source mode: {pcd_source}", flush=True)
    pcd = load_point_cloud(pcd_path)
    input_points = len(pcd.points)
    print(f"初始点云数量: {input_points}", flush=True)
    has_rgb = bool(pcd.has_colors() and len(np.asarray(pcd.colors)) == input_points)
    print(f"RGB colors loaded: {'yes' if has_rgb else 'no'}", flush=True)

    # Stage 1 保留原始 PCD，供人工检查 ROI、相机视场和深度采集质量。
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
    # 此平面用于早期去背景和后续基准面法向定向；最终高度零点来自空炉 baseline。
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
        # 初始三维 DBSCAN 仅帮助确定平面法向正方向；真正食物分割在基线差分后进行，
        # 因此这里无簇不应提前报错。
        print("initial DBSCAN found no cluster; orienting baseline from all remaining points.", flush=True)
        orientation_obj = remaining2

    print("\n=== 空炉 baseline 深度柱映射 ===", flush=True)
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
    stage2_geometries: list[object] = [ground]
    if wall is not None:
        stage2_geometries.append(wall)
    for component_index, component in enumerate(component_clouds):
        display_component = o3d.geometry.PointCloud(component)
        color = COMPONENT_COLORS[component_index % len(COMPONENT_COLORS)]
        apply_uniform_color(display_component, color.tolist())
        outline = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(component.get_axis_aligned_bounding_box())
        apply_uniform_color(outline, color.tolist())
        stage2_geometries.extend([display_component, outline])
    draw_stage(
        stage2_geometries,
        args.headless,
        auto_close_seconds=args.auto_close_seconds,
        window_name="PCD-DCM Stage 2 - Base Plane + All Foreground Food Components",
        **SEGMENT_VIEW,
    )
    # 同一栅格若有多组件候选，取最高表面并记录其组件归属，用于后续组件感知补洞。
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
    grid, baseline_lookup = build_height_grid_from_baseline_difference(
        food_surface_by_cell=food_surface_by_cell,
        baseline=baseline,
        min_height_m=args.min_height_m,
        max_height_m=args.max_height_m,
        fill_radius_cells=args.baseline_fill_radius_cells,
        component_label_by_cell=component_label_by_cell,
    )
    # 参考积分网格和 DCM 共用该补洞结果，仅作差分/对照；DCM 正式体积仍由下方柱体凸包计算。
    grid, hole_completion, _ = complete_component_aware_enclosed_holes(
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
    print(
        "reference integral grid: "
        f"surface_cells={surface_filter['surface_cell_count']}, "
        f"occupied_cells={grid.occupied_cell_count}, "
        f"footprint_area={grid.footprint_area_m2:.6f} m^2, "
        f"mean_height={grid.mean_height_m:.6f} m",
        flush=True,
    )
    print(
        "reference baseline lookup: "
        f"direct={baseline_lookup['direct']}, "
        f"neighbor-filled={baseline_lookup['neighbor-filled']}, "
        f"missing={baseline_lookup['missing']}",
        flush=True,
    )

    # Stage 3 所见的每一根连线都来自同一批有效深度柱，故可直接核对 DCM 的过滤结果。
    mapping, mapping_stats = build_depth_column_mapping(
        obj=obj,
        baseline=baseline,
        min_height_m=args.min_height_m,
        max_height_m=args.max_height_m,
        fill_radius_cells=args.baseline_fill_radius_cells,
        plane_roi=plane_roi,
    )
    print(
        "depth column mapping: "
        f"input_points={mapping_stats['input_points']}, "
        f"valid_points={mapping_stats['valid_points']}, "
        f"outside_roi={mapping_stats['discarded_outside_roi_points']}, "
        f"below_min_height={mapping_stats['rejected_below_min_height_points']}, "
        f"missing_baseline={mapping_stats['missing_baseline_points']}",
        flush=True,
    )
    print(
        "depth column lookup: "
        f"direct={mapping.base_lookup_histogram['direct']}, "
        f"neighbor-filled={mapping.base_lookup_histogram['neighbor-filled']}, "
        f"missing={mapping.base_lookup_histogram['missing']}",
        flush=True,
    )
    print(f"depth column mean height: {float(np.mean(mapping.heights_m)):.6f} m", flush=True)
    print(f"depth column max height: {float(np.max(mapping.heights_m)):.6f} m", flush=True)

    roi_outline = build_plane_roi_lineset(
        plane_roi=plane_roi,
        plane_origin=baseline.plane_origin,
        plane_u=baseline.plane_u,
        plane_v=baseline.plane_v,
        plane_n=baseline.plane_n,
        color_rgb=[0.96, 0.76, 0.12],
    )
    top_points_cloud = make_point_cloud(mapping.top_points_xyz, mapping.top_points_rgb)
    base_points_cloud = make_point_cloud(mapping.base_points_xyz, mapping.base_points_rgb)
    projection_lines = build_projection_lineset(mapping, max_pairs=args.max_visualization_pairs)

    draw_stage(
        [ground, roi_outline, base_points_cloud, top_points_cloud, projection_lines],
        args.headless,
        auto_close_seconds=args.auto_close_seconds,
        window_name="PCD-DCM Stage 3 - Valid Depth To Baseline Mapping",
        **VOLUME_VIEW,
    )

    print("\n=== 分对象 DCM 凸包体积估算 ===", flush=True)
    component_hull_meshes: list[o3d.geometry.TriangleMesh] = []
    component_hull_wireframes: list[o3d.geometry.LineSet] = []
    component_dcm_summaries: list[dict[str, int | float]] = []
    dcm_hull_input_point_count = 0
    dcm_hull_vertex_count = 0
    dcm_hull_triangle_count = 0
    dcm_hull_volume_m3 = 0.0
    # 关键的多物品处理：按前景组件分别建凸包、分别求体积，最后相加，避免整体凸包填平物品间隙。
    for component_index, (label, component) in enumerate(zip(selected_labels, component_clouds, strict=True)):
        component_mapping, component_mapping_stats = build_depth_column_mapping(
            obj=component,
            baseline=baseline,
            min_height_m=args.min_height_m,
            max_height_m=args.max_height_m,
            fill_radius_cells=args.baseline_fill_radius_cells,
            plane_roi=plane_roi,
        )
        component_hull_input = build_dcm_hull_input_cloud(component_mapping)
        component_mesh, _ = component_hull_input.compute_convex_hull(joggle_inputs=True)
        component_color = COMPONENT_COLORS[component_index % len(COMPONENT_COLORS)]
        component_mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.tile(component_color, (len(component_mesh.vertices), 1))
        )
        component_volume = compute_mesh_volume(component_mesh)
        component_hull_meshes.append(component_mesh)
        component_hull_wireframes.append(make_wireframe_from_mesh(component_mesh, component_color))
        dcm_hull_input_point_count += int(len(component_hull_input.points))
        dcm_hull_vertex_count += int(len(component_mesh.vertices))
        dcm_hull_triangle_count += int(len(component_mesh.triangles))
        dcm_hull_volume_m3 += component_volume
        component_dcm_summaries.append(
            {
                "label": int(label),
                "point_count": int(len(component.points)),
                "valid_depth_point_count": int(len(component_mapping.top_points_xyz)),
                "hull_input_point_count": int(len(component_hull_input.points)),
                "hull_vertex_count": int(len(component_mesh.vertices)),
                "hull_triangle_count": int(len(component_mesh.triangles)),
                "dcm_hull_volume_m3": component_volume,
                "missing_baseline_point_count": int(component_mapping_stats["missing_baseline_points"]),
            }
        )

    # AABB/OBB/整体凸包仅保留作几何参考，不能替代按组件求和的 DCM 正式结果。
    reference_obj = obj
    reference_aabb = reference_obj.get_axis_aligned_bounding_box()
    reference_obb_compact = compute_compact_minimum_volume_obb(
        reference_obj,
        max_faces=args.max_obb_faces,
        seed=args.seed,
    )
    reference_hull_mesh, _ = reference_obj.compute_convex_hull(joggle_inputs=True)

    reference_hull_volume_m3 = compute_mesh_volume(reference_hull_mesh)
    print(f"DCM component count: {len(component_hull_meshes)}", flush=True)
    print(f"DCM hull input points (sum): {dcm_hull_input_point_count}", flush=True)
    print(f"DCM hull vertices (sum): {dcm_hull_vertex_count}", flush=True)
    print(f"DCM hull triangles (sum): {dcm_hull_triangle_count}", flush=True)
    print(f"DCM hull volume (sum): {dcm_hull_volume_m3:.6f} m^3", flush=True)
    for component_summary in component_dcm_summaries:
        print(
            "DCM component: "
            f"label={component_summary['label']}, "
            f"points={component_summary['point_count']}, "
            f"mapped={component_summary['valid_depth_point_count']}, "
            f"hull_volume={component_summary['dcm_hull_volume_m3']:.6f} m^3",
            flush=True,
        )
    print(f"Reference raw integral volume: {grid.raw_volume_m3:.6f} m^3", flush=True)
    print(f"Reference repaired integral volume: {grid.volume_m3:.6f} m^3", flush=True)
    print(
        "hole completion: "
        f"measured={grid.measured_cell_count}, interpolated={grid.interpolated_cell_count}, "
        f"filled_cells={hole_completion['filled_cell_count']}, quality={hole_completion['quality_status']}",
        flush=True,
    )
    print(f"Reference AABB volume: {reference_aabb.volume():.6f} m^3", flush=True)
    print(f"Reference OBB-Compact volume: {reference_obb_compact.volume:.6f} m^3", flush=True)
    print(f"Reference convex hull volume: {reference_hull_volume_m3:.6f} m^3", flush=True)

    draw_stage(
        [
            roi_outline,
            top_points_cloud,
            base_points_cloud,
            *component_hull_meshes,
            *component_hull_wireframes,
        ],
        args.headless,
        auto_close_seconds=args.auto_close_seconds,
        window_name="PCD-DCM Stage 4 - Separate Food Component Convex Hulls",
        **VOLUME_VIEW,
    )

    benchmark = benchmark_timer.finish()
    metrics = DepthColumnConvexMetrics(
        input_points=input_points,
        downsampled_points=downsampled_points,
        cluster_count=len(cluster_candidates),
        selected_cluster_labels=selected_labels,
        selected_cluster_points=len(obj.points),
        component_count=len(component_clouds),
        valid_point_count=int(len(mapping.top_points_xyz)),
        mapped_base_point_count=int(len(mapping.base_points_xyz)),
        dcm_hull_input_point_count=dcm_hull_input_point_count,
        dcm_hull_vertex_count=dcm_hull_vertex_count,
        dcm_hull_triangle_count=dcm_hull_triangle_count,
        dcm_hull_volume_m3=dcm_hull_volume_m3,
        integral_volume_m3=float(grid.volume_m3),
        raw_integral_volume_m3=float(grid.raw_volume_m3),
        reference_aabb_volume_m3=float(reference_aabb.volume()),
        reference_obb_compact_volume_m3=float(reference_obb_compact.volume),
        reference_convex_hull_volume_m3=reference_hull_volume_m3,
        footprint_area_m2=float(grid.footprint_area_m2),
        mean_height_m=float(np.mean(mapping.heights_m)),
        max_height_m=float(np.max(mapping.heights_m)),
        missing_baseline_point_count=int(mapping.base_lookup_histogram["missing"]),
        benchmark=benchmark,
    )
    # payload 面向 GUI/自动化消费，包含可审计的中间统计而非只输出一个体积数值。
    payload = {
        "pcd_path": str(pcd_path),
        "pcd_source_mode": pcd_source,
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
        "hole_completion": hole_completion,
        "depth_column_mapping": {
            "valid_point_count": int(len(mapping.top_points_xyz)),
            "mapped_base_point_count": int(len(mapping.base_points_xyz)),
            "mean_height_m": float(np.mean(mapping.heights_m)),
            "max_height_m": float(np.max(mapping.heights_m)),
            "stats": mapping_stats,
            "lookup_histogram": mapping.base_lookup_histogram,
        },
        "volumes_m3": {
            "dcm_hull": metrics.dcm_hull_volume_m3,
            "pcd_integral": metrics.integral_volume_m3,
            "pcd_integral_raw": metrics.raw_integral_volume_m3,
            "reference_aabb": metrics.reference_aabb_volume_m3,
            "reference_obb_compact": metrics.reference_obb_compact_volume_m3,
            "reference_convex_hull": metrics.reference_convex_hull_volume_m3,
        },
        "mesh_summary": {
            "hull_input_point_count": metrics.dcm_hull_input_point_count,
            "hull_vertex_count": metrics.dcm_hull_vertex_count,
            "hull_triangle_count": metrics.dcm_hull_triangle_count,
            "joggle_inputs": True,
        },
        "dcm_components": component_dcm_summaries,
        "benchmark": benchmark.to_dict(),
    }
    return metrics, payload


def write_result_json(path: str | Path, metrics: DepthColumnConvexMetrics, payload: dict[str, object]) -> Path:
    """将计算摘要和可追溯中间统计写为 UTF-8 JSON，作为 GUI 的唯一结果输入。"""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "pcd_path": payload["pcd_path"],
        "pcd_source_mode": payload["pcd_source_mode"],
        "pcd_has_rgb": payload["pcd_has_rgb"],
        "voxel_size_m": payload["voxel_size_m"],
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
            "min_candidate_size": payload["target_min_candidate_size"],
            "summaries": payload["selected_clusters"],
        },
        "cluster_candidates": payload["cluster_candidates"],
        "surface_filter": payload["surface_filter"],
        "valid_point_filter": payload["valid_point_filter"],
        "baseline_lookup": payload["baseline_lookup"],
        "height_grid": payload["height_grid"],
        "hole_completion": payload["hole_completion"],
        "depth_column_mapping": payload["depth_column_mapping"],
        "mesh_summary": payload["mesh_summary"],
        "dcm_components": payload["dcm_components"],
        "benchmark": payload["benchmark"],
        "volumes_m3": payload["volumes_m3"],
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    """CLI 入口：运行算法、可选写 JSON，并打印人可读的最终摘要和 Benchmark。"""
    args = parse_args()
    try:
        metrics, payload = run_poc(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.result_json:
        result_path = write_result_json(args.result_json, metrics, payload)
        print(f"Saved result JSON: {result_path}", flush=True)

    print("\n=== PCD-DCM 复现完成 ===", flush=True)
    print(f"输入点数: {metrics.input_points}", flush=True)
    print(f"下采样点数: {metrics.downsampled_points}", flush=True)
    print(f"聚类数: {metrics.cluster_count}", flush=True)
    print(f"目标簇标签: {metrics.selected_cluster_labels}", flush=True)
    print(f"目标组件数: {metrics.component_count}", flush=True)
    print(f"目标簇点数: {metrics.selected_cluster_points}", flush=True)
    print(f"有效深度点数: {metrics.valid_point_count}", flush=True)
    print(f"映射到底面点数: {metrics.mapped_base_point_count}", flush=True)
    print(f"DCM hull 输入点数: {metrics.dcm_hull_input_point_count}", flush=True)
    print(f"DCM hull 顶点数: {metrics.dcm_hull_vertex_count}", flush=True)
    print(f"DCM hull 三角面数: {metrics.dcm_hull_triangle_count}", flush=True)
    print(f"Footprint 占用面积: {metrics.footprint_area_m2:.6f} m^2", flush=True)
    print(f"平均高度: {metrics.mean_height_m:.6f} m", flush=True)
    print(f"最大高度: {metrics.max_height_m:.6f} m", flush=True)
    print(f"DCM hull 体积: {metrics.dcm_hull_volume_m3:.6f} m^3", flush=True)
    print(f"参考原始积分体积: {metrics.raw_integral_volume_m3:.6f} m^3", flush=True)
    print(f"参考补洞后积分体积: {metrics.integral_volume_m3:.6f} m^3", flush=True)
    print(f"参考 AABB 体积: {metrics.reference_aabb_volume_m3:.6f} m^3", flush=True)
    print(f"参考 OBB-Compact 体积: {metrics.reference_obb_compact_volume_m3:.6f} m^3", flush=True)
    print(f"参考 Convex Hull 体积: {metrics.reference_convex_hull_volume_m3:.6f} m^3", flush=True)
    print(f"未匹配 baseline 点数: {metrics.missing_baseline_point_count}", flush=True)
    print_benchmark("PCD-DCM", metrics.benchmark)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
