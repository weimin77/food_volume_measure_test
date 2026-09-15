"""早期通用 Open3D 点云回放 POC 与当前 DCM/IM 的共享可视化工具。

该文件本身仍保持单簇 ``auto-center`` 的历史语义；当前多食材正式流程应使用
``replay_o3d_visual_poc_PCD-IM.py`` 或 ``...PCD-DCM.py``。后两者复用本模块
的窗口、格式化、聚类摘要、凸包和 OBB 工具函数。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# 在 Pi 上限制 NumPy/OpenBLAS 线程，避免 Open3D 回放与桌面渲染互相抢占 CPU。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPAT_DIR = REPO_ROOT / "compat"
if str(COMPAT_DIR) not in sys.path:
    sys.path.insert(0, str(COMPAT_DIR))

import open3d as o3d


DEFAULT_PCD = REPO_ROOT / "ZC_O3D_poc" / "cloud_bin_0.pcd"
RAW_VIEW = dict(
    window_name="POC Stage 1 - Raw Point Cloud",
    width=1280,
    height=720,
    point_show_normal=False,
)
SEGMENT_VIEW = dict(
    zoom=0.8,
    front=[-0.4999, -0.1659, -0.8499],
    lookat=[2.1813, 2.0619, 2.0999],
    up=[0.1204, -0.9852, 0.1215],
)
VOLUME_VIEW = dict(
    zoom=0.7,
    front=[0.5439, -0.2333, -0.8060],
    lookat=[2.4615, 2.1331, 1.338],
    up=[-0.1781, -0.9708, 0.1608],
)


@dataclass
class PocMetrics:
    """早期单对象 POC 的数值结果；不是当前 DCM/IM 正式指标。"""
    input_points: int
    downsampled_points: int
    cluster_count: int
    selected_cluster_label: int
    selected_cluster_points: int
    plane_1: np.ndarray
    plane_2: np.ndarray
    aabb_volume: float
    obb_compact_volume: float
    convex_hull_volume: float


@dataclass
class CompactObb:
    """自定义紧凑 OBB：中心、旋转矩阵、三个边长及其体积。"""
    center: np.ndarray
    rotation: np.ndarray
    extent: np.ndarray
    volume: float


@dataclass
class ClusterCandidate:
    """DBSCAN 单个候选簇的几何摘要，供日志、自动选择与 JSON 使用。"""
    label: int
    point_count: int
    centroid: np.ndarray
    min_corner: np.ndarray
    max_corner: np.ndarray
    extent: np.ndarray
    radial_distance_xy: float

    def to_dict(self) -> dict[str, object]:
        """转换为 JSON 可序列化的米制字段。"""
        return {
            "label": self.label,
            "point_count": self.point_count,
            "centroid_m": self.centroid.tolist(),
            "min_corner_m": self.min_corner.tolist(),
            "max_corner_m": self.max_corner.tolist(),
            "extent_m": self.extent.tolist(),
            "radial_distance_xy_m": self.radial_distance_xy,
        }


def parse_args() -> argparse.Namespace:
    """解析早期通用 POC 的重放、平面分割、聚类和显示参数。"""
    parser = argparse.ArgumentParser(
        description="Replay the Open3D POC algorithm flow from ZC_O3D_poc/visual.pyc."
    )
    parser.add_argument(
        "--pcd",
        default=str(DEFAULT_PCD),
        help="Point-cloud input used for the O3D POC replay.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.02,
        help="Voxel downsample size in meters.",
    )
    parser.add_argument(
        "--plane1-distance-threshold",
        type=float,
        default=0.01,
        help="RANSAC distance threshold for the first dominant plane.",
    )
    parser.add_argument(
        "--plane2-distance-threshold",
        type=float,
        default=0.03,
        help="RANSAC distance threshold for the second dominant plane.",
    )
    parser.add_argument(
        "--cluster-eps",
        type=float,
        default=0.05,
        help="DBSCAN eps parameter for the second-stage clustering.",
    )
    parser.add_argument(
        "--cluster-min-points",
        type=int,
        default=8,
        help="DBSCAN min_points parameter for the second-stage clustering.",
    )
    parser.add_argument(
        "--target-label",
        default="auto",
        help="Cluster label to retain after DBSCAN, or 'auto' to pick the oven-center object.",
    )
    parser.add_argument(
        "--max-obb-faces",
        type=int,
        default=500,
        help="Maximum convex-hull face count before compact-OBB normal deduplication.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Deterministic random seed for Open3D and NumPy.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Skip desktop visualization windows and print only numeric results.",
    )
    parser.add_argument(
        "--auto-close-seconds",
        type=float,
        default=None,
        help="Automatically close each desktop stage after N seconds.",
    )
    parser.add_argument(
        "--result-json",
        default=None,
        help="Optional JSON path for machine-readable validation results.",
    )
    return parser.parse_args()


def print_stage(message: str) -> None:
    """输出统一的阶段标题，便于 GUI 终端和手工日志定位处理进度。"""
    print(f"\n=== {message} ===", flush=True)


def draw_stage(
    geometries: list[o3d.geometry.Geometry],
    headless: bool,
    auto_close_seconds: float | None = None,
    **kwargs,
) -> None:
    """显示一个 Open3D 阶段窗口，或在 headless 模式下跳过绘制。

    此处使用 ``Visualizer.poll_events`` 循环而非反复 ``draw_geometries``，以
    避免 Raspberry Pi 的 XWayland 后端在多窗口关闭后出现不稳定。
    """
    if headless:
        return

    window_name = kwargs.pop("window_name", "Open3D")
    width = int(kwargs.pop("width", 1280))
    height = int(kwargs.pop("height", 720))
    left = int(kwargs.pop("left", 50))
    top = int(kwargs.pop("top", 50))
    point_show_normal = bool(kwargs.pop("point_show_normal", False))
    point_size = kwargs.pop("point_size", None)
    line_width = kwargs.pop("line_width", None)
    zoom = kwargs.pop("zoom", None)
    front = kwargs.pop("front", None)
    lookat = kwargs.pop("lookat", None)
    up = kwargs.pop("up", None)

    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported visualization kwargs: {unsupported}")

    vis = o3d.visualization.Visualizer()
    ok = vis.create_window(
        window_name=window_name,
        width=width,
        height=height,
        left=left,
        top=top,
        visible=True,
    )
    if not ok:
        # Wayland / 无显示环境下 GLFW 可能无法初始化窗口；回退为 headless 继续跑算法，
        # 保证 F5 调试时流程不会因弹窗失败而中断。
        print(f"[Desktop] 无法创建窗口: {window_name}。跳过可视化，继续执行。", flush=True)
        return

    try:
        for geometry in geometries:
            vis.add_geometry(geometry)

        render_option = vis.get_render_option()
        render_option.point_show_normal = point_show_normal
        if point_size is not None:
            render_option.point_size = float(point_size)
        if line_width is not None:
            render_option.line_width = float(line_width)

        # Pi/XWayland 下轮询 Visualizer 比重复 draw_geometries 稳定；预设相机向量
        # 暂保留为配置资料，待桌面后端稳定后再启用，避免引入渲染端崩溃风险。
        _ = (zoom, front, lookat, up)

        if auto_close_seconds is None:
            print(f"[Desktop] 已打开窗口: {window_name}。关闭该窗口后继续下一阶段。", flush=True)
        else:
            print(
                f"[Desktop] 已打开窗口: {window_name}。将在 {auto_close_seconds:.1f}s 后自动关闭。",
                flush=True,
            )

        start = time.monotonic()
        while True:
            if not vis.poll_events():
                break
            vis.update_renderer()
            if auto_close_seconds is not None and time.monotonic() - start >= auto_close_seconds:
                print(f"[Desktop] 自动关闭窗口: {window_name}", flush=True)
                break
            time.sleep(0.01)
    finally:
        vis.destroy_window()


def format_plane(plane: np.ndarray) -> str:
    """把 ``[a,b,c,d]`` 平面系数格式化为便于终端阅读的方程。"""
    a, b, c, d = plane
    return f"{a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0"


def format_vector(vector: np.ndarray) -> str:
    """以固定四位小数格式化三维向量。"""
    return "[" + ", ".join(f"{value:.4f}" for value in vector.tolist()) + "]"


def apply_uniform_color(
    geometry: o3d.geometry.Geometry,
    color: tuple[float, float, float] | list[float],
) -> None:
    """为点云、三角网格或线集整体赋色，供分阶段回放区分几何体。"""
    rgb = np.asarray(color, dtype=np.float64)
    # Open3D 三种几何对象的颜色数组长度语义不同：点、顶点和线分别对应各自数量。
    geometry_type = geometry.get_geometry_type()

    if geometry_type == o3d.geometry.Geometry.Type.PointCloud:
        point_count = len(geometry.points)
        geometry.colors = o3d.utility.Vector3dVector(np.tile(rgb, (point_count, 1)))
        return

    if geometry_type == o3d.geometry.Geometry.Type.TriangleMesh:
        vertex_count = len(geometry.vertices)
        geometry.vertex_colors = o3d.utility.Vector3dVector(np.tile(rgb, (vertex_count, 1)))
        return

    if geometry_type == o3d.geometry.Geometry.Type.LineSet:
        line_count = len(geometry.lines)
        geometry.colors = o3d.utility.Vector3dVector(np.tile(rgb, (line_count, 1)))
        return

    raise TypeError(f"Unsupported geometry type for uniform coloring: {geometry_type}")


def compute_compact_minimum_volume_obb(
    pcd: o3d.geometry.PointCloud,
    max_faces: int,
    seed: int,
) -> CompactObb:
    """以凸包面法向为候选轴搜索更紧凑的定向包围盒。

    这是参考几何，不参与 DCM/IM 正式体积。凸包面过多时按法向夹角去重，
    防止在树莓派上进行过多候选方向计算。
    """
    np.random.seed(seed)
    pts = np.asarray(pcd.points)
    if len(pts) < 4:
        obb = pcd.get_oriented_bounding_box()
        extent = np.asarray(obb.extent)
        return CompactObb(
            center=np.asarray(obb.center),
            rotation=np.asarray(obb.R),
            extent=extent,
            volume=float(np.prod(extent)),
        )

    hull, _ = pcd.compute_convex_hull()
    hull_verts = np.asarray(hull.vertices)
    hull_tris = np.asarray(hull.triangles)

    # 用凸包每个非退化三角面的法向作为候选姿态，避免直接依赖随机 PCA 朝向。
    normals = []
    for tri in hull_tris:
        v0, v1, v2 = hull_verts[tri[0]], hull_verts[tri[1]], hull_verts[tri[2]]
        n = np.cross(v1 - v0, v2 - v0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        normals.append(n / n_norm)

    normals = np.array(normals)
    if len(normals) == 0:
        obb = pcd.get_oriented_bounding_box()
        extent = np.asarray(obb.extent)
        return CompactObb(
            center=np.asarray(obb.center),
            rotation=np.asarray(obb.R),
            extent=extent,
            volume=float(np.prod(extent)),
        )

    if len(normals) > max_faces:
        # 仅保留夹角相差超过 2° 的法向，可在 Pi 上控制候选 OBB 的计算次数。
        unique_normals = [normals[0]]
        for n in normals[1:]:
            if all(
                np.arccos(np.clip(np.dot(n, u), -1.0, 1.0)) > np.deg2rad(2.0)
                for u in unique_normals
            ):
                unique_normals.append(n)
        normals = np.array(unique_normals)
        print(f"凸包面法向从 {len(hull_tris)} 降采样到 {len(normals)} 个")

    best_volume = np.inf
    best_box = None

    # 每个候选法向作为局部 z 轴；在该坐标系取 min/max 即得到一个候选 OBB。
    for z_axis in normals:
        if abs(z_axis[0]) < 0.9:
            x_axis = np.cross(z_axis, [1.0, 0.0, 0.0])
        else:
            x_axis = np.cross(z_axis, [0.0, 1.0, 0.0])
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        rotation = np.column_stack([x_axis, y_axis, z_axis])
        pts_local = pts @ rotation

        min_corner = pts_local.min(axis=0)
        max_corner = pts_local.max(axis=0)
        extent = max_corner - min_corner
        volume = extent[0] * extent[1] * extent[2]

        if volume < best_volume:
            best_volume = volume
            center_local = (min_corner + max_corner) / 2.0
            center_world = center_local @ rotation.T
            best_box = CompactObb(
                center=center_world.astype(np.float64),
                rotation=rotation.astype(np.float64),
                extent=extent.astype(np.float64),
                volume=float(volume),
            )

    if best_box is None:
        obb = pcd.get_oriented_bounding_box()
        extent = np.asarray(obb.extent)
        return CompactObb(
            center=np.asarray(obb.center),
            rotation=np.asarray(obb.R),
            extent=extent,
            volume=float(np.prod(extent)),
        )
    return best_box


def compact_obb_to_lineset(compact_obb: CompactObb) -> o3d.geometry.LineSet:
    """将紧凑 OBB 的八个角和十二条边转换为 Open3D 绿色线框。"""
    half_extent = compact_obb.extent / 2.0
    local_corners = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    corners = (local_corners * half_extent) @ compact_obb.rotation.T + compact_obb.center
    lines = np.array(
        [
            [0, 1],
            [1, 3],
            [3, 2],
            [2, 0],
            [4, 5],
            [5, 7],
            [7, 6],
            [6, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
        dtype=np.int32,
    )
    colors = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float64), (len(lines), 1))

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(corners)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector(colors)
    return lineset


def summarize_clusters(points: o3d.geometry.PointCloud, labels: np.ndarray) -> list[ClusterCandidate]:
    """对每个非噪声 DBSCAN 标签统计中心、范围、尺寸和距相机 XY 原点距离。"""
    pts = np.asarray(points.points)
    candidates: list[ClusterCandidate] = []

    for label in sorted(int(label) for label in np.unique(labels) if label >= 0):
        cluster_points = pts[labels == label]
        centroid = cluster_points.mean(axis=0)
        min_corner = cluster_points.min(axis=0)
        max_corner = cluster_points.max(axis=0)
        extent = max_corner - min_corner
        radial_distance_xy = float(np.linalg.norm(centroid[:2]))
        candidates.append(
            ClusterCandidate(
                label=label,
                point_count=len(cluster_points),
                centroid=centroid.astype(np.float64),
                min_corner=min_corner.astype(np.float64),
                max_corner=max_corner.astype(np.float64),
                extent=extent.astype(np.float64),
                radial_distance_xy=radial_distance_xy,
            )
        )

    return candidates


def print_cluster_candidates(candidates: list[ClusterCandidate]) -> None:
    """以稳定格式打印候选簇，供参数调试和 GUI 终端阅读。"""
    print("Cluster candidates:", flush=True)
    for candidate in candidates:
        print(
            "  "
            f"label={candidate.label}, "
            f"points={candidate.point_count}, "
            f"centroid={format_vector(candidate.centroid)}, "
            f"extent={format_vector(candidate.extent)}, "
            f"radial_xy={candidate.radial_distance_xy:.4f} m",
            flush=True,
        )


def parse_target_label(value: str | None) -> int | None:
    """解析早期单目标标签；None 表示保留 auto-center 历史选择策略。"""
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in {"", "auto", "auto-center", "center"}:
        return None

    try:
        return int(normalized)
    except ValueError as exc:
        raise ValueError("--target-label must be an integer or one of: auto, auto-center, center") from exc


def resolve_target_candidate(
    candidates: list[ClusterCandidate],
    cluster_min_points: int,
    requested_label: int | None,
) -> tuple[ClusterCandidate, str, int]:
    """从候选簇选出一个历史 POC 目标；当前多对象主链路不使用此逻辑。"""
    if not candidates:
        raise RuntimeError("DBSCAN 没有产生任何有效聚类，请调整参数后重试。")

    if requested_label is not None:
        for candidate in candidates:
            if candidate.label == requested_label:
                return candidate, "manual", cluster_min_points
        available = ", ".join(str(candidate.label) for candidate in candidates)
        raise RuntimeError(f"目标 cluster label={requested_label} 不存在。可用 labels: {available}")

    largest_cluster_size = max(candidate.point_count for candidate in candidates)
    min_candidate_size = max(cluster_min_points * 4, int(round(largest_cluster_size * 0.05)))
    eligible = [candidate for candidate in candidates if candidate.point_count >= min_candidate_size]
    if not eligible:
        eligible = candidates

    # 历史 auto-center 策略：优先选择接近相机 XY 光轴的簇，再以深度和点数打破平局。
    selected = min(
        eligible,
        key=lambda candidate: (
            candidate.radial_distance_xy,
            abs(float(candidate.centroid[2])),
            -candidate.point_count,
        ),
    )
    return selected, "auto-center", min_candidate_size


def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    """检查路径、扩展名和空点云后，用 Open3D 读取 PCD。"""
    if not path.exists():
        raise FileNotFoundError(f"PCD 文件不存在: {path}")

    if path.is_dir():
        raise IsADirectoryError(
            "PCD 路径指向的是目录，不是点云文件: "
            f"{path}\n"
            "如果你使用了 \"$LATEST_PCD\"，先执行:\n"
            "  LATEST_PCD=$(ls -t captures/d405_260322274982_*.pcd | head -n 1)\n"
            "再重新运行复现命令。"
        )

    if path.suffix.lower() != ".pcd":
        raise ValueError(f"输入文件不是 .pcd: {path}")

    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise RuntimeError(f"读取到空点云: {path}")
    return pcd


def select_cluster(points: o3d.geometry.PointCloud, labels: np.ndarray, target_label: int) -> o3d.geometry.PointCloud:
    """按 DBSCAN 标签从原点云提取一个组件，并在空选择时明确报错。"""
    mask = labels == target_label
    indices = np.where(mask)[0]
    if len(indices) == 0:
        raise RuntimeError(f"目标 cluster label={target_label} 为空。")
    return points.select_by_index(indices.tolist())


def run_poc(args: argparse.Namespace) -> tuple[PocMetrics, dict[str, object]]:
    """执行早期四阶段单对象 POC，并返回指标和可写入 JSON 的载荷。"""
    # 固定 NumPy 和 Open3D 随机种子，使 RANSAC/凸包参考结果尽量可复现。
    np.random.seed(args.seed)
    o3d.utility.random.seed(args.seed)

    pcd_path = Path(args.pcd).expanduser().resolve()
    print_stage("加载 PCD")
    print(f"PCD path: {pcd_path}")
    pcd = load_point_cloud(pcd_path)
    print(f"初始点云数量: {len(pcd.points)}")
    has_rgb = bool(pcd.has_colors() and len(np.asarray(pcd.colors)) == len(pcd.points))
    print(f"RGB colors loaded: {'yes' if has_rgb else 'no'}")

    draw_stage([pcd], args.headless, auto_close_seconds=args.auto_close_seconds, **RAW_VIEW)

    print_stage("体素下采样")
    pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
    print(f"体素下采样后点云数量: {len(pcd.points)}")

    print_stage("RANSAC 平面 1")
    # 两个 RANSAC 平面是旧 POC 的背景去除方式；主算法已改用空炉基线差分分割。
    plane_1, inliers_1 = pcd.segment_plane(
        distance_threshold=args.plane1_distance_threshold,
        ransac_n=3,
        num_iterations=1000,
    )
    print(f"Plane 1: {format_plane(np.asarray(plane_1))}")
    ground = pcd.select_by_index(inliers_1)
    apply_uniform_color(ground, [1.0, 0.0, 0.0])
    remaining = pcd.select_by_index(inliers_1, invert=True)

    print_stage("RANSAC 平面 2")
    plane_2, inliers_2 = remaining.segment_plane(
        distance_threshold=args.plane2_distance_threshold,
        ransac_n=3,
        num_iterations=1000,
    )
    print(f"Plane 2: {format_plane(np.asarray(plane_2))}")
    wall_a = remaining.select_by_index(inliers_2)
    apply_uniform_color(wall_a, [0.0, 0.0, 1.0])
    remaining2 = remaining.select_by_index(inliers_2, invert=True)

    print_stage("DBSCAN 聚类")
    # DBSCAN 标签 -1 表示噪声，不进入候选簇与体积估算。
    labels = np.array(
        remaining2.cluster_dbscan(
            eps=args.cluster_eps,
            min_points=args.cluster_min_points,
            print_progress=False,
        )
    )

    if not np.any(labels >= 0):
        raise RuntimeError("DBSCAN 没有找到任何非噪声聚类。请减小 voxel-size 或增大 cluster-eps 后重试。")

    cluster_candidates = summarize_clusters(remaining2, labels)
    print(f"DBSCAN cluster count: {len(cluster_candidates)}")
    print_cluster_candidates(cluster_candidates)

    requested_target_label = parse_target_label(args.target_label)
    selected_candidate, target_mode, min_candidate_size = resolve_target_candidate(
        cluster_candidates,
        cluster_min_points=args.cluster_min_points,
        requested_label=requested_target_label,
    )
    print(
        f"Selected target label: {selected_candidate.label} "
        f"(mode={target_mode}, radial_xy={selected_candidate.radial_distance_xy:.4f} m, "
        f"points={selected_candidate.point_count}, min_candidate_size={min_candidate_size})"
    )

    obj = select_cluster(remaining2, labels, selected_candidate.label)
    print(f"过滤后点云数量: {len(obj.points)}")
    selection_outline = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        obj.get_axis_aligned_bounding_box()
    )
    apply_uniform_color(selection_outline, [1.0, 1.0, 0.0])

    draw_stage(
        [ground, wall_a, obj, selection_outline],
        args.headless,
        auto_close_seconds=args.auto_close_seconds,
        window_name="POC Stage 2 - Plane Segmentation + DBSCAN",
        **SEGMENT_VIEW,
    )

    print_stage("包围盒体积估算")
    aabb = obj.get_axis_aligned_bounding_box()
    aabb_lineset = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
    apply_uniform_color(aabb_lineset, [1.0, 0.0, 0.0])
    obb_compact = compute_compact_minimum_volume_obb(
        obj,
        max_faces=args.max_obb_faces,
        seed=args.seed,
    )
    print(f"体积 (AABB): {aabb.volume():.6f} m^3")
    print(f"体积 (OBB-Compact): {obb_compact.volume:.6f} m^3")
    compact_obb_lineset = compact_obb_to_lineset(obb_compact)

    draw_stage(
        [obj, aabb_lineset, compact_obb_lineset],
        args.headless,
        auto_close_seconds=args.auto_close_seconds,
        window_name="POC Stage 3 - Bounding Boxes",
        **VOLUME_VIEW,
    )

    print_stage("凸包体积估算")
    hull_mesh, _ = obj.compute_convex_hull()
    hull_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(hull_mesh)
    apply_uniform_color(hull_wireframe, [0.0, 0.0, 0.0])
    print(f"凸包体积: {hull_mesh.get_volume():.6f} m^3")

    draw_stage(
        [obj, hull_wireframe],
        args.headless,
        auto_close_seconds=args.auto_close_seconds,
        window_name="POC Stage 4 - Convex Hull",
        width=1280,
        height=720,
    )

    # 结果同时保留数字指标和调试载荷；后者写 JSON 供 GUI/人工追溯选择依据。
    metrics = PocMetrics(
        input_points=len(load_point_cloud(pcd_path).points),
        downsampled_points=len(pcd.points),
        cluster_count=len(cluster_candidates),
        selected_cluster_label=selected_candidate.label,
        selected_cluster_points=len(obj.points),
        plane_1=np.asarray(plane_1),
        plane_2=np.asarray(plane_2),
        aabb_volume=float(aabb.volume()),
        obb_compact_volume=float(obb_compact.volume),
        convex_hull_volume=float(hull_mesh.get_volume()),
    )
    debug_payload = {
        "pcd_path": str(pcd_path),
        "pcd_has_rgb": has_rgb,
        "voxel_size_m": args.voxel_size,
        "plane_1": np.asarray(plane_1).tolist(),
        "plane_2": np.asarray(plane_2).tolist(),
        "cluster_count": len(cluster_candidates),
        "selected_cluster_label": selected_candidate.label,
        "target_selection_mode": target_mode,
        "target_min_candidate_size": min_candidate_size,
        "selected_cluster": selected_candidate.to_dict(),
        "cluster_candidates": [candidate.to_dict() for candidate in cluster_candidates],
    }
    return metrics, debug_payload


def write_result_json(path: str | Path, metrics: PocMetrics, payload: dict[str, object]) -> Path:
    """把旧 POC 的关键中间结果和参考体积写入 UTF-8 JSON。"""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "pcd_path": payload["pcd_path"],
        "pcd_has_rgb": payload["pcd_has_rgb"],
        "input_points": metrics.input_points,
        "downsampled_points": metrics.downsampled_points,
        "cluster_count": metrics.cluster_count,
        "plane_1": payload["plane_1"],
        "plane_2": payload["plane_2"],
        "target": {
            "mode": payload["target_selection_mode"],
            "label": metrics.selected_cluster_label,
            "point_count": metrics.selected_cluster_points,
            "min_candidate_size": payload["target_min_candidate_size"],
            "summary": payload["selected_cluster"],
        },
        "cluster_candidates": payload["cluster_candidates"],
        "volumes_m3": {
            "aabb": metrics.aabb_volume,
            "obb_compact": metrics.obb_compact_volume,
            "convex_hull": metrics.convex_hull_volume,
        },
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    """早期通用 POC 的 CLI 入口；异常转换为非零退出码供 GUI 识别。"""
    args = parse_args()
    try:
        metrics, payload = run_poc(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.result_json:
        result_path = write_result_json(args.result_json, metrics, payload)
        print(f"Saved result JSON: {result_path}")

    print_stage("POC 复现完成")
    print(f"输入点数: {metrics.input_points}")
    print(f"下采样点数: {metrics.downsampled_points}")
    print(f"聚类数: {metrics.cluster_count}")
    print(f"目标簇标签: {metrics.selected_cluster_label}")
    print(f"目标簇点数: {metrics.selected_cluster_points}")
    print(f"AABB 体积: {metrics.aabb_volume:.6f} m^3")
    print(f"OBB-Compact 体积: {metrics.obb_compact_volume:.6f} m^3")
    print(f"Convex Hull 体积: {metrics.convex_hull_volume:.6f} m^3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
