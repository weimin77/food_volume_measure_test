"""Generate point-cloud effect images for the Sphinx documentation.

Renders the IM volume-measurement pipeline stages (input -> voxel downsampling
-> background-plane removal -> food components -> top surface) as PNG images
under docs/sphinx/images/ using Open3D's headless EGL renderer.

Run:  python3 make_doc_images.py
"""
import argparse
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIDDLE_DATA = PROJECT_ROOT / "middle_data" / "python"
TEST_DATA = PROJECT_ROOT / "py" / "test_data"

# 默认写到本项目；用 --out-dir 可以指向库仓库的 docs/sphinx/images
DEFAULT_OUT_DIR = PROJECT_ROOT / "out" / "doc_images"

W, H = 900, 620


def colorize(pcd, cmap_name="turbo"):
    """Color points by their Z (height) coordinate."""
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd
    z = pts[:, 2]
    lo, hi = z.min(), z.max()
    norm = (z - lo) / (hi - lo + 1e-12)
    # turbo-style colormap approximated with a simple gradient for clarity.
    # Use matplotlib if available for a nicer palette.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm
        colors = cm.get_cmap(cmap_name)(norm)[:, :3]
    except Exception:
        # fallback: blue -> green -> red gradient
        colors = np.stack([norm, 1 - np.abs(2 * norm - 1), 1 - norm], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def render(pcd, out_path, point_size=3.0, bg=(1.0, 1.0, 1.0, 1.0)):
    renderer = rendering.OffscreenRenderer(W, H)
    renderer.scene.set_background(bg)
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = point_size
    renderer.scene.add_geometry("pcd", pcd, mat)

    pts = np.asarray(pcd.points)
    center = pts.mean(axis=0)
    ext = pts.max(axis=0) - pts.min(axis=0)
    diag = float(np.linalg.norm(ext))

    # Camera from a slanted top view so the 3D structure is visible.
    eye = center + np.array([0.0, -diag * 1.6, diag * 1.1])
    up = np.array([0.0, 0.0, 1.0])
    renderer.scene.camera.look_at(center, eye, up)
    # Perspective projection matched to the cloud size.
    renderer.scene.camera.set_projection(
        55.0, float(W) / float(H), diag * 0.05, diag * 20.0,
        rendering.Camera.FovType.Vertical,
    )

    img = renderer.render_to_image()
    o3d.io.write_image(str(out_path), img)
    renderer = None


def main() -> int:
    parser = argparse.ArgumentParser(description="渲染 IM 流水线各阶段点云示意图")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="PNG 输出目录")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stages = [
        ("im_input.png", TEST_DATA / "d405_260322274982_20260819_180010.pcd",
         "input point cloud", 1.5),
        ("im_downsampled.png", MIDDLE_DATA / "1_downsampled.pcd",
         "voxel downsampling", 5.0),
        ("im_plane_removed.png", MIDDLE_DATA / "2_remaining.pcd",
         "background-plane removal", 6.0),
        ("im_food_components.png", MIDDLE_DATA / "4_food_components.pcd",
         "food components", 8.0),
        ("im_top_surface.png", MIDDLE_DATA / "5_top_surface.pcd",
         "top surface / volume", 9.0),
    ]

    for name, path, label, ps in stages:
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        pcd = o3d.io.read_point_cloud(str(path))
        if len(pcd.points) == 0:
            print(f"[skip] empty {path}")
            continue
        pcd = colorize(pcd)
        render(pcd, out_dir / name, point_size=ps)
        print(f"[ok] {name} <- {path} ({len(pcd.points)} pts)")

    print(f"\n输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
