"""统一加载编译好的 ``food_volume_measure_python`` pybind 模块。

``py/`` 下的脚本一律通过 ``from vm_module import load_vm`` 取模块，避免各自
重复写死 SO 路径与 Python ABI 后缀。

模块来自 ``conan install ... --deployer=direct_deploy`` 的部署产物；部署目录
是生成物（见项目根 ``.gitignore``），需要时重新生成即可。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 本文件位于 <项目根>/py/ 下
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SO_DIR = PROJECT_ROOT / "pylib" / "direct_deploy" / "food_volume_measure" / "lib"
TEST_DATA = PROJECT_ROOT / "py" / "test_data"

MODULE_NAME = "food_volume_measure_python"


def _missing_message() -> str:
    return (
        f"未找到编译产物：{SO_DIR}/{MODULE_NAME}*.so\n\n"
        "请先在 food_volume_measure 仓库构建并部署 Python 模块：\n"
        "  # 1) 构建库（同时产出 pybind 扩展）\n"
        "  conan create . -pr:b=default -pr:h=default -s build_type=Release --build=missing\n\n"
        "  # 2) 部署到本项目（在本项目根目录执行）\n"
        "  conan install --requires=food_volume_measure/0.1.0 \\\n"
        "      -pr:b=default -pr:h=profiles/gcc13_release \\\n"
        "      -of pylib --deployer=direct_deploy -nr\n\n"
        "注意：扩展模块的 Python ABI 必须与解释器一致"
        "（本工程统一使用 conda ``volume_measure`` 环境，Python 3.10）。"
    )


def load_vm():
    """把部署目录加入 ``sys.path`` 并导入 pybind 模块。"""
    candidates = sorted(SO_DIR.glob(f"{MODULE_NAME}*.so")) if SO_DIR.is_dir() else []
    if not candidates:
        raise SystemExit(_missing_message())
    if str(SO_DIR) not in sys.path:
        sys.path.insert(0, str(SO_DIR))
    import food_volume_measure_python as vm

    return vm
