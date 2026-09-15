# food_volume_measure_test

[`food_volume_measure`](../food_volume_measure)（烤箱托盘食材 3D 体积测量库，IM 算法）
的**独立测试工程**：用真实 RealSense D405 点云，在 **桌面 x86_64** 与
**RK3562（aarch64）** 两套目标上验证库的消费接口、算法结果与运行性能。

与库仓库的关系：这里只做**消费者**——依赖 Conan 包 `food_volume_measure/0.1.0`，
只用安装出来的公开头（`volume_types.hpp` / `volume_measurement.hpp` / `volume_log.hpp`），
不触碰库的 `src/` 私有实现。

## 目录结构

```
food_volume_measure_test/
├── profiles/                     # Conan profile
│   ├── gcc13_release             #   桌面 x86_64 / gcc-13
│   └── rk3562_aarch64            #   RK3562 aarch64 / gcc-12 交叉
├── config/
│   └── measurement_config.example.json   # MeasurementConfig 全字段示例
├── example/                      # C++ 消费者示例（两套目标共用一套源码）
│   ├── CMakeLists.txt
│   ├── conanfile.py
│   ├── main.cpp                  #   功能演示
│   ├── bench.cpp                 #   端到端性能基准
│   └── README.md                 #   构建 / 运行 / 部署细节
├── py/                           # Python 侧
│   ├── vm_module.py              #   统一的 pybind 模块加载器
│   ├── test_pybind_module.py     #   公开 API 冒烟测试
│   ├── dump_intermediate_pcd.py  #   触发各阶段中间点云导出
│   ├── make_ground_truth.py      #   生成真值表 ground_truth.json
│   ├── het_ai_volume_trainer.py  #   基于 Optuna 的参数标定
│   ├── replay_o3d_visual_poc*.py #   Open3D 算法对比 POC（不依赖 C++ 库）
│   └── test_data/                #   D405 实测点云 + 真值
├── scripts/
│   └── make_doc_images.py        # 渲染各阶段示意图（供文档使用）
└── het-ai-framework.md           # het-ai 框架说明
```

**生成物一律不入库**（见 `.gitignore`）：`example/build/`、`pylib/`、`middle_data/`、
`out/`、`optuna_trials_*/`、`__pycache__/`。

## 前置：构建库

在本工程可消费之前，`food_volume_measure` 必须先进本地 Conan 缓存，且 **profile 要与
消费时一致**。

```bash
# 桌面（gcc-13 / x86_64 / Release）
cd ../food_volume_measure
conan create . -pr:b=default -pr:h=default -s build_type=Release --build=missing

# 板端（gcc-12 / armv8 / Release）—— 用本工程的交叉 profile
conan create . -pr:b=default -pr:h=../test/food_volume_measure_test/profiles/rk3562_aarch64 \
    -s build_type=Release --build=missing
```

## 两套目标

| | 桌面 x86_64 | RK3562 aarch64 |
|---|---|---|
| profile | `profiles/gcc13_release` | `profiles/rk3562_aarch64` |
| 编译器 | gcc-13 / g++-13 | arm gcc-12（`arm-toolchain/12.3.rel1`） |
| 构建目录 | `example/build/desktop` | `example/build/armv8` |
| 额外选项 | — | 必须关 PCL 的 libusb / pcap |

## 快速开始（桌面）

```bash
cd example
conan install . -pr:b=default -pr:h=../profiles/gcc13_release \
    -of build/desktop --build=missing
cmake -S . -B build/desktop \
    -DCMAKE_TOOLCHAIN_FILE=build/desktop/conan_toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build/desktop -j

# 功能演示（顺带把中间点云导出到 middle_data/cpp）
./build/desktop/example ../py/test_data/d405_260322274982_20260805_142739.pcd \
                        ../py/test_data/d405_260322274982_20260819_180010.pcd \
                        --dump-middle

# 性能基准（重复 5 次）
./build/desktop/bench ../py/test_data/d405_260322274982_20260805_142739.pcd \
                      ../py/test_data/d405_260322274982_20260819_180010.pcd 5
```

交叉编译与板端部署见 `example/README.md`。

## Python 侧

统一用 conda `volume_measure` 环境（Python 3.10，与扩展模块的 ABI 匹配）。

先把 Python 扩展部署进来（生成物，不入库）：

```bash
conan install --requires=food_volume_measure/0.1.0 \
    -pr:b=default -pr:h=profiles/gcc13_release \
    -of pylib --deployer=direct_deploy -nr
```

然后：

```bash
python py/test_pybind_module.py         # 公开 API 冒烟测试
python py/dump_intermediate_pcd.py      # 导出 6 个阶段的中间点云
python py/make_ground_truth.py          # 重新生成真值表
python py/het_ai_volume_trainer.py      # HPO 标定（加 --run 才真正跑）
python scripts/make_doc_images.py       # 渲染阶段示意图
```

`py/` 下所有脚本通过 `vm_module.load_vm()` 取 pybind 模块，SO 路径只需在一处维护。

## 中间结果点云

流水线现在由 `FoodVolumeMeasurer::run()` 一次封装完成。要拿到中间阶段点云，用库自带的
导出开关（C++ 与 Python 都可）：

```cpp
measurer.set_save_middle_cloud(true).set_middle_cloud_dir("middle_data/cpp");
```

```python
measurer.set_save_middle_cloud(True).set_middle_cloud_dir("middle_data/python/case1")
```

每次成功测量写出 6 个 PCD：

| 文件 | 阶段 |
|------|------|
| `0_input_food.pcd` | 原始输入食材帧 |
| `1_downsampled.pcd` | 体素降采样后 |
| `2_remaining.pcd` | 移除背景平面后 |
| `3_baseline_surface.pcd` | 空炉基线表面（由基线栅格重建） |
| `4_food_components.pcd` | 选中的食材连通块（合并） |
| `4_food_component_label<NN>.pcd` | **每个连通块单独一个**（`NN` 为连通块标签，对应 `selected_cluster_labels`） |
| `5_top_surface.pcd` | 顶表面（体积积分所用） |

这也是 `scripts/make_doc_images.py` 生成文档配图的数据来源。
