# example — 通过 Conan 包消费 `food_volume_measure`

把 `food_volume_measure` 当作**外部依赖**消费的最小工程，用来验证安装出来的包
（头文件 + 静态库）在**桌面 x86_64** 与 **RK3562（aarch64）** 两套目标上都能正常
构建、链接和运行。

- `example` — 功能演示：跑一次测量并打印完整诊断（含逐连通块明细）
- `bench` — 性能基准：`run()` 端到端耗时 + CPU 时间 + 进程峰值内存

两者都是纯消费者：只 include 安装出来的公开头
（`volume_types.hpp` / `volume_measurement.hpp` / `volume_log.hpp`），
不依赖任何 `src/` 下的私有实现。

## 前置：把库装进本地 Conan 缓存

```bash
# 在 food_volume_measure 仓库根目录
conan create . -pr:b=default -pr:h=default -s build_type=Release --build=missing
```

`example/conanfile.py` 依赖 `food_volume_measure/0.1.0`，**profile 必须与编库时一致**
（桌面 gcc-13 / RK3562 gcc-12 + armv8，均 Release）。

## 构建与运行（桌面 x86_64）

```bash
cd example
conan install . -pr:b=default -pr:h=../profiles/gcc13_release \
  -of build/desktop --build=missing

cmake -S . -B build/desktop \
  -DCMAKE_TOOLCHAIN_FILE=build/desktop/conan_toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/desktop -j

./build/desktop/example ../../py/test_data/d405_260322274982_20260805_142739.pcd \
                        ../../py/test_data/d405_260322274982_20260819_180010.pcd
./build/desktop/bench <baseline.pcd> <food.pcd> 5
```

## 构建（交叉编译到 RK3562 / aarch64）

```bash
cd example
conan install . -pr:b=default -pr:h=../profiles/rk3562_aarch64 \
  -of build/armv8 --build=missing \
  -o 'pcl/*:with_libusb=False' -o 'pcl/*:with_pcap=False'

cmake -S . -B build/armv8 \
  -DCMAKE_TOOLCHAIN_FILE=build/armv8/conan_toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/armv8 -j
```

> 交叉编译必须关闭 PCL 的 libusb / pcap，否则会撞 libudev 报错。

## 板端部署

```bash
scp build/armv8/example build/armv8/bench root@<板子IP>:/root/
scp <baseline.pcd> <food.pcd> root@<板子IP>:/root/
ssh root@<板子IP> '/root/bench /root/<baseline.pcd> /root/<food.pcd> 5'
```

## 用法

```
example <baseline.pcd> <food.pcd> [config.json] [--dump-middle[=DIR]]
bench   <baseline.pcd> <food.pcd> [runs] [config.json]
```

- `config.json` 可选，对应 `MeasurementConfig`；示例见 `../config/measurement_config.example.json`
- `--dump-middle[=DIR]` 让库把各阶段中间点云写到 `DIR`（默认 `middle_data/cpp`），
  便于用 Open3D / CloudCompare 逐阶段排查。这会顺序写出 6 个文件：

  | 文件 | 阶段 |
  |------|------|
  | `0_input_food.pcd` | 原始输入食材帧 |
  | `1_downsampled.pcd` | 体素降采样后 |
  | `2_remaining.pcd` | 移除背景平面后 |
  | `3_baseline_surface.pcd` | 空炉基线表面（由基线栅格重建） |
  | `4_food_components.pcd` | 选中的食材连通块（合并） |
  | `4_food_component_label<NN>.pcd` | 每个连通块单独一个，`NN` 为连通块标签 |
  | `5_top_surface.pcd` | 顶表面（体积积分所用） |

## 目录说明

```
example/
├── CMakeLists.txt   # 两个可执行目标
├── conanfile.py     # 声明 food_volume_measure 依赖
├── main.cpp         # 功能演示
└── bench.cpp        # 端到端性能基准
```

`build/` 及其下所有产物均为生成物，不入版本管理。

```cpp
measurer.set_save_middle_cloud(true).set_middle_cloud_dir("middle_data/cpp");
```

`run()` 成功后写出各阶段点云；**每个选中的连通块还会单独写出一个 PCD**，
多食材场景可以逐个连通域排查：
