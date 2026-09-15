# het-ai 框架说明

> 目标读者：需要在本地用 het-ai 跑「食材体积测量」参数校准/消融的开发者。
> 本文基于已安装的 `het-ai==1.0.0`（`het_ai.studio` / `het_ai.dvc` / `het_ai.mlflow` 三个子模块）。

---

## 1. het-ai 是什么

一句话：**一个「框架无关」的 MLOps 训练 DSL，底层用 Optuna 做超参搜索（HPO），上层可选挂 DVC（数据版本化）和 MLflow（实验追踪）。**

它要解决的核心痛点：

- 写 HPO 时，搜索空间定义、数据加载、目标函数、结果导出、实验追踪这些样板代码，每个项目都要重复写一遍；
- het-ai 把它们固化成一套 **约定式接口**（子类化 `BaseTrainer` + 一个 `@search` 装饰器），你只需要写「数据怎么来」和「一组超参怎么算出一个分数」两件事；
- 剩下的（Optuna study 创建、trial 采样、剪枝、best trial 选择、模型导出、MLflow 上报、DVC 数据版本注入）由框架自动完成。

它本身**不做任何具体的机器学习**——不绑定 PyTorch/TensorFlow/sklearn，只负责「编排训练这件事的工程骨架」。

---

## 2. 三个子模块

| 子模块 | 职责 | 是否必装 |
|---|---|---|
| `het_ai.studio` | 核心：`BaseTrainer`、`TrainConfig`、`DataBundle`、`Tunable*` 类型、`@search` 装饰器、`dry_run()` / `run()` | 核心包自带 |
| `het_ai.dvc` | 数据版本化：`DVCConfig`、`DVCLoader`（拉取数据并注入 tag/sha 元信息） | `[platform]` extra |
| `het_ai.mlflow` | 实验追踪：`MLflowConfig`、`MLflowRunLogger`（训练结束自动上报） | `[platform]` extra |

安装：

```bash
pip install het-ai                  # 仅 HPO（核心）
pip install "het-ai[platform]"      # HPO + DVC + MLflow
pip install "het-ai[torch]"         # 按需加载框架 extra
```

---

## 3. 核心编程模型

### 3.1 三步式结构

```python
from het_ai.studio import BaseTrainer, TrainConfig, DataBundle

class MyTrainer(BaseTrainer):
    objectives = {"accuracy": "maximize"}          # ① 声明目标方向

    @BaseTrainer.search(                           # ② 用 Tunable* 标注搜索空间
        lr=BaseTrainer.TunableFloat(1e-4, 1e-2, log=True),
        hidden=BaseTrainer.TunableInt(32, 256, step=32),
    )
    def train(self, data, lr, hidden):             # ③ 一组超参 → 一个分数
        ...                                        #    中间可 self.report(step, v) 支持剪枝
        return score                               # float / (float, artifact) / Result(...)

    def load_data(self, dvc_data_root):            # 正式数据加载
        ...
        return DataBundle(splits={"train": ..., "val": ...})

    def mock_data(self):                           # 可选：dry-run 用的快速数据
        ...
```

### 3.2 `@search` 装饰器与 Tunable 类型

`@BaseTrainer.search(**kw)` 只是给 `train()` 挂上搜索空间元数据，**不做采样**。装饰器里的参数名必须和 `train()` 签名一一对应（框架会在子类创建时校验）。

三种「幽灵类型」（本质是 `int/float/str` 子类，额外携带 `_meta`）：

| 类型 | 参数 | 采样方式 |
|---|---|---|
| `TunableInt(low, high, step, log)` | 整数网格 | `trial.suggest_int` |
| `TunableFloat(low, high, step, log)` | 浮点区间 | `trial.suggest_float` |
| `TunableCategorical(choices)` | 离散候选 | `trial.suggest_categorical` |

多目标用 `Result(a=..., b=...)` 包装返回值，并配合 `objectives = {"a": "minimize", "b": "maximize"}`。

### 3.3 `train()` 的返回值约定

```python
float                          # 单目标
(float, artifact)              # 单目标 + 导出产物
(float, artifact, dict)        # 单目标 + 产物 + 自定义指标
Result(a=..., b=...)           # 多目标
(Result(...), artifact)        # 多目标 + 产物
(Result(...), artifact, dict)  # 多目标 + 产物 + 指标
```

### 3.4 `DataBundle`

通用数据容器，不预设任务类型：

```python
DataBundle(
    splits={"train": {"X": ..., "y": ...}, "val": {...}},  # 任意结构
    feature_list=[...], target_list=[...],
    meta={"任意": "元信息"},                                # 比如 vm 模块、基线点云
    lineage_datasets=[...],                                 # 可选：预构造的 mlflow Dataset
)
```

它带 `X_train / y_train / X_val / y_val ...` 便捷属性，但 `splits` 里放什么完全由你决定（我们的场景放的是 `samples` 列表）。

---

## 4. 运行流程

### 4.1 `dry_run()` —— 快速验证链路

```python
trainer = MyTrainer(TrainConfig(...))
trainer.dry_run()
```

内部步骤：

1. 先尝试 `mock_data()`；未实现（抛 `NotImplementedError`）则回退 `load_data()`；
2. 把搜索空间每个参数取 **默认值 = low 下界**（categorical 取第一个候选）；
3. 用这套默认参数跑一次 `train()`；
4. 若有 artifact，调 `export_model()` 导出；
5. 返回 `{"score", "artifact", "elapsed", "export_path"}`。

**关键点：dry-run 不做优化，只跑一次。** 所以它的 score 由「low 边界参数」决定——这就是为什么我们把 low 设成参考配置时 dry-run 分数恒为 0（自洽性验证）。

### 4.2 `run()` —— 完整 HPO

```python
result = trainer.run()
print(result.metric_dict, result.params_dict)
```

内部（`WorkflowRunner.execute`）：

```
Step 0  若配置了 config.dvc → DVCLoader.pull() 拉取数据并注入版本元信息
Step 1  load_data()
Step 2  Optuna study：按搜索空间采样，逐个 trial 调 train()，支持剪枝
Step 3  选 best trial（pareto 前端 / select_best_trial 钩子）
Step 4  export_model(artifact, best_dir) + on_study_end() + 写 study_summary.json
        （可选）MLflowRunLogger 上报 + on_model_registered()
```

### 4.3 `TrainConfig` 关键字段

| 字段 | 默认 | 说明 |
|---|---|---|
| `n_trials` | env `N_TRIALS` 或 100 | trial 数 |
| `direction` | env `OPTUNA_DIRECTION` 或 maximize | 优化方向 |
| `n_jobs` / `timeout` / `pruner` / `storage` | 1 / None / median / None | Optuna 并行、超时、剪枝、存储 |
| `study_name` | 自动生成 | study 名 |
| `dvc_data_root` | `dvc_data` | 传给 `load_data()` 的数据根 |
| `export_formats` / `onnx_opset_version` | `["onnx"]` / 18 | 模型导出 |
| `log_level` | INFO | 日志 |
| `dvc` / `mlflow` | None | 可选集成配置（None = 不启用） |

配置优先级：**环境变量 > 显式传参 > 默认值**。

---

## 5. 可选钩子（按需覆写）

| 钩子 | 时机 |
|---|---|
| `report(step, value)` | `train()` 内部阶段性上报，供剪枝 |
| `export_model(artifact, dir)` | 导出产物到磁盘 |
| `select_best_trial(pareto_front)` | 自定义 best trial 选择 |
| `on_study_end(study, best_trial)` | study 结束后回调 |
| `before_mlflow_log(result)` | MLflow 上报前修改结果 |
| `on_model_registered(result)` | 模型注册后（部署/通知） |
| `predict(model_path, inputs)` | 推理验证 |

---

## 6. 在我们「食材体积测量」场景的用法

我们不是训练神经网络，而是**对 IM 算法的参数做校准/消融**——这正是 het-ai 通用性的体现：

- **真值**：每个食材点云用 C++ 接口（参考配置）算出的体积，离线存在 `ground_truth.json`；
- **训练样本**：`load_data()` 里用 `load_pcd()` 把点云载入内存，样本 = 「点云 + 真值体积」；
- **搜索空间**：`voxel_size_m / plane_distance_threshold_m / integration_resolution_m / min_height_m`；
- **目标函数**：`train()` 用一组超参跑 `VolumePipeline().measure()`，返回「预测 vs 真值」的平均相对误差，方向 `minimize`；
- **意义**：找到「能把参数放宽（更粗、更快）到什么程度，体积结果仍不偏离参考」——即步骤/参数的敏感度与可省略性分析。

```
make_ground_truth.py    →  离线生成 ground_truth.json（真值表）
het_ai_volume_trainer.py →  BaseTrainer：load_data 读表载点云 / train 调库算误差
                            python ...  (dry_run 快速验证)  /  --run (完整 HPO)
```

---

## 7. het-ai vs 直接用 Optuna

het-ai 并没有发明新优化算法，它只是把 Optuna 包了一层**约定式脚手架**：

| 维度 | 直接用 Optuna | 用 het-ai |
|---|---|---|
| 搜索空间 | 手动 `trial.suggest_*` | `@search` 装饰器 + `Tunable*` 类型 |
| 数据加载 | 每次自己写 | 约定 `load_data`/`mock_data` |
| 结果导出 | 自己写 | `export_model` + `TrainResult` |
| DVC/MLflow | 自己接 | 配置即用（可选） |
| dry-run | 自己写冒烟 | `dry_run()` 一行 |

**适用场景**：有多个小项目/实验要跑 HPO、希望统一的工程骨架、要接 DVC/MLflow 平台。

**不适用场景**：只想在一个脚本里快速调一次 Optuna、需要完全自定义的 study 生命周期、不想引入框架约定。

---

## 8. 一句话总结

het-ai = **「子类化 BaseTrainer + @search 声明搜索空间 + 返回分数」的 Optuna 编排框架**，附带 DVC 数据版本化与 MLflow 实验追踪；它把 HPO 的工程样板固化为约定，让你只关心「数据」和「目标函数」。
