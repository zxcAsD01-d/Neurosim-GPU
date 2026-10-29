# 核心引擎检查（2026-10-30）

> 范围：`src/neurosim_gpu/`（dynamics、runtime、topology、dopamine、neurochem、chemistry）。
> 方法：跑完整测试套件 + 针对今日实验暴露的疑点做定点验证。

## 1. 测试套件

```
PYTHONPATH=src python -m unittest discover -s tests
Ran 107 tests in 110.170s
OK
```

**全部通过。** 注：直接跑会因 `neurosim_gpu` 不在解释器路径上而报 8 个 import 错误，
必须设 `PYTHONPATH=src`（或装 editable 包）。这不是引擎问题，但会误导首次使用者。

## 2. 定点验证：权重更新后是否使用陈旧矩阵

`MatrixRuntime.__init__` 把 `topology.sparse_matrix()` 缓存到 `self.weight`，
之后 `DopamineMatrixRuntime` 通过 `apply_source_weights_` 原地更新权重。
若 CSR 不共享存储，传播就会用旧权重——这是一个潜在严重缺陷。

实测：

```python
before = runtime._propagate(state)
topology.apply_source_weights_(topology.weight * 2.0)
after = runtime._propagate(state)
max|after - 2*before| == 0.0
```

且 `runtime.weight.values().data_ptr() == topology.in_weight.data_ptr()` 为真，
确认 CSR 与 `in_weight` 共享存储。**结论：无陈旧权重问题。**

不过这一点依赖"原地写"这一隐含约定；若将来 `apply_source_weights_` 改成
重新赋值 `self.in_weight = ...`，缓存就会失效。建议加回归测试固定该行为。

## 3. 发现的问题

### 3.1 缺少稳态机制（设计缺口，影响最大）

引擎没有任何活动/权重归一化。实测默认 `weight_scale=0.18` 下：

- 脑区状态均值 0.018、eligibility 0.0074、单次权重变化 ~2e-6；
- 2000 个样本训练后权重范数 5.150 → 5.150，**规则实际未生效**。

学习率必须与权重尺度联合标定，但 `DopaminePlasticityConfig` 里没有这个约束，
也没有自动调节。建议：加权重归一化/突触缩放，或在文档中强制标定流程并给出
`weight_scale → eligibility` 的参考曲线。

### 3.2 默认随机权重尺度偏小

`random_topology` 使用 `weight_scale / sqrt(mean_degree)`，默认 0.15–0.18。
在扇入 8 的稀疏核上，单权重约 0.06，配合 0.1 量级的活动，
净输入只有 0.06，系统长期停在低活动区（见 3.1）。

### 3.3 硬钳位死区（性质，不是缺陷）

`s' = clamp(αs + (1−α)g·tanh(...), 0, 1)` 导致约 **51.7%** 的神经元-时刻
落在 `pre ≤ 0`（导数为 0、`post = 0`）。

对照实验（soft-clamp 臂）显示：大预算下引擎 **38.4%** vs 对照 **6.6%**，
即硬钳位带来的稀疏性是有益的。**记录为性质，不建议改。**

### 3.4 GPU 同步点

- `Dynamics.__post_init__` 用 `bool(torch.any(...))` 校验，每次构造同步一次；
- `validate_inputs=True` 每一步 `step` 都同步。

研究模式可接受（文档已说明），但在循环里频繁构造 `Dynamics` 会明显拖慢。
建议缓存校验结果或在训练循环外构造。

### 3.5 `Dynamics` 是持张量的 frozen dataclass

`module.to(device)` 不会移动它（它不是 buffer）。今日实验中我遇到并绕开过：
需要在 `_apply` 里重建。建议文档里给出这一集成写法，或提供 `Dynamics.to()`
辅助方法。

### 3.6 精确可塑性限 B=1

`batch != 1` 且 `learning_rate > 0` 时构造直接报错，这是有意设计（文档已写明）。
但它意味着**批量训练必须走近似路径**，而近似路径的结论不能当精确结论用——
今日已因此作废过一版实验结论。

### 3.7 拓扑合并重复边

`from_edges` 会合并重复 `(src, dst)`，实际 `n_edges` 与标称 `N*d` 略有差异
（如 2010 vs 2016）。文档有提及，但报告参数时应使用实际值。

## 4. 结论

- **引擎实现本身没有发现功能性缺陷**：107 个测试通过，关键路径（权重更新后的
  传播一致性）也验证无误。
- 今日暴露的问题集中在**设计层面**：缺稳态、权重尺度默认值偏小、
  精确路径限 B=1。
- 之前怀疑的"钳位死区是底层缺陷"**已被对照实验否定**。
