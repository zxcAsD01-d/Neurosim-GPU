# 0.2.0 树突神经元性能与核验

2026-11-17。此次优化对象是 `DendriticDynamics` 的完整 tick，包含稀疏传播、
树突低通/饱和、胞体更新和慢适应。比较双方使用同一拓扑、参数、输入和 FP32。
原点神经元的历史性能表不作为此次加速比的分母。

## 测量环境与协议

- NVIDIA GeForce RTX 5070 Laptop GPU，8 GiB，驱动 616.92。
- Windows，Python 3.13.0，PyTorch `2.13.0.dev20260514+cu132`。
- CUDA toolkit 13.1 编译，PyTorch 报告 runtime 13.2，MSVC 14.44。
- `torch.inference_mode()`，CPU 线程数 1；拓扑 seed=23。
- 每路径预热 30 ticks；7 轮，每轮 100 ticks；每轮随机交错路径顺序。
- 报告同步块 wall-time 的每 tick 中位数；JSON 同时保存 CUDA event 时间与全部样本。
- 编译、拓扑构建和 profiler 都不在计时窗口。内存指标另行测量。
- 输入随时间变化：`drive=0.2+0.1*sin(tick*0.11)`，`dendritic_drive=0.1`。
- `alpha=.72, gain=.8, dt=1, tau_d=4, tau_a=25, coupling=.5, strength=.2`。
- PELS 双方 `theta=.02`。参考 PELS 用矩阵传播发射增量；本次加速比不等同于
  相对原点神经元 native event kernel 的加速比。

```powershell
$env:PYTHONPATH="src"
scripts\native_env.bat python benchmarks/compartmental.py --large --profile --output results/compartmental_performance.json
```

初步探索曾使用不同 CPU 线程环境得到更高的参考耗时；发布只采用上面的
同进程、交错测量结果。笔记本 GPU 功耗、温度及系统负载会影响绝对值。

## 实测结果

单位为毫秒/tick；M 是去重后的真实边数。

| N | M | B | 参考矩阵 | 融合矩阵 | 矩阵加速 | 参考 PELS | 原生 PELS | PELS 加速 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 65,416 | 1 | 0.284 | 0.012 | 23.84x | 0.355 | 0.021 | 17.27x |
| 50,000 | 999,787 | 1 | 0.327 | 0.063 | 5.19x | 0.423 | 0.041 | 10.20x |
| 50,000 | 999,787 | 8 | 0.671 | 0.397 | 1.69x | 0.747 | 0.184 | 4.06x |
| 166,000 | 9,998,214 | 1 | 0.769 | 0.489 | 1.57x | 0.848 | 0.347 | 2.44x |
| 166,000 | 24,988,704 | 1 | 1.444 | 1.094 | 1.32x | 1.541 | 0.964 | 1.60x |

小图收益主要来自减少 kernel 启动，大图收益受稀疏访存限制。没有外推到 125M 边、
其他 GPU、多 GPU、FP64 或训练。计时后额外 16 ticks 的活动探针显示约 18.59–18.61% 的边对应非零发射增量；
该比例来自计时后的连续输入窗口，不是整段计时的精确活动均值。原始数据含源文件 SHA256：
[performance JSON](../results/compartmental_performance.json)。

Profiler 在第一组配置观测到矩阵路径 **29→1 kernel/tick**、PELS **36→2**。
单 tick 额外峰值分配：矩阵 114,688→49,152 bytes（约减少 57%），PELS
98,304→81,920 bytes（约减少 17%）。1M 边/B=8 时矩阵为
11,200,000→4,800,000 bytes。指标是 allocator 的瞬时增量，不是整个模型显存。
三组输出共享一块新分配的存储，但不会在后续 tick 被覆盖；保留旧输出仍然有效。

Profiler trace 保留在结果目录中，供本机诊断；公开汇总只保留 kernel 名称和计数，
不发布含主机文件路径的完整 trace。

## 实现

`NativeDendriticMatrixRuntime` 用一个 kernel 完成 CSR 行归约及全部分区更新。
根据平均入度选择 8/16/32 lanes，允许手工指定 `threads_per_row`；这是一条启发式，
不是所有图的最优配置。参数直接按原 tensor stride 读取，支持广播和非连续驱动，
不会缓存一份失去梯度或不能反映修改的参数副本。

`NativeDendriticPELSRuntime` 第一 kernel 生成新的胞体、树突、适应和发射状态，
第二 kernel 在设备上扫描源、跳过零增量并仅访问有发射源的出边。两阶段的 stream
顺序确保读完旧电流后才执行原子散射。每 tick 没有 CPU 读取活动数或 Python 逐事件循环。
原子加法存在 FP32 非确定性。

所有新 kernel 使用 state 所属设备的 current stream，构造及调用检查 shape、dtype
和设备；GPU 数值有效性仍遵循有限输入契约。完整参数保持 live，外部权重版本更新
会触发 PELS 当前值重建。非零发射阈值不是整个动力系统的误差界。

```python
from neurosim_gpu import NativeDendriticMatrixRuntime, NativeDendriticPELSRuntime

# graph/model/initial 与研究文档示例相同，必须是 CUDA float32。
matrix = NativeDendriticMatrixRuntime(graph, model, initial)
events = NativeDendriticPELSRuntime(graph, model, initial, theta=0.02)
state = matrix.step(drive=0.2, dendritic_drive=0.1)
```

有任一状态、参数、权重或驱动需要梯度时，自动回退可微 PyTorch 路径，
`last_backend` 标记实际路径。`collect_stats=True` 回退参考 PELS。
没有宣称训练加速；没有更换默认模型或修改 Cortex-LM。

## 误差与边界

计时滚动结束后，融合矩阵相对参考矩阵的状态最大绝对误差为
`4.47e-7–5.96e-7`。非零阈值原生 PELS 相对同阈值参考 PELS 为
`5.27e-4–5.59e-4`；相对不丢弃增量的精确矩阵约 `0.00386–0.00438`。
这些都是指定输入下的观测值，不是普适误差保证。

首轮基准用 `atol=2e-4` 比较两条非零阈值路径时失败（最大误差约 `5.41e-4`）。
分析确认浮点舍入可改变接近阈值的发射时刻；发布基准将该项作为经验验收，
明确使用绝对 `atol=2e-3` 并保存实际误差。零阈值轨迹和梯度测试仍采用更严格容差，
没有把这次放宽用于矩阵路径或零阈值检查。

## 验证

发布分支与原工作区的测试数量不同：原工作区包含未发布的海马体实验。
本分支从 GitHub main 开始，只带入神经元更新与本次优化，完整发现 **204 项测试**。

| 检查 | 结果 |
|---|---|
| 完整 suite，CUDA + 原生测试显式启用 | 204 项通过，无跳过 |
| 相同环境屏蔽 CUDA | 204 项执行，12 项硬件相关跳过，其余通过 |
| 独立 CPU-only PyTorch 2.14.0，缺 NumPy/transformers | 204 项执行，23 项可选项跳过，其余通过 |
| 旧 `benchmarks/validate_native.py` | PASS |
| 新原生矩阵/事件 500-tick 轨迹、三种 lane 宽度、非连续/广播输入 | PASS |
| 空边/空 batch、batch 隔离、参数修改、输出不被覆盖、外部权重更新 | PASS |
| 状态/参数/驱动/权重梯度回退、断点、非默认 CUDA stream | PASS |
| wheel 与源码包版本、CUDA 源码逐字节检查 | PASS |
| 已安装 wheel 的原生源码编译及矩阵/事件 smoke | PASS |
| Compute Sanitizer memcheck 与 synccheck | 各 10 项测试通过，均为 0 errors |

测试与 sanitizer 的精确命令、发布最终检查见 [verification](verification.md)。
没有进行真实任务重训或生物数据拟合，模型复杂度提升不自动等于任务性能提升。

---

[← Documentation index](index.md)
