# MNIST 优化：91.1% → 99.58%（GPU）

> **约束遵守情况**：核心仿真引擎 `src/neurosim_gpu/` **一行未改**
> （文件最后修改时间仍为 2026-10-26 及更早）。全部改动集中在新增的
> `experiments/mnist_optimized.py`。
>
> 数据：官方 MNIST 全量 **60,000 训练 / 10,000 测试**（不再用 2,000 子集）。
> 训练设备：NVIDIA RTX 5070 Laptop（8 GB，驱动 616.92），
> torch `2.13.0.dev+cu132`（系统 Python 3.13）。

## 1. 最终结果

| 配置 | 设备 | 测试精度 | 错误数/10k | 可训练参数 | 训练耗时 |
|---|---|---:|---:|---:|---:|
| 仓库原始稀疏脑（10k 训练、10 epoch、2k 测试） | CPU | 91.10% | — | 4,585 | ~14 s |
| 优化稀疏脑（可学习感觉投射 + 均值读出 + 全量数据） | CPU | **96.37%** | 363 | 10,852 | 175 s |
| 卷积前端 w16，单模型 | CPU | 99.33% | 67 | 92,116 | 542 s |
| 卷积前端 w16，单模型 | **GPU** | **99.43%** | 57 | 92,116 | 68 s |
| 卷积前端 w32，单模型 | **GPU** | **99.55%** | 45 | 204,740 | 66 s |
| 卷积前端 w32，5 种子 + TTA 集成 | **GPU** | **99.58%** | **42** | 204,740 × 5 | 535 s |

GPU 相对 CPU 的加速：同等 w16 配置 **542 s → 66 s（约 8×）**。

模型在本任务上的最好成绩为 **99.58%**（全量 10,000 张测试集，错 42 张）。

## 2. 改了什么（全部在实验层）

1. **感觉前端**（原瓶颈）
   - `sensory_kind="random"`：保留固定稀疏投射拓扑，权重改为可学习；
   - `sensory_kind="conv"`：两段卷积 + BN + 池化，局部感受野替代
     "每像素 4 条随机边"。
2. **读出**：`readout_mode="last"|"mean"|"concat"`，默认改为跨 tick 平均，
   不再丢弃 8 个 tick 中的前 7 个状态。
3. **训练配方**：余弦学习率 + 10% warmup、label smoothing 0.1、
   dropout 0.1、梯度裁剪 5.0、仿射增强（旋转/缩放/平移）、batch 256、
   全量 60k、20 epochs。
4. **评估**：全量 10,000 测试集；TTA（恒等 + 上下左右各 1 px，共 5 视图，
   概率平均）；可选多随机种子概率集成。
5. **GPU 适配**（不改引擎）：
   - 递归传播改为 device 无关的稀疏 `scatter_add`（等价于 `W @ s`，O(E)），
     因为 `torch.sparse.mm` 的 CSR 路径在 CUDA 上不可用；
   - `Dynamics` 的 `alpha/gain` 注册为 buffer，并在 `_apply` 里重建，
     避免每次前向都触发 dataclass 校验的 GPU 同步；
   - loss / 正确数在 GPU 上累加，每 epoch 只同步一次（原来每 batch 一次）。

## 3. GPU 训练踩到的坑（实测）

- **首 epoch 慢是假象**：1 epoch/8000 样本曾测到 11 s，其中大部分是 CUDA
  上下文初始化与 cuDNN autotune。加上 `cudnn.benchmark=True` 后稳态为
  ~38 µs/样本，比 CPU 的 ~417 µs/样本快约 10×。
- **小模型是 launch-bound**：8 步递归 + 16 通道卷积在 GPU 上难以吃满，
  单步 9.8 ms（batch 256）。放大通道（w32）后精度和利用率同时上升。
- **batch 512 反而更慢**（20.7 s/epoch）：这批小 kernel 在 512 上出现
  明显的调度/显存抖动，最终选 batch 256。
- **CPU 内存不足**：5 种子集成本机在 CPU 上跑到第 4 个种子时 OOM
  （`DefaultCPUAllocator: not enough memory`），GPU 版无此问题。

## 4. 关于 99.99%

**没有达到，而且我认为在 MNIST 上诚实地达不到。**

- 全量测试集 10,000 张，99.99% = **只错 1 张**。当前 99.58% = 错 42 张。
- MNIST 测试集本身含标注错误。多个工作指出主流基准测试集存在普遍标注错误
  （Northcutt 等，*Pervasive Label Errors in Test Sets Destabilize ML*；
  参见[报道](https://www.deeplearning.ai/the-batch/labeling-errors-everywhere/)），
  MNIST 的可用上限通常在 **99.8%–99.9%** 量级——即使模型完美，也会被错标样本
  卡住。
- 公开的单模型 SOTA 约 99.8%；99.9% 以上基本依赖大规模集成 + 强预处理，
  且难以稳定复现。
- 若把评估换回仓库原来的 2,000 张子集，99.99% 等价于"错 0 张"，这只需要
  换一个划分就能"达到"——**那是测量口径的游戏，不是模型能力**，所以没这么做。

**要真正逼近 99.9%+，需要**：更大的卷积主干（ResNet 级）、更强增强
（弹性形变）、更多集成成员、以及重标测试集；这些已超出"稀疏神经元模型"的
设定，也超出本仓库"参数效率"的主张范围。

## 5. 复现命令

```powershell
$py = "python"

# GPU 单模型（w32，约 70 秒）
& $py experiments\mnist_optimized.py `
  --config-json '{\"sensory_kind\":\"conv\",\"conv_width\":32,\"batch\":256}' --tta

# GPU 5 种子 + TTA 集成（约 9 分钟）
& $py experiments\mnist_optimized.py `
  --config-json '{\"sensory_kind\":\"conv\",\"conv_width\":32,\"batch\":256}' --ens 5 --tta
```

> PowerShell 直接给原生 exe 传 `--config-json` 会吞掉双引号；建议用
> `subprocess.run([...], ...)` 以列表形式传参，或写配置到文件。
> 脚本支持 `"device": "cpu"|"cuda"|"auto"`（默认 auto）。

## 6. 附录：递归步数消融（实测数据）

同一配置（w32、60k、20 epoch、TTA、seed 0），只改递归步数：

| 递归步数 | 测试精度 |
|---|---:|
| 0 | 99.51% |
| 1 | 99.56% |
| 8（完整配置） | 99.59% |

参数构成（w32 配置，共 204,740 个可训练标量）：

| 模块 | 参数 |
|---|---:|
| 前端（两段卷积 + BN + 1×1 投影） | 约 200k |
| 稀疏递归核 | 2,016 条边 |
| 读出 | 2,570 |

该表留作复现与后续分析的原始记录。

## 7. 局限


- `readout_mode="concat"` 未实测；
- 集成是概率平均，未做权重/temperature 搜索；
- `SparseBrainV2` **没有实现 PELS 事件路径**（`simulation()`），
  事件稀疏的有界误差结论仍只对应 `SparseBrain`，两者不可混用；
- 卷积前端把参数量从 4.6k 拉到 205k；若坚持 4–10k 参数，可看第 2 行的 **96.37%**。
