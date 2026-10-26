# NeuroSim GPU 中文说明

GPU 优先的稀疏神经仿真原型：固定拓扑神经核 + 持久突触电流（PELS）事件传播 +
区域多巴胺化学 + 资格迹可塑性 + 多速率调度。

对 `N` 个神经元、`M` 条固定突触、批大小 `B`，一个 tick 就是一次稀疏矩阵乘法
加少量向量运算：

```text
synaptic_current = W_csr @ state
state = clamp(alpha * state + (1 - alpha) * gain * tanh(current + drive))
```

时间 `O(BM + BN)`，内存 `O(M + BN)`。化学、延迟、资格迹和结构发育各有独立的
状态、节奏与成本记录，详见[架构](docs/architecture.md)与[成本模型](docs/efficiency.md)。

## 快速开始

```powershell
python -m pip install -e .
python -m unittest discover -s tests -q     # 107 个测试；无 CUDA 时跳过 1 个
```

第一个 tick（纯 CPU 即可）：

```python
import torch
from neurosim_gpu import Dynamics, MatrixRuntime, random_topology

topology = random_topology(1000, 20_000, device="cpu", seed=0)
dynamics = Dynamics(
    alpha=torch.full((1000,), 0.72),
    gain=torch.full((1000,), 1.05),
)
runtime = MatrixRuntime(topology, dynamics, torch.zeros(1, 1000))
for _ in range(10):
    state = runtime.step(drive=0.1)
print(topology.n_edges, float(state.mean()))
```

更多内容：[上手指南](docs/getting-started.md) · [可运行 API](docs/api.md)。

## 运行时

| 运行时 | 路径 | 说明 |
|---|---|---|
| `MatrixRuntime` / `DopamineMatrixRuntime` | 可移植 PyTorch CSR | 参考实现，便于阅读与对照 |
| `HybridPELSRuntime` / `DopaminePersistentRuntime` | 持久突触电流 | 只传播状态增量；参考实现的增量散射仍是完整 SpMM |
| `NativeCanonicalMatrixRuntime` 等 | 融合 CUDA | 单权重数组 + 目的端归约 |
| `NativeEventPELSRuntime` 等 | 融合 CUDA 事件路径 | 低活动时传播为 `O(BN + M_A)` |

原生路径需要 CUDA 版 PyTorch，首次调用时编译，见[原生构建](docs/native-build.md)。
注意：启用 eligibility 时多巴胺运行时每个 tick 仍扫描全部 `M` 条边，整 tick 的
事件稀疏只在关闭 eligibility 且该 tick 不做权重更新时成立。

## 已测结果

带硬件、图规模与命令的正式数字见[结果](docs/results.md)，解读见[成本模型](docs/efficiency.md)。要点：

- 固定核前向（B=1）：10–25M 边时为同参数稀疏核成本的 **0.74–0.82×**；
- 低活动事件传播：活跃边约 2.6% 时比矩阵路径快 **3–4.5×**；
- MNIST 上的发放阈值：**21% 的边活跃，精度损失 0.12 个点**（仅前向传播）。

任务级实验（MNIST 手写数字、倒立摆稳定控制）见[实验说明](docs/experiments.md)，
完整报告在 [results/REPORT.md](results/REPORT.md)：

| 实验 | 结果 |
|---|---|
| MNIST：稀疏脑 vs 同参数 MLP | 91.1% vs 87.3%（4,585 个可训练标量） |
| MNIST：稀疏脑 vs 逻辑回归 | 91.1% vs 91.8%——精度上没有优势 |
| 倒立摆：行为克隆 | MLP 与稀疏脑都是 200/200 步；只有极端延迟/噪声才失败 |
| 执行器突变下的在线三因子可塑性 | 无可测量收益；消融显示学到的递归权重近乎可省 |

## 文档

从 [docs/index.md](docs/index.md) 开始（含阅读路径与文档权威表）：

| 文档 | 内容 |
|---|---|
| [上手指南](docs/getting-started.md) | 安装、第一个 tick、常见问题 |
| [API](docs/api.md) | 公共工作流、张量契约、错误 |
| [设计](docs/design.md) · [架构](docs/architecture.md) | 数值契约、状态归属、运行时模式 |
| [生物学语义](docs/biology.md) | 神经化学模型代表什么、类比在哪里停止 |
| [成本模型](docs/efficiency.md) · [结果](docs/results.md) · [验证](docs/validation.md) | 成本公式、已测数字、验收标准 |
| [实验](docs/experiments.md) · [五子棋协议](docs/gomoku-protocol.md) | 任务级实验与协议 |
| [状态](docs/status.md) · [交付验证](docs/verification.md) | 里程碑记分卡、发布验证记录 |

## 边界声明

本项目是**研究原型**，不是经过生物学验证的脑模型。它不主张生物真实性、全脑
规模或自博弈强度；所有性能数字都与记录的硬件和图规模绑定。引用前请先读
[scope](docs/scope.md)。实验中的行为克隆不是强化学习，CPU 上的精度/延迟结论
不能外推到 GPU 大图。

## 贡献与许可

欢迎贡献，请先读 [CONTRIBUTING.md](CONTRIBUTING.md) 与[验证标准](docs/validation.md)；
提交前请保证测试全绿、文档链接可解析（`python scripts/check_links.py`）。

采用 Apache License 2.0，见 [LICENSE](LICENSE)；引用信息见 [CITATION.cff](CITATION.cff)。
