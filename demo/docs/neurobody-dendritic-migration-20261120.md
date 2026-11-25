# NeuroBody 神经元核心迁移与失败复盘

实验日期：2026-11-20。公开整理日期：2026-11-25。采用树突＋慢适应速率核心。

## 当前状态

NeuroBody 已调用公共 DendriticDynamics.advance；胞体、树突、适应三状态跨 tick 延续，episode 重置，TBPTT 边界一起 detach。训练、评估和 checkpoint 复载可运行；**前端未接入新版模型**。旧服务 metadata 与 tracking clip 不一致，本机 VRM 缺失，未生成新版展示或截图。

本目录公开离线源码和轻量结果，不公开 viewer、个人 clip、模型权重、VRM、AMASS 数据及本机配置。许可沿用根目录声明，已有第三方权利不变。

## 上次失败的证据

1. 随机目标任务：旧 vNext 末段姿态误差 0.10941、goal drift 0.06830；plain 为 0.04369、0.00024，PD 为 0.05518、0.00827。连续目标更容易：vNext 0.02964、plain 0.02831。[旧评估记录](../results/dendritic_migration_20261120/historical/) 保留了相应 JSON。失败主要出现在随机切换场景，但缺少单变量/多训练 seed 消融，不能归结为一个唯一原因。
2. 旧规划辅助实验没有改善主控制指标，预测头并未成为闭环动作选择器；辅助预测 loss 不应替代控制效果。
3. error_to_motor=False 曾只屏蔽误差图连接，却保留 motor readout 的直接误差项。针对旧 checkpoint 的固定状态诊断，去掉该旁路使平均关节速度输出变化 0.0658（最大 0.0894）。现已让 flag 同时控制两条路径，并测试关闭旁路后的行为。
4. 旧 NeuroBody 更新没有新版核心的独立树突低通状态和慢适应状态；此前的同名 neurosim 分支不能当作新版动力学。现已接到公共方程，参数仍是示例而非生物或动作任务拟合。
5. 旧随机目标 checkpoint 约 48.65% 原始边权符号不符合 source 类型。新树突分支按 source 类型约束实际传播符号；这是模型约束，不是性能提升证据。
6. 旧 checkpoint 为 6,020 神经元，后来配置默认扩到 6,604；按种子重建图还会因合并重复边而改变边数和拓扑。loader 现推断旧扩展池为零并恢复存储的 CSR 图；跨种子和旧配置测试覆盖此问题。
7. 训练历史没有显式固定 PyTorch 全局 RNG；runner 现固定 CPU/CUDA seed。新旧历史不是完全同版本的因果对照。
8. point＋Dale 初版 checkpoint 元数据曾把约束标志覆盖为 false。已经修复并重新评估；元数据修复前的 horizon 文件与旧 checkpoint 副本不参与公开匹配结果。

这些是局部诊断观察；旧 checkpoint 本身不随仓库发布，无法仅凭公开 JSON 重新执行固定状态旁路/符号诊断。公开的 [provenance](../results/dendritic_migration_20261120/provenance.json) 记录本机 checkpoint 和原始源码哈希。

## 实现与评估口径

新核心以胞体活动驱动动作；树突和慢适应状态在 tick 间持续，三个状态一起 detach。Dale 约束使用源类型固定符号乘可学习幅度。checkpoint 精确恢复 CSR 拓扑、核心参数和独立 Dale 标志。标量广播缓冲区在注册前 clone，避免零 stride 导致 load_state_dict 写入失败。

三种网络均为 goals/random、dwell 150、6,000 train ticks、batch 8、seed 0、6,020 神经元；M1→motor=8、reflex gain=0、lr=1e-3，TBPTT 更新 750 次。训练 episode 长度 450。评估 4,500 ticks × 3 episodes，seed base 2000，每 episode 固定 CPU/CUDA 噪声 RNG。GPU 为 RTX 5070 Laptop，Windows、Python 3.13.0、PyTorch 2.13.0.dev20260514+cu132、CUDA 13.2，FP32。每个训练图的精确边数见各条件 eval.json 的 n_edges。

postureError 为 episode 最后 30 tick 的平均姿态误差；goalDrift 为最后一个 dwell 窗口误差减第一个窗口误差，不是 root 位移。只有一个训练 seed，表中 std 是评估 episode 的标准差。

| policy | 姿态误差 | goal drift | 姿态误差 std | drift std |
|---|---:|---:|---:|---:|
| dendritic＋Dale | 0.08947 | 0.02868 | 0.04679 | 0.04872 |
| point vNext | 0.08671 | 0.04579 | 0.05114 | 0.05307 |
| point vNext＋Dale | 0.10477 | 0.04952 | 0.05351 | 0.05630 |
| PD | **0.05376** | **0.00496** | 0.00363 | 0.00288 |

[匹配结果 JSON](../results/dendritic_migration_20261120/matched_comparison.json) 为有效版本。三种网络都落后 PD；新核心可训练、可复载，但尚不能称为性能修复。dendritic 对 point 同时改变动力学和 Dale，point＋Dale 是额外控制条件；仍需多训练 seed 和单因素消融。

[时域诊断](../results/dendritic_migration_20261120/eval_horizon_diagnostic.json) 每个 horizon 独立重置，不是同一条不断延长的轨迹；长时域 episode 方差较大，不应从一个均值宣称长期稳定。

## 记录、复现和剩余工作

[各条件配置与日志](../results/dendritic_migration_20261120/) 保留 November 20 的历史训练记录。本次公开整理没有重新训练那三组 6,000-tick 模型。[使用入口](../README.md) 给出重新训练、长评估及 CPU smoke 命令；新输出写入忽略的 demo/runs/，不会混入归档数据。复现实验需要安装该子项目依赖。

November 20 验证：vNext 单测 18 通过，核心回归 30 通过，py_compile 通过；本机完整 demo 26 项中 **22 通过、1 失败、3 跳过**，失败为旧 tracking clip 契约。[捕获记录](../results/dendritic_migration_20261120/verification_log_20261120.json) 保留命令、输出摘要和退出码；先前报告把跳过项误算入通过项，公开版已改正。

November 25 基于 GitHub 最新 main 的验证单独写入 [publication_validation.json](../results/dendritic_migration_20261120/publication_validation.json)，不与历史 GPU 结果混算。复现命令以该文件和 README 为准。

下一步优先恢复获准使用的本机 VRM，再接 viewer/clip 并截图；随后补多 seed、动力学/适应/Dale/误差信号消融，最后才考虑含重力与接触的物理任务。当前 servo、无视觉结果不证明步行或平衡。
