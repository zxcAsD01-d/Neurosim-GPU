# Cortex-LM 研究报告

**日期**：2026-10-31  
**项目**：`neurosim-gpu/llm`  
**状态**：研究原型；可复现训练和推理，但不是可用对话系统。

## 摘要

本报告记录一个“冻结 Qwen3 两端、用神经元中段替换中间层”的语言模型实验。
Qwen3-0.6B 的 28 层保留前 3 层和后 2 层，删除中间 23 层，换成一个 1024 神经元
的递归核。中段 W 用被删层的线性化残差映射初始化，Qwen 保留权重冻结，只训练
中段和最后一层的 LoRA。

20k 步训练后，dense 模型的 eval perplexity 为 **22.69**，能生成通顺中文句子，
但采样跑题，不是对话模型。之后依次尝试空间迁移/重连、损失驱动重连、双 tick
时序、慢记忆、BOS/EOS 加权、短时突触资源、eligibility trace、多巴胺巩固、
homeostatic scaling 和开环一致性损失。stage3（生物启发记忆）在同配置 8-batch
上达到 **21.49**，比 stage2 的 22.31 好约 3.7%，但仍未超过 dense 的 20.46。
stage4 的 8 步零 drive 开环损失让 ppl 退化到 **22.15**，是负面结果。

诊断表明：脑是唯一主通道（去掉脑 ppl 约 7 万），但脑读出有效秩只有 56.5/1024；
开环解码的失败主要不是边数，而是状态秩和 teacher-forced 训练与无反馈推理之间
的 exposure bias。

## 1. 目标与范围

目标是验证一条管线，而不是宣称语言能力：

1. 复用 Qwen3 的文本编码器和解码器；
2. 在两者之间放一个可训练的神经元递归核；
3. 用 Qwen3 被删层的权重初始化中段；
4. 在真实文本上训练、采样、做消融；
5. 研究迁移、重连、时序、生物启发记忆和开环读出。

不在范围内：真实生物神经元、全脑模拟、指令微调、RLHF、多模态、生产级推理。

## 2. 架构

### 2.1 3+2 切分

| 部分 | Qwen3-0.6B |
|---|---|
| 编码器 | 前 3 层，冻结 |
| 删除 | 中间 23 层 |
| 解码器 | 后 2 层，冻结；LoRA rank 32 注入 q/v/o/down/up |
| 中段 | 1024 神经元，dense W，可训练 |
| 可训练参数 | 约 4.2M（中段 3.15M + LoRA 1.05M） |

### 2.2 中段动力学

```text
current = state @ W.T
state   = clamp(alpha * state + (1-alpha) * gain * tanh(current + drive))
readout = readout_proj(LayerNorm(state))
```

- 1024 个单房室 rate neuron；
- `alpha` 0.90–0.97；
- 每 token 2 个 tick：第一个 tick 内部“思考”，最后一个 tick 读出；
- 状态跨 token 保留，block 边界重置；
- TBPTT 每 32 token detach。

### 2.3 权重移植

对被删除的每一层构造线性化残差矩阵：

```text
B_l = o_proj @ repeat(v_proj) + down_proj @ up_proj
A   = Π (I + B_l)
```

对 A 做谱归一化到半径 0.95，作为 W 的初始化。这个映射忽略 softmax、QK、gate、
SiLU 等非线性，是**线性化近似**，不是逐层等价。

### 2.4 空间几何

- 前极板 `z=0`，后极板 `z=50`；标称平均神经元距离 `d̄=1`；
- 方形板边长 `sqrt(N·d̄³/D)=4.525`；
- 神经元在板间均匀分布；前极板只编码输入，后极板只解码；
- 连接：每个 source 保留 64 条 `|W|·exp(-distance/5)` 最强的边；
- 65,536 / 1,048,576 = **6.25%** dense 连接。

位置和 mask 是 `persistent=False` 的 buffer；checkpoint 只保存可训练参数，加载
时按 seed 重新采样位置、按 W 重新 top-k。因此当前图是**加载后重建**的结构，
不是训练轨迹中精确的迁移/拓扑。这是已知缺陷，下一阶段必须持久化。

## 3. 数据与标注

- 语料：MiniMind 预训练 JSONL，字段 `text`；
- 使用前 **100 MB** 子集，`block_size=128`，`batch=2`，`grad_accum=2`；
- 20k 步约 **10.24M token**（约 100 MB 子集的一半）；
- 文档之间插入 EOS；stage2 起加入 BOS，并对 EOS/BOS 使用加权交叉熵；
- 数据不进入仓库；配置使用相对路径 `data/minimind/pretrain_hq.jsonl`；
- 数据许可证/来源以 MiniMind 上游为准；仓库不重新分发语料；
- 没有做 PII 审计，不能声称语料无个人信息；
- 没有人工标注；损失是自监督 next-token CE。

数据统计（100 MB 子集）：

| 项 | 值 |
|---|---:|
| 训练 block | 156,140 |
| 验证 block | 3,191 |
| block size | 128 |
| 训练 token（20k 步） | 10.24M |

## 4. 训练结果

### 4.1 主结果

| 阶段 | 主要改动 | ppl |
|---|---|---:|
| 20k dense | 3+2、移植、LoRA | **22.69**（训练 eval，32 batch） |
| spatial 20200 | 64 边/神经元；迁移/重连 | 35.50 |
| spatial 20400 | 继续 200 步 | 25.72 |
| stage2 | 双 tick、慢记忆、噪声、BOS/EOS、loss-driven 重连 | 22.31（8-batch） |
| stage3 | 短时突触资源 + eligibility + 多巴胺巩固 + 稳态 | **21.49**（8-batch） |
| stage4 | stage3 + 8 步零 drive 开环损失 | 22.15（8-batch） |

同配置 8-batch 消融（stage3 配置）：

| 配置 / checkpoint | ppl |
|---|---:|
| stage3 配置 + stage3 ckpt | **21.49** |
| stage3 配置 + stage2 ckpt | 26.11 |
| legacy 配置 + stage3 ckpt | 26.46 |
| legacy 配置 + stage2 ckpt | 22.35 |

stage3 比 stage2 好约 3.7%，但收益依赖短时动力学：关掉 short-term/eligibility/
ticks 后，stage3 W 退化到 26.46。

### 4.2 诊断

4-batch 消融：

| 变体 | 20k dense | stage2 | stage3 |
|---|---:|---:|---:|
| 完整 sandwich | **20.46** | 22.71 | 21.96 |
| 去掉脑（只用 h_enc） | 70,308 | 70,482 | 67,386 |
| 只用脑读出（去掉 h_enc） | 59.09 | 69.88 | **60.84** |
| 解码器零输入 | 151,936 | 151,936 | 151,936 |
| 脑有效秩 | **60.9** | 52.8 | **56.5** |

结论：

- 脑是唯一主通道；去掉脑 ppl 升到约 7 万。
- 脑读出单独只有 ppl 59–70，信息不足。
- 有效秩 56.5/1024，是明显瓶颈。
- 边数不是主因：stage3 的 k=16→1024 扫描为 32.3→30.6，几乎平。
- stage2 的慢记忆把秩从 60.9 压到 52.8；stage3 的 eligibility 拉回 56.5。

### 4.3 开环读出

`stream_generate`：prefill 后前极板不再接收输入，脑每 tick 零 drive 更新，后极板
解码不断增长的 hidden，生成 token 不回灌。

20k 模型、prompt `中国的首都是`、12 token：

```text
哪王。，。.. .12.
```

stage4 加入开环辅助损失后，辅助 loss 从 11.6 只降到 10.9，主 ppl 从 21.49
退化到 22.15。结论：一次性 8 步零输入太难，需要课程式训练/反馈/单独输出头。

## 5. 推理样例

原始 next-token 推理，prompt `你好`：

| 模型/设置 | 输出 |
|---|---|
| stage3, T=0.8 | `。好的，我来帮你查一下明天天气预报吧。您需要帮我查询一下吗？` |
| stage3, greedy | `，我需要帮助你。请问您有什么问题或需求？` |
| 20k, T=0.8 | `。` |
| 20k, greedy | `，我需要知道您想了解哪个地区的天气预报呢？我可以帮你查询当地的天气情况。` |

多 prompt（stage3 greedy）：

| prompt | 输出 |
|---|---|
| `你是谁` | `？` |
| `今天天气怎么样` | `？` |
| `1+1等于几` | `倍？` |
| `再见` | `，我正在寻找一家新餐厅，他们的名字是“小明”和“小红”。` |

判断：能续写，不是可靠对话；`你好` 的客服式回答是语料模式触发，不是理解。

## 6. 当前结构图

![当前神经元结构、位置与连接](results/cortex/current_structure.png)

| 项 | 值 |
|---|---:|
| 神经元 | 1,024 |
| 突触 | 65,536 |
| 每神经元 | 64 |
| 连接密度 | 6.25% |
| 自环 | 1,022 |
| 正权重比例 | 34.9% |
| 平均边长 | 2.93 d̄ |
| 最长边 | 9.66 d̄ |
| 极板间距 | 50 d̄ |
| 板边长 | 4.525 |
| z 均值 | 25.28 |
| 实际最近邻 | 0.58 |

图是 stage3/step21,200 的**加载后重建**结构；不是训练轨迹中精确的迁移位置。
3D 面板是 `z=18–32` 的局部放大；完整几何看 side view。

## 7. 局限

1. **不是对话模型**：没有指令微调/chat template；`你好` 之外的推理失败。
2. **开环不可用**：stage4 开环损失失败。
3. **状态秩瓶颈**：56.5/1024；需要扩容而不是慢记忆平滑。
4. **结构未持久化**：positions/mask 不存 checkpoint；resume 会重置。
5. **单 seed、100 MB 子集**：没有多 seed 显著性。
6. **ppl 口径不统一**：首尾加权后的 ppl 与未加权 ppl 不可直接比较；报告用同配置
   对照。
7. **移植是线性化近似**：不能复现原 23 层。
8. **不是生物仿真**：rate unit、单房室、BPTT；eligibility 是 block 平均共活动，
   不是逐 spike STDP。

## 8. 数据与伦理声明

- 仓库不包含 MiniMind 语料，也不包含 Qwen3 权重；只包含训练/推理代码和结果图。
- 语料许可证、Qwen3 许可证以各自上游为准；使用者需自行获取。
- 语料未做 PII 审计；不要把模型输出用于个人决策或隐私场景。
- 仓库已移除本地绝对路径、用户名、凭据日志；配置文件使用相对路径。
- 结果只用于研究，不构成语言能力、生物真实性或安全性声明。

## 9. 复现

```powershell
$env:PYTHONPATH="src"
.\.venv-str\Scripts\python.exe -m neurosim_gpu.cortex.train `
  --config llm/configs/cortex_qwen3_3p2_minimind.json --dry-run --device cuda `
  --set backbone.model_id=data/qwen3_06b `
  --set backbone.tokenizer_id=data/qwen3_06b `
  --set backbone.local_files_only=true
```

结构图：

```powershell
.\.venv-str\Scripts\python.exe llm\scripts\plot_cortex_structure.py `
  --checkpoint runs/cortex_stage3/final.pt `
  --output llm/results/cortex/current_structure.png
```

诊断：

```powershell
.\.venv-str\Scripts\python.exe llm\scripts\diagnose_cortex.py `
  --checkpoints runs/cortex_stage3/final.pt `
  --batches 4 --sweep-batches 1 `
  --output llm/results/cortex/diagnostics_stage3.json
```

## 10. 持续运行脑波

用 stage3 记录 4096 个 token 的逐 tick 群体活动（2 tick/token = 8192 个采样），
再做 2048 tick 的零 drive 自主运行。为了把归一化频率放到人类 EEG 频段，使用
**1 tick = 10 ms（100 Hz）** 的参考映射；这不是模型真实时间尺度。

![持续运行脑波](results/cortex/brainwaves.png)

| 指标 | teacher-forced | autonomous |
|---|---:|---:|
| ticks | 8,192 | 2,048 |
| 平均活动 | 0.0438 | 0.0123 |
| 活动 std | 0.0956 | 0.0385 |
| PSD 斜率 | **-1.66** | **-4.08** |
| 自相关时间 | 3 tick | 4 tick |
| 状态有效秩 | 243 | **1.65** |
| delta 相对功率 | 28.9% | ≈0% |
| theta | 26.3% | ≈0% |
| alpha | 15.7% | ≈0% |
| beta | 15.8% | ≈0% |
| gamma | 1.9% | ≈0% |

解读：

- teacher-forced 的 PSD 斜率 -1.66，落在人类静息 EEG 常见的 1/f 范围附近；
  但**没有 alpha 峰**，频段功率是 delta/theta 主导的宽谱。
- autonomous 的 PSD 斜率 -4.08，活动迅速衰减到固定点；有效秩只有 **1.65**。
  也就是说：**没有输入时，脑不会维持振荡或持续活动，而是“死掉”**。
- 人类静息 EEG 有持续的自发活动、明显的 alpha（8–13 Hz）峰和状态依赖的频段
  比例；这个模型没有这些。它更像一个被文本驱动的衰减递归网络。

**重要限制**：没有采集人类 EEG，因此这不是统计意义上的“与人类比较”。频段
边界和 100 Hz 轴是参考映射；模型是 rate unit、无 spike、无生物时间。上表只能
说明“谱形状/持续活动”的定性差异，不能说明模型具有人类脑波。

## 11. 结论


- 3+2 管线可以训练，能生成通顺中文，但语言能力有限。
- 空间迁移/重连和损失驱动重连能运行，但没有超过 dense 基线。
- 生物启发记忆（stage3）是目前最好的阶段间结果，但仍低于 dense 诊断。
- 开环解码未解决；下一步应优先扩状态秩和课程式开环，而不是继续加结构。
