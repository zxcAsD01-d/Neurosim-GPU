# Cortex-LM：Qwen3 3+2 × 神经元中段

这是 `neurosim-gpu` 仓库里的 LLM 子项目。它保留 Qwen3-0.6B 的前 3 层和后
2 层，删除中间 23 层，在冻结的两端之间放一个 1024 神经元的递归核。

![当前神经元结构、位置与连接](results/cortex/current_structure.png)

## 结构

```text
tokens → [embedding + 前 3 层] → [1024 神经元, W 1024×1024] → [后 2 层 + LM head]
```

- 前极板：Qwen3 前 3 层，只编码输入，不接收神经元回传。
- 中段：1024 个单房室 rate neuron，递归矩阵 W 由被删 23 层的线性化残差映射
  初始化；每 token 2 个 tick，最后一个 tick 读出。
- 后极板：Qwen3 后 2 层 + LM head，只解码；生成 token 不回灌。
- 连接：65,536 条（64/神经元，dense 的 6.25%）；自环 1,022 条；正权重 34.9%。
- 几何：前极板 `z=0`，后极板 `z=50 d̄`；方形板边长 4.525；神经元在板间。

## 当前状态

| 阶段 | 说明 | ppl |
|---|---|---:|
| 20k dense | 3+2、W 移植、LoRA rank32 | **22.69**（训练 eval） |
| spatial 20200 | 64 边/神经元 + 迁移/重连 | 35.50 |
| spatial 20400 | 继续 200 步恢复 | 25.72 |
| stage2 | 双 tick、慢记忆、噪声、BOS/EOS 加权、loss-driven 重连 | 22.31（同配置 8-batch） |
| stage3 | 短时突触资源、eligibility、多巴胺巩固、稳态缩放 | **21.49**（同配置 8-batch） |
| stage4 | stage3 + 8 步零 drive 开环损失 | 22.15 |

stage3 是目前最好的生物启发结果，但仍没有超过 dense 诊断的 **20.46**。
脑读出有效秩：dense 60.9 → stage3 **56.5**；开环读出仍然崩坏。

## 快速开始

```powershell
$env:PYTHONPATH="src"
.\.venv-str\Scripts\python.exe -m neurosim_gpu.cortex.train `
  --config llm/configs/cortex_qwen3_3p2_minimind.json --dry-run --device cuda `
  --set backbone.model_id=data/qwen3_06b `
  --set backbone.tokenizer_id=data/qwen3_06b `
  --set backbone.local_files_only=true
```

数据路径在配置里是 `data/minimind/pretrain_hq.jsonl`（相对路径，仓库不包含
数据）；见 [DATA.md](DATA.md)。

## 文档

- [完整报告](REPORT.md)
- [技术记录](docs/cortex-lm.md)
- [数据与标注说明](DATA.md)
- 源码：`llm/src/neurosim_gpu/cortex/`
- 配置：`llm/configs/`
- 脚本：`llm/scripts/`
- 结果图与诊断：`llm/results/cortex/`

## 重要限制

- 不是对话模型：`你好` 能触发泛化客服续写，但 `你是谁`、`1+1`、`再见` 失败。
- 开环读出不可用；stage4 的 8 步零 drive 辅助损失反而让 ppl 变差。
- 位置和连接 mask 没有持久化：checkpoint 只存可训练参数；加载时按 seed/W
  重建，训练轨迹中的精确位置/拓扑会丢失。
- 单 seed、100 MB 子集、1024 神经元；所有结论都受此限制。
- 权重 ppl 与首尾 token 加权 ppl 不可直接比较；报告同时给出同配置对照。
