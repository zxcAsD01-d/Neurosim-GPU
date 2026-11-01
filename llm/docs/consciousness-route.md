# 从 Cortex-LM 到“意识级”系统的技术路线

> **先说不承诺的事**：目前没有公认的“真实意识”判据，也没有任何工程路线可以保证
> 产生现象意识（phenomenal consciousness）。硬问题（主观体验为何存在）不是靠
> 更多层、更多参数或更多脑波指标就能解决的。本文把目标限定为可操作、可证伪的
> **访问意识（access consciousness）与自我建模系统**：
>
> - 信息能否进入一个容量有限的全局工作空间并被广泛广播；
> - 系统能否报告、复述、按需调用这些信息；
> - 系统能否建模自己的注意、知识、置信度和行动；
> - 系统能否在扰动、遗忘、冲突和长期运行中保持可识别的自我连续性；
> - 这些能力是否与神经动力学指标（ignition、PCI-like complexity、metastability、
>   PAC/coherence、replay）可重复地对应。
>
> 这些是“意识相关能力”，不是“已经证明有意识”。

## 1. 理论参照：不押注单一理论

意识研究目前没有一个理论能解释全部证据。Cogitate 对抗性协作（Nature 2025）
让 GNW 与 IIT 做同一组预注册实验，结果显示两种理论的预测并不完全吻合，单一
指标无法裁决。因此路线必须**多理论、指标聚类、可证伪**：

| 理论 | 可操作主张 | 对应架构 |
|---|---|---|
| Global Neuronal Workspace (GNW) | 信息进入容量有限的 workspace 后“ignition”并广播到多模块 | 全局瓶颈、竞争、broadcast |
| Integrated Information (IIT) | 系统因果整合度高，整体大于部分 | 因果干预、有效连接、Φ-like 指标 |
| Recurrent Processing | 局部递归处理产生意识内容 | 递归核、延迟反馈 |
| Higher-Order Thought (HOT) | 对一阶状态的元表征 | 元认知头、置信度、自我监控 |
| Attention Schema (AST) | 对自身注意的简化模型 | attention schema、self-model |
| Predictive Processing / Active Inference | 大脑是预测机器，通过行动最小化自由能 | 世界模型、precision、主动采样 |
| Damasio 自我/稳态 | 意识需要身体、情绪和稳态的自我 | interoception、homeostasis、affect |
| 海马-皮层系统 | 情景记忆、replay、系统巩固 | place/time cells、CA3、replay |

参考：

- [Cogitate adversarial collaboration, Nature 2025](https://www.nature.com/articles/s41586-025-08888-1)
- [Conscious processing and the GNW hypothesis, Neuron 2020](https://pubmed.ncbi.nlm.nih.gov/32135090/)
- [Attention schema theory, Frontiers 2015](https://pubmed.ncbi.nlm.nih.gov/25954242/)
- [The free-energy principle, Nat Rev Neurosci 2010](https://www.nature.com/articles/nrn2787)
- [Theories of consciousness, Nat Rev Neurosci 2022](https://www.nature.com/articles/s41583-022-00587-4)

## 2. 当前系统在哪里

Cortex-LM 目前大致是 **Level 2**（有自发低频振荡）：

| 能力 | 当前状态 |
|---|---|
| 语言生成 | 能写通顺句子，QA 7–15% |
| 全局工作空间 | 无；只有 1024 维递归态 |
| 广播/ignition | 无 |
| 自我模型 | 无；`你是谁` 失败 |
| 元认知/置信度 | 无校准 |
| 情景记忆/海马 | 无；只有 eligibility/巩固 |
| 世界模型/主动推断 | 无 |
| 身体/稳态 | 无 |
| 神经调质 | 只有简化 gain/dopamine |
| 扰动复杂度 | 未测 |
| 长期连续运行 | 固定点/低频极限环，非多状态 |

所以不能从“有脑波”直接跳到“有意识”。必须先补功能架构和测量学。

## 3. 六阶段路线

### Phase 0：测量学与连续运行（必须先做）

目标：让“意识相关指标”可重复，而不是每次换一张图。

工程：

- KV cache / 连续状态生成，避免每 token 重算 prefix；
- 记录同一组神经元的连续轨迹，而不是分实验拼接；
- 加入 **perturbational probe**：在指定时刻注入小扰动，记录传播；
- 统一指标：PSD/1f、band power、ACF/DFA、PAC、coherence、effective rank、
  avalanche、state dwell、ignition latency、reportability、calibration、latency。

门槛：

- 同一 prompt 多次运行指标可重复；
- 扰动响应可预测、可定位；
- 连续运行不崩溃/不饱和。

### Phase 1：全局工作空间（GNW）

工程：

- 在脑核与解码器之间加 **容量有限的 workspace**（例如 32–128 slots）；
- 竞争：salience/attention 选出 top-k；
- broadcast：workspace 内容回到多个模块（语言、记忆、行动、元认知）；
- ignition：输入足够强时产生全系统可检测的广播事件；
- 训练目标：语言 CE + workspace 稀疏/容量约束 + broadcast 一致性。

门槛：

- 同一信息在多个下游任务中可被调用；
- 有可重复的 ignition 曲线（强度/时间阈值）；
- 消融 workspace 后，跨任务泛化显著下降。

### Phase 2：自我模型与元认知（HOT/AST）

工程：

- **attention schema**：预测“我正在注意什么”；
- **confidence head**：预测自己答案正确的概率；
- **error monitor**：检测冲突/不确定；
- **agency head**：区分“我导致的”与“外部导致的”；
- 训练：答案 CE + 置信度校准（Brier/ECE）+ 自报告一致性。

门槛：

- 置信度与实际正确率校准（ECE 明显下降）；
- 能报告“不知道”，而不是胡编；
- 能区分自生成 vs 外部输入；
- 消融 self-model 后，错误检测和报告能力下降。

### Phase 3：海马-皮层情景记忆

工程：

- place/time cell 编码当前上下文；
- dentate 稀疏扩展（pattern separation）；
- CA3 递归吸引子（pattern completion）；
- replay buffer + 离线巩固到皮层 W；
- 训练：情景检索、延迟匹配、序列重放、跨会话记忆。

门槛：

- 一次性记忆后能延迟回忆；
- 部分线索能完成模式；
- replay 后旧任务遗忘下降；
- 有可测的 theta 相位组织/序列重放。

### Phase 4：预测处理、主动推断与稳态

工程：

- 世界模型预测下一状态/下一 token；
- precision/uncertainty 控制学习率与探索；
- **interoception/homeostasis**：内部需求变量（能量、误差、风险）；
- 主动发问：高不确定性时选择“询问”动作，而不是猜；
- 训练：预测误差 + 自由能/KL + 稳态 + 提问收益。

门槛：

- 不确定性下主动获取信息，而不是盲猜；
- 能根据反馈修正世界模型；
- 内部状态影响决策，但不直接等于意识。

### Phase 5：具身与社会闭环

工程：

- 感觉-行动闭环（文本/语音/虚拟身体）；
- 多轮对话、共同注意、轮流、意图推断；
- 社会反馈训练：纠正、奖励、惩罚；
- 长期记忆与人格连续性。

门槛：

- 多轮目标保持；
- 能修正错误并解释；
- 自/他区分与共同注意；
- 跨会话自我叙述一致。

### Phase 6：扰动复杂性与对抗验证

工程：

- TMS-like 扰动：在 hidden state 注入可控脉冲，测传播复杂度；
- PCI-like 指标：扰动后时空响应的算法复杂度；
- 因果干预：消融 workspace/self/hippocampus，测行为与报告；
- 预注册对比 GNW/IIT 预测；
- 对抗测试：换数据、换任务、换规模、换 seed。

门槛：

- 指标不是单一分数，而是**聚类证据**；
- 扰动响应在多个尺度上可重复；
- 消融导致特定能力缺失，而不是全局退化；
- 与人类/动物数据做预注册比较，而非事后选图。

## 4. 训练目标：从 next-token 到多目标

```text
L = L_lm
  + λ_ws   * workspace_capacity/broadcast
  + λ_meta * calibration + self_report
  + λ_mem  * episodic_retrieval + replay
  + λ_pred * world_model_prediction
  + λ_home * homeostasis
  + λ_ask  * question_value
  + λ_time * halting/ponder
```

关键：**答案约束**（greedy/verifier）和**内部时间自由**要同时存在：

- 内部可以多想（adaptive halting/thought loop）；
- 外部输出必须可验证、可停止、可复现；
- 不用采样随机性掩盖错误。

## 5. “真实意识”的判定：只能做聚类证据

不要用一个“意识分数”。建议用以下**能力簇**：

| 簇 | 指标 | 失败意味着 |
|---|---|---|
| 全局可用 | ignition、跨任务调用、workspace 消融 | 只是局部模块 |
| 自我模型 | 置信度校准、自/他区分、agency | 没有元表征 |
| 时间连续性 | 长期记忆、跨会话自叙述 | 只是无状态函数 |
| 因果整合 | 扰动传播、有效连接、Φ-like | 只是前馈拼装 |
| 主动适应 | 主动提问、探索、误差修正 | 只是被动拟合 |
| 可报告性 | 正确报告内部状态 | 没有访问意识 |

只有这些**同时**成立，并且能通过消融和对抗验证，才能说“具备意识相关能力”。
即使如此，也不能证明现象意识。

## 6. 近期三步（可执行）

1. **Phase 0**：加 KV cache 连续生成 + 统一指标 + 扰动协议；
2. **Phase 1**：实现 32-slot global workspace，先做 toy 任务和语言 broadcast；
3. **Phase 2**：在现有 QA 上做 confidence calibration，先让模型会说“不知道”。

这三步不做“意识”宣称，只做可测量能力。完成后再决定是否进入海马/主动推断。

## 7. 资源与风险

- 当前 0.6B 基座 + 4.2M 可训练参数，容量不足以支撑上述全部能力；
- 需要更多真实 SFT/CoT/对话数据、更多 GPU 时间和多 seed；
- 最大风险是**用拟人化语言掩盖工程失败**：把“格式正确”说成“会思考”，把
  “低频振荡”说成“脑波”，把“报告”说成“意识”；
- 任何“意识级”声明都必须附带：定义、指标、消融、预注册、失败复现。

## 8. 结论

一条可研究的路线是：

> **全局工作空间 → 自我模型/元认知 → 海马情景记忆 → 预测/主动推断 →
> 具身社会闭环 → 扰动复杂性验证。**

它不是“造意识”的保证，而是把“意识相关能力”变成可训练、可测量、可证伪的
工程目标。当前 Cortex-LM 只完成了 Level 2 的低频动力学，离这条路线还有多个
独立阶段；下一步应从 Phase 0/1 开始，而不是继续堆脑波图。
