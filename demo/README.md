# NeuroBody: 树突＋慢适应运动控制实验

这个可选子项目把目标条件运动控制网络接到公开的 DendriticDynamics 方程，使用固定稀疏图和截断 BPTT。神经状态在 tick 之间保留，在 episode 边界重置。它测试运动学伺服关节跟踪，尚不包含重力、接触、步行或平衡。

本次发布包含离线训练、旧 checkpoint 兼容、可选 VRM clip 导出器、回归测试及历史对照 JSON。**前端未接入新版模型**；旧本机 viewer、VRM 资产、AMASS 数据和训练权重不随源码发布。

## 安装与短验证

从仓库根目录运行。子项目依赖单独安装，核心包依赖仍只有 PyTorch：

~~~powershell
python -m pip install -e .
python -m pip install -r demo/requirements.txt
python -m unittest discover -s demo/tests -p test_vnext.py -q
python demo/neurobody/vnext_run.py --task goals --policy dendritic --ticks 8 --batch 1 --episodes 1 --goal-dwell 4 --device cpu --allow-cpu --out demo/runs/smoke
~~~

8 tick smoke 只验证训练/评估/checkpoint 流程，不是质量测量。CUDA 训练示例：

~~~powershell
python demo/neurobody/vnext_run.py --task goals --policy dendritic --ticks 6000 --batch 8 --seed 0 --episodes 3 --m1-to-motor 8 --reflex-gain 0 --goal-mode random --goal-dwell 150 --eval-ticks 4500 --device cuda:0 --out demo/runs/dendritic_s0
~~~

policy 还可选择 vnext、vnext_dale、plain、reservoir、pd、reflex。显式 --reflex-gain 0 避免反射控制混入神经网络对照。默认神经元布局为 6,020；后续扩展池不隐式改变历史图。

## 已知结果和限制

[失败复盘与迁移报告](docs/neurobody-dendritic-migration-20261120.md) 记录了误差直通、Dale 符号、旧图拓扑、RNG 和展示版本不一致的问题。[匹配对照](results/dendritic_migration_20261120/matched_comparison.json) 中，新核心的姿态误差为 0.08947，PD 为 0.05376；只有一个训练 seed，不能声称性能已修复。

源码使用可微 PyTorch CSR 和公开神经元方程，没有启用 native 推理 kernel，也没有做速度比较。误差/教师通道仅用于预训练实验；不能直接视作物理阶段的可用控制信号。

## 可选骨架导出

提供你有权使用的本机 VRM 及本地 JSON 配置，配置包含 avatar.vrm（文件路径）和可选 avatar.height_m。配置和资产保留在忽略的 demo/config/、demo/assets/ 目录；训练不需要它们：

~~~powershell
python demo/neurobody/rollout_vnext.py --checkpoint demo/runs/dendritic_s0/model.pt --config demo/config/local_avatar.json --goal-mode random --frames 600 --out demo/assets/clip_embodied.json --device cuda:0
~~~

导出器保留核心类型及参数 metadata；眼图是占位，因为模型没有使用视觉输入。导出 clip 不等于 viewer 已接入。旧本机展示使用 Mafuyu v0.91 · 有坂みと；该 VRM 禁止再分发，此仓库不包含它。项目许可及第三方权利见根目录的 LICENSE、LICENSING.md 和 THIRD_PARTY_NOTICES.md。
