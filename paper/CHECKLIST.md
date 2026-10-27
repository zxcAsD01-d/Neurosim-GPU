# 投稿前检查清单

按顺序勾选。前六项是硬性条件，后面的建议项会影响审稿印象。

## 1. 身份与许可

- [ ] arXiv 作者字段使用**真实姓名**（不是 `zxcAsD01`），单位填 `Independent researcher` 或实际单位
- [ ] 仓库 URL 写进 comments 字段，并能公开访问
- [ ] 选定许可证（建议 `CC BY 4.0`；若计划投期刊，用 arXiv 默认非独占许可）
- [ ] 若为首次向 `cs.NE` 投稿，确认是否**需要 endorsement**，提前找推荐人
- [ ] PDF 元数据中没有不想公开的信息（`main.tex` 的 `\author{}` 会写进 PDF）

## 2. 数字与声明

- [ ] 摘要里每个数字都能在正文表格中找到（0.74–0.82×、2.8–6.6%、3.0–4.5×、0.1 个百分点、5–81×）
- [ ] 没有出现"首个""最快""超越 X"这类未经对照的措辞
- [ ] GPU 数字标注了硬件与软件栈（RTX 5070 Laptop / CUDA 13.1 / FP32）
- [ ] 125M 边明确写成**外推**而非实测
- [ ] 事件路径的 3–4.5× 明确限定为 native kernel，且说明 eligibility 会破坏该结论
- [ ] 行为克隆明确写成"不是强化学习"

## 3. 图表与排版

- [ ] `python paper/make_assets.py` 重新生成后与 PDF 一致
- [ ] `python paper/check_tex.py` 通过（inputs / cites / braces / ASCII）
- [ ] `tectonic -X compile main.tex` 无 overfull / undefined 警告（当前：10 页，0 警告）
- [ ] 5 张图在 100% 缩放下可读（pgfplots 矢量图，无位图）
- [ ] 4 张表没有被挤出页边（cartpole 表 6 列最宽）

## 4. 参考文献

- [ ] 11 条引用全部解析（`check_tex.py` 会报 unknown key）
- [ ] Brian 2 / NEST / GeNN 的年份、期刊、卷页正确
- [ ] 没有引用未读过的文献来凑数

## 5. 工件与可复现

- [ ] 仓库公开，`main` 与 PDF 描述的版本一致（`paper/main.pdf` 的提交在 `main` 上）
- [ ] `results/raw/` 的 105 份载荷与摘要数字一致
- [ ] 若挂了 Zenodo DOI，`CITATION.cff` 与 README 已更新
- [ ] 论文第三条贡献（trained weights 上的有界误差）能被 `results/raw/*.json` 的 `activity` 字段复算

## 6. 提交当天

- [ ] 本地编译出的 `main.pdf` 与 arXiv 预览一致
- [ ] 摘要为纯文本，无 LaTeX 命令、无数学符号乱码（用 `x` 而不是 `×`）
- [ ] 分类：primary `cs.NE`，cross-list `cs.LG`（可选 `cs.MS`）
- [ ] 记住 arXiv 的修改窗口：首次提交后 24 小时内可自行替换版本

## 建议但非必须

- [ ] 补一张"与 GeNN/Norse 同图对照"的表（目前论文明确说没有做，审稿人会问）
- [ ] 补 3 seeds 的误差棒（目前表格有 std，图上是散点）
- [ ] 在 GitHub Release 里附上 `main.pdf`，让读者不必进 `paper/` 目录
- [ ] 给论文加一句 artifact availability 的具体路径（`paper/`、`results/raw/`、`benchmarks/performance_suite.py`）
