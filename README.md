# ARIN7101：SEC Search Agent 的本地 Dr. GRPO 复现

本项目复现 [Training Search Agents with GRPO](https://jasperlu.com/blog/training-search-agents-grpo/) 的 SEC 文档搜索任务。
原项目使用 gpt-oss-20b 和 Tinker；本仓库按实验要求改用 **Qwen/Qwen3.5-9B**、本地 vLLM/PyTorch 和 8 张 80GB GPU，因此结果用于本地复现与方法分析，不宣称与原文数值完全等同。

## 已完成实验

主实验目录：[`runs/20261105-120000-qwen35-9b-sec-search-drgrpo-f4-32u-main`](runs/20261105-120000-qwen35-9b-sec-search-drgrpo-f4-32u-main)

| 项目 | 结果 |
| --- | --- |
| 模型 | Qwen/Qwen3.5-9B |
| 更新次数 | 80 |
| 训练轨迹 | 40,960 |
| 训练查询 / 开发查询 | 1,024 / 64 |
| 每次更新 | 64 查询 × 8 条轨迹 |
| 开发评估 | 固定 32 题，每个 checkpoint 采样 2 次 |
| 最终 F1 / F4 | 0.4269 / 0.5951 |
| 最高 F1 | 0.4316（更新 66） |
| 最高 F4 | 0.6003（更新 74） |

综合选择规则是：在 F1 达到全局最高值 95% 以上的候选中，优先选择 F4/reward，再看 recall。
因此推荐使用 [`checkpoint-000074`](runs/20261105-120000-qwen35-9b-sec-search-drgrpo-f4-32u-main/checkpoint-000074)，快捷链接为 [`best-checkpoint`](runs/20261105-120000-qwen35-9b-sec-search-drgrpo-f4-32u-main/best-checkpoint)。最终权重保存在 [`checkpoint-000080`](runs/20261105-120000-qwen35-9b-sec-search-drgrpo-f4-32u-main/checkpoint-000080)。

完整审计结果：[`completion-audit.json`](runs/20261105-120000-qwen35-9b-sec-search-drgrpo-f4-32u-main/completion-audit.json)、[`token-audit.json`](runs/20261105-120000-qwen35-9b-sec-search-drgrpo-f4-32u-main/token-audit.json)、[`checkpoint-selection.json`](runs/20261105-120000-qwen35-9b-sec-search-drgrpo-f4-32u-main/checkpoint-selection.json)。80 个 checkpoint、40,960 条轨迹以及 token、logprob、mask 和 advantage 检查均已通过。

## W&B 与报告图表

主实验持续写入同一个 W&B run：[`mqdlvirg`](https://forge.coreweave.com/wandb/yy20040716-mincroft/ARIN7101-Project/runs/mqdlvirg)。

报告图表位于 [`figures/qwen35_9b_report`](figures/qwen35_9b_report/)，包含开发集质量、Training Reward、检索行为、Agent 交互行为和优化稳定性。图表使用 EMA（span=5）展示训练趋势，同时保留原始点；评估质量图保留原始 checkpoint 结果。完整解释见 [`CONCLUDE.md`](figures/qwen35_9b_report/CONCLUDE.md)。

## 训练配置

- Reward：fact-level F4
- LoRA：rank 32，alpha 32，dropout 0
- 学习率：`1e-4`
- 精度：BF16
- 轨迹上限：30,720 tokens；单次生成上限 2,048 tokens
- 单条轨迹：最多 40 轮、128 次工具调用
- 优势：组内奖励减均值，不除标准差
- 策略损失：token importance sampling，无 ratio clipping、无 KL
- 检索：作者实现的 BM25/grep/文档读取工具，SQLite FTS5

训练数据来自 `pat-jj/harness-1-train-data` 的 SEC fact-level 子集：训练语料库 237,533 个 chunks，评估语料库 124,395 个 chunks。数据和模型文件均已完成版本与 SHA 校验。

## 环境与运行

```bash
source scripts/env.sh
bash scripts/bootstrap.sh
.venv/bin/python -m pytest -q upstream/tests tests
```

烟雾测试：

```bash
bash scripts/launch.sh --smoke --run-dir runs/qwen35-local-smoke --gpus 8 --wandb-mode offline
```

正式运行使用 [`configs/qwen35_9b_f4_32.json`](configs/qwen35_9b_f4_32.json)。已完成的 80 次实验配置保存在主实验目录的 [`config.json`](runs/20261105-120000-qwen35-9b-sec-search-drgrpo-f4-32u-main/config.json)。

断点恢复：

```bash
bash scripts/launch.sh \
  --config configs/qwen35_9b_f4_32.json \
  --run-dir runs/20261105-120000-qwen35-9b-sec-search-drgrpo-f4-32u-main \
  --gpus 8 \
  --wandb-mode online
```

恢复逻辑只接受带 `complete` 标记的 checkpoint；每个 optimization step 保存独立的 adapter、optimizer 和 sampler 状态。

## 数据与代码版本

- 上游代码：[jasper-lu/sec-search-rl](https://github.com/jasper-lu/sec-search-rl)
- 固定 commit：`a461bf07ac96e78bffd6fe44112d9f8a74304223`
- 原始数据完整性：[dataset-integrity.json](docs/dataset-integrity.json)
- 模型完整性：[model-integrity.json](docs/model-integrity.json)
- 训练配置：[`config.json`](runs/20261105-120000-qwen35-9b-sec-search-drgrpo-f4-32u-main/config.json)

## 重要限制

训练显著提升了证据覆盖和开发集分数，但 Agent 仍经常达到上下文上限，且 precision 低于 recall。Context overflow 不等于没有提交证据：环境会对超限前已经收集的 curated evidence 评分。当前结果支持“检索策略得到改善”的结论，不代表已解决可靠停止、上下文管理或生产级泛化问题。

`Invalid Curation` 表示 Agent 尝试 curate 未经搜索返回的文档 ID，或超过最多 30 个 curated 文档；本实验没有对该诊断指标启用额外惩罚。
