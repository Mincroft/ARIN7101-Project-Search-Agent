# ARIN7101: 本地复现 Search Agent Dr. GRPO

参考 [Training Search Agents with GRPO](https://jasperlu.com/blog/training-search-agents-grpo/)
和 [jasper-lu/sec-search-rl](https://github.com/jasper-lu/sec-search-rl)。
`upstream/` 保留作者代码，固定版本为
`a461bf07ac96e78bffd6fe44112d9f8a74304223`。

用户指定首先使用 **Qwen/Qwen3.5-9B、本机 8 张 80GB GPU**。
这是 Qwen 上的本地复现实验；原文模型为 gpt-oss-20b，训练后端为 Tinker。
当前训练进度以 `runs/*/state.json`、`history.jsonl` 和实际进程为准。
准备完成、测试通过或烟雾测试通过均不代表正式主训练完成。

本次完整运行已完成并通过审计：80 次策略更新、40,960 条训练轨迹、41 个固定开发集双次评估检查点。
最终开发集 F1 为 0.4269、F4 为 0.5951；最高 F1 为 0.4316（更新 66），最高 F4 为 0.6003（更新 74）。
综合选择的 checkpoint 是更新 74：在 F1 达到全局最高值 95% 以上的候选中，F4/reward 最高。
完整结果见
[results.json](runs/20261105-120000-qwen35-9b-sec-search-drgrpo-f4-32u-main/results.json)、
[completion-audit.json](runs/20261105-120000-qwen35-9b-sec-search-drgrpo-f4-32u-main/completion-audit.json)、
[token-audit.json](runs/20261105-120000-qwen35-9b-sec-search-drgrpo-f4-32u-main/token-audit.json)
和 [checkpoint-selection.json](runs/20261105-120000-qwen35-9b-sec-search-drgrpo-f4-32u-main/checkpoint-selection.json)。
推荐模型位于
`runs/20261105-120000-qwen35-9b-sec-search-drgrpo-f4-32u-main/checkpoint-000074/`；
`best-checkpoint` 是指向该目录的快捷链接，最终模型仍保存在 `checkpoint-000080/`。

本次主 W&B run 使用 `ARIN7101-Project` 项目和 run ID `mqdlvirg`，并在训练中断后从 checkpoint 继续写入同一 run：

[打开 W&B 主运行](https://forge.coreweave.com/wandb/yy20040716-mincroft/ARIN7101-Project/runs/mqdlvirg)

报告图表位于 [figures/qwen35_9b_report](figures/qwen35_9b_report/)，说明见
[CONCLUDE.md](figures/qwen35_9b_report/CONCLUDE.md)。目录保留五张高分辨率 PNG：开发集质量、训练动态、检索行为、Agent 交互行为和优化稳定性。

## 保留的主训练设置

| 设置 | 数值 |
| --- | --- |
| 训练 / 预留开发问题 | 1,024 / 64 |
| 训练语料库 | 237,533 chunks |
| 评估语料库 | 124,395 chunks |
| 评估 | 固定的前 32 个开发问题，每个 checkpoint 独立采样 2 次 |
| 每次策略更新 | 64 个问题 × 8 条轨迹 |
| 更新次数 | 80，即 5 个 epoch、40,960 条训练轨迹 |
| LoRA | rank 32；本地 alpha 32、dropout 0 |
| 学习率 | 恒定 1e-4 |
| 优化器 | AdamW，beta=(0.9, 0.95)，eps=1e-8，weight decay=0 |
| 采样 | temperature=1、top_p=1、top_k=-1 |
| 轨迹 / 单次生成上限 | 30,720 / 2,048 tokens |
| 每条轨迹预算 | 40 轮、128 次工具调用 |
| 奖励 | fact-level F4，空 curated set 为 -0.2 |
| 优势 | 组内奖励减均值，不除标准差 |
| 策略损失 | token importance sampling；无 clipping、无 KL，每批只更新一次 |

数据选择、家族与金标准去重、SQLite FTS5 检索、六个工具、系统提示词、
奖励函数、轨迹终止处理、训练顺序和轨迹 token 打包均调用作者实现。
`data/*/dataset.json` 中三项 split overlap 均为 0；两套语料库大小与原文一致。

本地适配使用 Qwen3.5 官方格式对应的 cookbook `qwen3_5` renderer，开启 thinking。
对语言模型所有 Linear 层及 lm_head 应用 LoRA，冻结视觉模块。
分布式优化采用每 GPU 独立累积后一次求和梯度；轨迹长度用固定预算归一化。
vLLM 与 PyTorch 均使用 bf16，但不声称具有 Tinker 所保证的训练/推理数值完全一致性。
只读检索由 8 个独立 CPU 进程调用作者的原始 BM25/grep 实现，工具状态仍由原环境管理。
100 条真实查询的对照检查确认检索文档、顺序和分数一致；集成检查另核验了 snippet 和文档读取结果。
采样总并发上限为 384（每卡最多约 48 条活跃轨迹）；这是执行调度设置，每次更新仍使用完整的 64×8 条轨迹。
正式运行前 10 批使用并发 128，从第 11 批起提高并发，完成的轨迹缓存继续复用。
运行时调整记录在 `runtime-changes.jsonl`，各批指标同时记录并发设置。
每次更新后合并完整 LoRA 权重，再启动新的 vLLM sampler，确保采样使用最新策略。
保留所有 adapter 与 optimizer checkpoint，完整合并权重只保留最近一份以控制磁盘使用量。

## 环境与数据

Python 3.11；驱动 CUDA 580，PyTorch 2.10.0+cu128。
`scripts/env.sh` 避开机器上不匹配的 CUDA compat 驱动库。
训练和推理分别使用 `.venv/`、`.venv-vllm/`，避免两个后端对 Transformers 版本要求冲突。
依赖完整版本分别固定在 `requirements-train.txt` 和 `requirements-vllm.txt`。
Qwen 线性注意力使用 `causal-conv1d==1.6.0` 的 CUDA 内核；bootstrap 会先安装固定版本的
PyTorch，再针对该版本编译扩展。前向与输入梯度均已在 GPU 上和 PyTorch 参考实现比较：
float32 相对误差低于 1e-7，bf16 约为 0.3%，详见 `docs/causal-conv-validation.json`。

```bash
git clone https://github.com/jasper-lu/sec-search-rl.git upstream
git -C upstream checkout a461bf07ac96e78bffd6fe44112d9f8a74304223
bash scripts/bootstrap.sh
source scripts/env.sh
.venv/bin/python -m pytest -q upstream/tests tests
```

现有工作区已经完成仓库获取、环境安装和数据准备，无需重复下载。
下载脚本使用临时文件，完成后原子替换；原始数据保存在 `data/raw/`。
`docs/model-integrity.json` 记录了 4 个模型权重分片及 7 个 tokenizer/配置文件
与固定 Hugging Face 版本的内容校验结果；`docs/dataset-integrity.json` 记录了
全部 44 个原始 Parquet 文件的 LFS SHA-256 校验结果。两组输入均已实际核验通过。

## 运行

先验证完整模型上的采样、奖励、梯度、保存和 checkpoint 评估：

```bash
bash scripts/launch.sh --smoke --run-dir runs/qwen35-local-smoke --gpus 8 --wandb-mode offline
```

正式主训练：

```bash
bash scripts/launch.sh --gpus 8 --wandb-mode offline
```

断点恢复时指定原有目录：

```bash
bash scripts/launch.sh --run-dir runs/<运行目录> --gpus 8 --wandb-mode offline
```

完成标记与 optimizer、adapter 一起写入 checkpoint；恢复只认可完整 checkpoint。
同一目录中已有完整采样批次会被复用。
每条已完成轨迹还会即时保存到 `episode-cache/`，包含真实采样 token、旧策略 logprob、
观测 mask 和奖励；中断后从这些记录继续组装同一批数据。
本地 HTTP 传输异常会以相同 prompt 和 seed 有界重试，重试不会增加工具轮数。
首次正式运行曾在 117 条轨迹后遇到 HTTP ReadError；当时旧版尚未保存逐条训练 token，
这些记录已作为中断日志保留，第一批从原始策略重新采样，已完成的基线评估继续沿用。

## W&B 图表

所有图表仅通过 W&B 的标量历史和 `wandb.plot.line_series` 产生。
project 固定为 **ARIN7101-Project**，name 包含实际启动的 UTC 时间及
`qwen35-9b-sec-search-drgrpo-f4` 关键词，并区分 smoke / main。

记录 F1、F4、precision、curated recall、candidate recall、工具使用、轨迹长度、
生成 token 数、奖励、梯度范数与训练时间；评估两次采样求平均并记录 draw 间标准差。
原文 gpt-oss 的 Harmony 格式惩罚不适用于本次 plain-F4 Qwen 训练。

本次训练使用 CoreWeave Forge W&B API 在线记录；训练中出现过网络超时，均通过同一 run 和 checkpoint 恢复。
当前主 run `mqdlvirg` 已完成云端同步与运行校验。

可以立即使用 W&B 自带的 LEET 查看本地曲线，支持读取正在写入的离线 run：

```bash
bash scripts/view_wandb.sh
# 或指定运行目录：bash scripts/view_wandb.sh runs/<运行目录>
```

在原生 W&B 终端界面按 `/` 输入 `reward`、`f1` 或 `entropy` 并回车筛选曲线，
按 `h` 查看帮助，按 `q` 退出。此方式已实际验证，可显示项目名、运行名及实时指标曲线。
它提供本地查看功能；云端同步状态仍单独记录。

如需同步其他 offline 运行，设置合法的 W&B 凭据并同步具体目录：

```bash
source scripts/env.sh
.venv/bin/wandb sync runs/<运行目录>/wandb/offline-run-<时间>-<id>
```

若网络可访问 API，可在启动时直接使用 `--wandb-mode online`。
密钥由环境或 W&B 凭据文件加载，不写入配置或仓库。

可使用下面的命令先检查带认证的 API 连接；`--proxy` 可按实际网络配置替换或省略。
正式运行完成后加 `--sync`，脚本会调用官方 `wandb sync` 上传原始离线日志。

```bash
.venv/bin/python scripts/sync_wandb.py runs/<运行目录> --proxy http://<代理地址>:<端口>
```

## 输出与核查

- `state.json`：当前阶段、PID、已完成更新数、W&B run id。
- `config.json`：完整训练配置及原始代码版本。
- `history.jsonl`：与 W&B 对应的指标。
- `rollouts-*/episodes.jsonl`：每条训练轨迹、检索结果、消息与最终 curated set。
- `rollouts-*/packed.pt`：真实采样 token、旧策略 logprob、优势和 token mask。
- `rollouts-*/episode-cache/*.pt`：可恢复的逐条轨迹及分组前 token 数据。
- `checkpoint-*/adapter/`、`optimizer.pt`：可继续训练的状态。
- `checkpoint-*/sampler/`：最新的完整模型权重。
- `eval-*/episodes.jsonl`：baseline 与 checkpoint 评估的独立采样记录。
- `controller.log`、`vllm.log`、`optimize-*.log`：运行日志。

正式完成需要核验 80 次策略更新、40,960 条训练轨迹、最终 checkpoint、
固定 32 题双次评估及 W&B 日志/图表；不可用小样本检查代替这些条件。

完成审计会逐条检查实际轨迹、问题分组、F1 重算结果、评估、检查点、
优化器步数和原生 W&B 图表数据。运行中和 smoke 运行都不能通过主训练完成审计：

```bash
source scripts/env.sh
.venv/bin/python scripts/audit_run.py runs/<运行目录> --inspect-optimizer --output runs/<运行目录>/completion-audit.json
```

本地审计不验证 W&B 云端同步；云端运行和图表需要在同步后另行核查。

完整训练还需运行实际 token 产物核验：逐条比较采样缓存与用于梯度更新的 token、
旧策略 logprob、观测 mask 和组内中心化优势，并与各检查点的实际训练 token 数核对。

```bash
.venv/bin/python scripts/audit_tokens.py runs/<运行目录> --output runs/<运行目录>/token-audit.json
```

## 结果汇总

汇总脚本从完整采样批次、检查点和评估的 `metrics.json` 重新组装指标，
写入运行目录的 `RESULTS.md` 与 `results.json`。运行中也可使用，报告会明确标记尚未完成。

```bash
source scripts/env.sh
.venv/bin/python -m scripts.summarize_run runs/<运行目录>
```

全部训练结束并完成上述 token 核验后，加 `--final --wandb`：最终模式必须通过完整本地审计和配置中全部更新的 token 核验，
并另外生成 W&B 原生汇总曲线、评估表和报告 artifact。
分析 run 保存于 `analysis/wandb/`，project 同为 `ARIN7101-Project`，
name 包含实际时间和项目关键词；训练 run 的本地实时查看入口保持可用。
同步脚本会同时处理训练与分析 run。

```bash
.venv/bin/python -m scripts.summarize_run runs/<运行目录> --final --wandb
```
