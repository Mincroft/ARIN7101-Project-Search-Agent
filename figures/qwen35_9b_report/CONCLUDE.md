# Search Agent Training: Figure Interpretation and Conclusions

## Experiment and reading guide

These five figures describe the Qwen3.5-9B SEC search agent trained with local, eight-GPU Dr. GRPO and a fact-level F4 reward. Training completed 80 optimization updates: 64 queries per update and eight sampled trajectories per query, totaling 40,960 trajectories. The 1,024 training queries were traversed five times. These are repeated queries with newly sampled trajectories, not 40,960 distinct questions.

The associated W&B run is [mqdlvirg](https://forge.coreweave.com/wandb/yy20040716-mincroft/ARIN7101-Project/runs/mqdlvirg). Figures are rendered from the authoritative local metrics used by that run, stored under `runs/20261105-120000-qwen35-9b-sec-search-drgrpo-f4-32u-main/`. Development evaluation uses the same 32 held-out queries with two independent draws per query, at baseline and every two updates: 41 evaluation checkpoints in total.

All figures are PNG files exported at 1200 DPI. Figures 02, 03, 04, and 06 show raw observations as faint points and recursive exponential moving averages (EMA) as solid lines. EMA uses span=5, alpha=1/3, `adjust=False`, and initializes from the first observation. Smoothing is descriptive and lags changes; numerical conclusions and checkpoint selection use raw metrics. Figure 01 is not smoothed. Its shaded regions are the recorded standard deviations across two draw-level evaluation means, **not confidence intervals**, query-level uncertainty, or variation across independent training seeds.

## 01 — Development-set quality

![Development-set quality](01_development_quality.png)

**Question:** Does reinforcement learning improve performance on held-out queries?

The F1 and F4 curves rise substantially from the untrained policy. F1 summarizes the balance between evidence precision and fact coverage; F4 puts greater weight on recall, matching the training objective. The late-stage fluctuations show why the last checkpoint should be compared with earlier candidates rather than assumed to be optimal.

| Checkpoint | Development F1 | Development F4 | Interpretation |
| --- | ---: | ---: | --- |
| Baseline | 0.1308 | 0.1347 | Initial agent |
| Update 66 | **0.4316** | 0.5959 | Highest observed F1 |
| Update 74 | 0.4193 | **0.6003** | Highest observed F4 and reward |
| Update 80 | 0.4269 | 0.5951 | Final policy |

At update 74, precision is 0.3285 and recall is 0.6479. It is the existing task-aligned recommendation because it maximizes F4 while retaining at least 95% of the maximum observed F1. This 95% rule was chosen after inspecting development results; it is an exploratory selection heuristic, not a preregistered criterion. Update 66 remains the alternative for an F1-first application. The small gaps among late candidates do not establish statistically significant superiority.

**Report contribution:** Primary evidence that the learned search policy transfers to held-out queries within this SEC evaluation setup. Report the F1-best and F4-best checkpoints separately.

## 02 — Training dynamics

![Training dynamics](02_training_dynamics.png)

**Question:** Does the policy learn from the reward signal?

The top panel shows the per-update mean reward and its EMA. Reward increases from -0.0461 at update 1 to 0.5365 at update 80. The lower panel shows improvements in training F1 and F4, with F4 higher than F1 as expected for a recall-oriented objective. Reward and F4 are closely related by construction, so agreement between them is not independent validation.

Each update samples a different batch of queries, and queries repeat across passes. Batch difficulty, sampling randomness, and policy changes all contribute to fluctuations. Training and development also use different retrieval corpora; their absolute score difference is not a controlled generalization-gap estimate.

The training and development curves improve together over the full run. There is no clear sustained development collapse while training reward rises. However, only one training run and a small, repeatedly inspected development set are available, so these curves cannot rule out overfitting or establish broad generalization.

**Report contribution:** Evidence that reward optimization changes task performance, supporting the development-set results rather than replacing them.

## 03 — Retrieval behavior

![Retrieval behavior](03_retrieval_behavior.png)

**Question:** What retrieval and evidence-selection behaviors accompany the quality gains?

The left panel tracks candidate recall, curated-evidence precision, and fact-level recall on training trajectories. Candidate recall measures coverage in encountered candidates; recall measures coverage retained in the submitted evidence set. Training recall rises from 0.1203 at update 1 to 0.5829 at update 80, while candidate recall changes from 0.5049 to 0.6297. This pattern is consistent with improvements in both candidate discovery and retention of useful evidence. Changing query batches mean these endpoints are descriptive, not a controlled decomposition of causality.

Precision remains lower than recall, which is consistent with an F4 objective favoring coverage. Consequently, high recall should not be described as uniformly precise evidence selection.

The right panel shows curated-document counts and curate-tool calls. The agent moves beyond rarely curating evidence, briefly over-collects early in training, and later uses a more bounded evidence set with repeated curation. A larger evidence set alone would not demonstrate better retrieval; the quality metrics in the left panel are needed to interpret this behavior.

**Report contribution:** A behavioral explanation for improved search scores and a visible precision–recall trade-off.

## 04 — Agent interaction behavior

![Agent interaction behavior](04_agent_behavior.png)

**Question:** How does tool use evolve, and what limitations remain?

The left panel shows mean interaction turns, BM25 calls, and document-read calls. Mean turns increase from approximately 6.10 at update 1 to 22.91 at update 80, consistent with the agent engaging in longer search-and-inspection sequences. Tool-use counts are behavioral evidence; longer trajectories do not by themselves prove efficiency or better reasoning.

The right panel requires careful interpretation. `Finished` and `Context overflow` are episode fractions. `Invalid curation` is the **mean number of invalid curation attempts per episode**, not an error probability. The current shared axis labeled “Fraction” therefore applies only to the first two series; quote invalid curation as a count in report text.

Context overflow remains a major limitation: at update 80 it affects 83.01% of training episodes, while only 16.41% finish explicitly. In development evaluation at update 80, overflow is 79.69% and explicit finishing is 20.31%. The environment scores the accumulated evidence even when the agent does not finish cleanly. Thus improved F4 demonstrates better evidence coverage under the rollout budget, **not reliable end-to-end completion**. Invalid curation attempts fall from approximately 0.633 per episode at update 16 to 0.127 at update 80, but this does not eliminate the termination bottleneck.

**Report contribution:** Shows policy behavior and honestly identifies the next engineering priorities: stopping decisions, context management, and reliable curation.

## 06 — Optimization stability

![Optimization stability](06_optimization_stability.png)

**Question:** Did the numerical optimization remain well behaved?

The left panel shows the policy objective and the right panel shows gradient norm. Finite gradients remain nonzero, with a generally lower norm late in training and no sustained explosion visible in the recorded values. The faint points preserve transient excursions that EMA might otherwise conceal.

This is a group-centered policy-gradient objective. Negative loss values are expected, and decreasing loss is not equivalent to increasing search quality. The loss should be used as a numerical diagnostic alongside development metrics, not as the checkpoint-selection criterion.

The full artifact audit passed for all 80 checkpoints and 40,960 trajectories, including token/log-probability alignment, observation masks, group-centered advantages, and optimizer action-token counts. Sampling suffered a local vLLM timeout after update 52 and subsequently resumed from the saved checkpoint in the same W&B run. Numerical stability does not imply uninterrupted infrastructure operation.

**Report contribution:** Supports the implementation's numerical integrity and reproducibility, while keeping optimization evidence distinct from task-performance evidence.

## Overall conclusion for the report

Across this single 80-update experiment, local Dr. GRPO substantially improved the Qwen3.5-9B search agent's held-out SEC evidence-retrieval scores. The quality gains co-occurred with increased engagement in tool-based search and more effective evidence coverage, while checkpoint and token audits confirmed the integrity of the completed optimization. The reward-aligned candidate is checkpoint 74; checkpoint 66 gives the highest observed development F1.

The remaining weakness is termination and context management: high evidence scores coexist with frequent context-budget exhaustion. Precision also trails recall. The results support a claim of improved retrieval policy within the measured setting, rather than universal generalization, optimal stopping, or a fully reliable production agent.

For a stronger final evaluation, freeze the checkpoint-selection rule, compare checkpoints 66/74/80 on untouched held-out questions, use additional sampling draws and independent training seeds, and report query-level paired bootstrap intervals. Repeatedly selecting among 41 checkpoints on the same 32 development questions can produce selection optimism. Differences between the selected and final checkpoint should therefore be treated as observed development differences until independently confirmed.

Suggested report order: **01 → 02 → 03 → 04 → 06**: held-out improvement, learning dynamics, retrieval behavior, remaining operational limitations, and numerical integrity.

## Reproduction

From the project root:

```bash
python figures/plot_training_report_suite.py
```

The script generates the five PNG figures only. Original numeric metrics remain in the experiment directory; this report folder contains only PNG figures and this document. Existing figure numbers are preserved, with 05 intentionally removed.
