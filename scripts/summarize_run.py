"""Summarize authoritative checkpoints and render all figures with native W&B.

This can report partial progress, but only the full artifact audit can mark the
local experiment complete. Cloud synchronization is reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.audit_run import audit

ROOT = Path(__file__).resolve().parents[1]


def paired_f1(reference_path, candidate_path):
    def load(path):
        groups = defaultdict(dict)
        for line in path.open():
            row = json.loads(line)
            key, draw = row["query_id"], row["draw"]
            if draw in groups[key]:
                raise ValueError(f"Duplicate query/draw in {path}")
            groups[key][draw] = float(row["metrics"]["f1"])
        if len(groups) != 32 or any(set(v) != {0, 1} for v in groups.values()):
            raise ValueError(f"Expected 32 queries and two draws each in {path}")
        return {key: sum(values.values()) / 2 for key, values in groups.items()}

    reference, candidate = load(reference_path), load(candidate_path)
    if reference.keys() != candidate.keys():
        raise ValueError("Evaluation query sets differ")
    deltas = np.array([candidate[k] - reference[k] for k in sorted(reference)])
    generator = np.random.default_rng(42)
    samples = deltas[generator.integers(0, len(deltas), (10000, len(deltas)))].mean(
        axis=1
    )
    return {
        "mean_difference": float(deltas.mean()),
        "question_bootstrap_95pct_ci": np.quantile(samples, [0.025, 0.975]).tolist(),
        "queries": len(deltas),
        "draws_per_query": 2,
        "resamples": 10000,
        "seed": 42,
    }


def summarize(directory, *, final=False):
    config = json.loads((directory / "config.json").read_text())
    state = json.loads((directory / "state.json").read_text())
    rows = {}
    for pattern in [
        "rollouts-*/metrics.json",
        "checkpoint-*/metrics.json",
        "eval-*/metrics.json",
    ]:
        for path in sorted(directory.glob(pattern)):
            kind, number = path.parent.name.rsplit("-", 1)
            if kind == "checkpoint" and not (path.parent / "complete").exists():
                continue
            rows.setdefault(int(number), {"update": int(number)}).update(
                json.loads(path.read_text())
            )
    evaluations = [row for _, row in sorted(rows.items()) if "eval/f1" in row]
    completed = [
        int(p.parent.name.rsplit("-", 1)[1])
        for p in directory.glob("checkpoint-*/complete")
    ]
    baseline = next((row for row in evaluations if row["update"] == 0), None)
    last = evaluations[-1] if evaluations else None
    best = max(evaluations, key=lambda row: row["eval/f1"]) if evaluations else None
    check = audit(directory, inspect_optimizer=final)
    connectivity = directory / "wandb-connectivity.json"
    cloud = (
        json.loads(connectivity.read_text())
        if connectivity.exists()
        else {"authenticated": False}
    )
    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_name": config["run_name"],
        "model": config["model"],
        "upstream_commit": config["upstream_commit"],
        "completed_updates": sorted(completed),
        "target_updates": config["max_steps"],
        "training_episodes_committed": sum(
            row.get("train_raw/episodes", 0) for row in rows.values()
        ),
        "baseline": baseline,
        "latest_evaluation": last,
        "best_dev_f1_evaluation": best,
        "history": [row for _, row in sorted(rows.items())],
        "local_completion_audit": check,
        "wandb_connectivity": cloud,
        "cloud_run_verified": False,
        "controller_state": state,
    }
    changes = directory / "runtime-changes.jsonl"
    result["runtime_changes"] = (
        [json.loads(line) for line in changes.read_text().splitlines()]
        if changes.exists()
        else []
    )
    result["training_config"] = config
    token_audit = directory / "token-audit.json"
    result["training_token_audit"] = (
        json.loads(token_audit.read_text()) if token_audit.exists() else None
    )
    result["dataset_pass_checks"] = {
        path.name: json.loads(path.read_text())
        for path in sorted(directory.glob("epoch-*-audit.json"))
    }
    result["input_integrity_checks"] = {
        path.name: json.loads(path.read_text())
        for path in [
            ROOT / "docs/model-integrity.json",
            ROOT / "docs/dataset-integrity.json",
        ]
        if path.exists()
    }
    sources = [
        *sorted((ROOT / "local_grpo").glob("*.py")),
        ROOT / "requirements-train.txt",
        ROOT / "requirements-vllm.txt",
    ]
    result["source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    if last is not None and last["update"] > 0 and baseline is not None:
        result["latest_vs_baseline_f1"] = paired_f1(
            directory / "eval-000000/episodes.jsonl",
            directory / f"eval-{last['update']:06d}/episodes.jsonl",
        )
    if final and not check["training_and_local_charts_complete"]:
        raise RuntimeError(
            "The full local completion audit failed; no final report produced"
        )
    token_audit = result["training_token_audit"] or {}
    if final and token_audit.get("audited_updates") != list(
        range(1, config["max_steps"] + 1)
    ):
        raise RuntimeError(
            "Run scripts/audit_tokens.py for all configured updates before producing the final report"
        )
    return result


def write_report(directory, report):
    (directory / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    complete = report["local_completion_audit"]["training_and_local_charts_complete"]
    lines = [
        "# Qwen3.5-9B SEC search Dr. GRPO",
        "",
        f"Run: `{report['run_name']}`",
        "",
        f"本地完整训练审计：{'通过' if complete else '未通过（训练尚未完成或存在未核验项目）'}。",
        f"已完成 {len(report['completed_updates'])}/{report['target_updates']} 次更新，已提交 {report['training_episodes_committed']:,} 条训练轨迹。",
        "W&B 云端运行和图表尚未由本报告核验，连接状态见 wandb-connectivity.json。",
        "",
        "本实验按用户要求使用 Qwen3.5-9B 和本地 8 卡；Blog 使用 gpt-oss-20b 和 Tinker，数值结果不作等同声明。",
        "",
        "| 更新 | F1 | F4 | Precision | Recall | 双次评估轨迹 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["history"]:
        if "eval/f1" in row:
            lines.append(
                f"| {row['update']} | {row['eval/f1']:.4f} | {row['eval/f_beta']:.4f} | "
                f"{row['eval/precision']:.4f} | {row['eval/recall']:.4f} | {int(row['eval/episodes'])} |"
            )
    comparison = report.get("latest_vs_baseline_f1")
    if comparison:
        low, high = comparison["question_bootstrap_95pct_ci"]
        lines.extend(
            [
                "",
                (
                    f"最新检查点相对基线的 F1 差值为 {comparison['mean_difference']:+.4f}，"
                    f"按问题配对 bootstrap 的 95% 区间为 [{low:+.4f}, {high:+.4f}]。"
                ),
                "区间以问题为重采样单位，每题先平均两次轨迹；10,000 次重采样，seed=42。",
            ]
        )
    best = report.get("best_dev_f1_evaluation")
    if best:
        lines.extend(
            [
                "",
                f"已评估检查点中最高开发集 F1 为 {best['eval/f1']:.4f}（更新 {best['update']}）。",
                "该值用于开发集检查点选择，不能作为独立测试集估计。",
            ]
        )
    lines.extend(
        [
            "",
            "所有曲线使用 W&B 原生标量及 line_series 绘制。",
            "本地查看：`bash scripts/view_wandb.sh`；最终汇总曲线可用 `--wandb` 生成。",
            "",
        ]
    )
    (directory / "RESULTS.md").write_text("\n".join(lines))


def log_wandb(directory, report, mode):
    import wandb

    config = json.loads((directory / "config.json").read_text())
    name = (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        + "-qwen35-9b-sec-search-drgrpo-results"
    )
    analysis_dir = directory / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    run = wandb.init(
        project="ARIN7101-Project",
        name=name,
        mode=mode,
        dir=str(analysis_dir.resolve()),
        config={**config, "source_training_run": report["run_name"]},
        group="qwen35-9b-sec-search",
        job_type="analysis",
        tags=["qwen3.5-9b", "sec-search", "results"],
    )
    run.define_metric("update")
    for prefix in ["train_raw/*", "eval/*", "optim/*"]:
        run.define_metric(prefix, step_metric="update")
    for row in report["history"]:
        run.log(row)
    charts = {
        "eval_quality": [("F1", "eval/f1"), ("F4", "eval/f_beta")],
        "training_reward": [("Reward", "train_raw/reward")],
        "optimization_loss": [("Loss", "optim/loss")],
    }
    for label, named_metrics in charts.items():
        series = {
            label_name: [
                (row["update"], row[name]) for row in report["history"] if name in row
            ]
            for label_name, name in named_metrics
        }
        series = {label_name: values for label_name, values in series.items() if values}
        if series:
            run.log(
                {
                    f"charts/{label}": wandb.plot.line_series(
                        xs=[[x for x, _ in values] for values in series.values()],
                        ys=[[y for _, y in values] for values in series.values()],
                        keys=list(series),
                        title={
                            "eval_quality": "Evaluation quality · F1 and F4",
                            "training_reward": "Training reward",
                            "optimization_loss": "Optimization loss",
                        }[label],
                        xname="Policy update",
                    )
                }
            )
    evaluations = [row for row in report["history"] if "eval/f1" in row]
    run.log(
        {
            "evaluation_table": wandb.Table(
                columns=["update", "F1", "F4", "precision", "recall"],
                data=[
                    [
                        row[k]
                        for k in [
                            "update",
                            "eval/f1",
                            "eval/f_beta",
                            "eval/precision",
                            "eval/recall",
                        ]
                    ]
                    for row in evaluations
                ],
            )
        }
    )
    run.summary.update(
        {
            "training_updates": len(report["completed_updates"]),
            "training_episodes": report["training_episodes_committed"],
            "local_full_audit_passed": report["local_completion_audit"][
                "training_and_local_charts_complete"
            ],
        }
    )
    artifact = wandb.Artifact(name + "-report", type="evaluation")
    for path in [
        directory / "results.json",
        directory / "RESULTS.md",
        directory / "config.json",
    ]:
        artifact.add_file(str(path))
    for path in sorted(directory.glob("epoch-*-audit.json")):
        artifact.add_file(str(path))
    for path in [directory / "token-audit.json", directory / "completion-audit.json"]:
        if path.exists():
            artifact.add_file(str(path))
    for path in [
        ROOT / "docs/model-integrity.json",
        ROOT / "docs/dataset-integrity.json",
    ]:
        if path.exists():
            artifact.add_file(str(path), name="provenance/" + path.name)
    for path in [
        *sorted((ROOT / "local_grpo").glob("*.py")),
        ROOT / "requirements-train.txt",
        ROOT / "requirements-vllm.txt",
    ]:
        artifact.add_file(str(path), name="source/" + str(path.relative_to(ROOT)))
    run.log_artifact(artifact)
    location = {
        "run_id": run.id,
        "name": name,
        "project": "ARIN7101-Project",
        "mode": mode,
        "files_dir": run.dir,
        "url": run.url if mode == "online" else None,
    }
    run.finish()
    (directory / "results-wandb.json").write_text(json.dumps(location, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--final",
        action="store_true",
        help="Require and inspect all full-training artifacts",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Render native W&B summary charts"
    )
    parser.add_argument(
        "--wandb-mode", choices=["offline", "online"], default="offline"
    )
    args = parser.parse_args()
    result = summarize(args.run_dir, final=args.final)
    write_report(args.run_dir, result)
    if args.wandb:
        log_wandb(args.run_dir, result, args.wandb_mode)
    print(
        json.dumps(
            {
                "updates": len(result["completed_updates"]),
                "episodes": result["training_episodes_committed"],
                "latest_f1": result["latest_evaluation"]["eval/f1"]
                if result["latest_evaluation"]
                else None,
                "complete": result["local_completion_audit"][
                    "training_and_local_charts_complete"
                ],
            },
            indent=2,
        )
    )
