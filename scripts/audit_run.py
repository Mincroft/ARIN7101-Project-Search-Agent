"""Inspect actual artifacts against the full Qwen main-training requirements.

An incomplete or smoke run cannot pass this audit. This is evidence for review,
not a substitute for inspecting the experiment and W&B results.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from sec_rl.data import load_examples
from sec_rl.environment import _sec_train_order
from sec_rl.rewards import score_submission


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def audit(directory, *, inspect_optimizer=False):
    config = read_json(directory / "config.json") or {}
    state = read_json(directory / "state.json") or {}
    evidence = []

    def check(requirement, passed, details):
        evidence.append(
            {"requirement": requirement, "passed": bool(passed), "evidence": details}
        )

    expected = {
        "model": "Qwen/Qwen3.5-9B",
        "groups_per_batch": 64,
        "group_size": 8,
        "eval_queries": 32,
        "eval_draws": 2,
        "reward": "f4",
        "learning_rate": 1e-4,
        "lora_rank": 32,
        "max_trajectory_tokens": 30720,
        "max_generation_tokens": 2048,
        "max_turns": 40,
        "max_tool_calls": 128,
        "gpus": 8,
        "reward_standardization": False,
        "ratio_clipping": False,
        "kl_coefficient": 0,
        "optimizer_steps_per_batch": 1,
    }
    differences = {
        k: {"expected": value, "actual": config.get(k)}
        for k, value in expected.items()
        if config.get(k) != value
    }
    check("Full requested Qwen main configuration", not differences, differences)
    check(
        "Approved complete main-training budget",
        config.get("max_steps") in (24, 32, 48, 80),
        {"max_steps": config.get("max_steps")},
    )
    check(
        "W&B project and timestamped keyword name",
        config.get("wandb_project") == "ARIN7101-Project"
        and re.fullmatch(
            r"\d{8}-\d{6}-qwen35-9b-sec-search-drgrpo-f4-main",
            config.get("run_name", ""),
        ),
        {k: config.get(k) for k in ("wandb_project", "run_name")},
    )
    for name, count, documents in [
        ("sec_1024", 1024, 237533),
        ("sec_subset", 256, 124395),
    ]:
        summary = read_json(Path("data") / name / "dataset.json") or {}
        check(
            f"{name} data size and split isolation",
            summary.get("documents") == documents
            and summary.get("queries") == {"train": count, "dev": 64}
            and summary.get("split_guards")
            == {
                "gold_chunk_overlap": 0,
                "normalized_query_text_overlap": 0,
                "query_family_overlap": 0,
            },
            summary,
        )
    examples = {
        example.query_id: example
        for example in load_examples(
            Path(config.get("data_dir", "data/sec_1024")), "train"
        )
    }
    dev = load_examples(Path(config.get("eval_data_dir", "data/sec_subset")), "dev")[
        :32
    ]
    counts, failures = {}, []

    def inspect_episodes(
        folder, groups, draws, allowed, *, expected_dev=False, expected_ids=None
    ):
        path = folder / "episodes.jsonl"
        if not path.exists():
            return {"count": 0, "complete": False, "issues": ["missing episodes.jsonl"]}
        seen, query_groups, rows, issues = Counter(), {}, [], []
        with path.open() as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    issues.append("incomplete or invalid JSON record")
                    continue
                rows.append(row)
                key = (row["group"], row["draw"])
                seen[key] += 1
                qid = row["query_id"]
                if row["group"] in query_groups and query_groups[row["group"]] != qid:
                    issues.append("group mixes questions")
                query_groups[row["group"]] = qid
                if qid not in allowed:
                    issues.append("question outside requested split")
                    continue
                example = allowed[qid]
                f1 = score_submission(
                    row["curated_ids"],
                    example.qrels,
                    beta=1,
                    fact_groups=example.fact_groups,
                )
                if not math.isclose(row["metrics"]["f1"], f1.f_beta, abs_tol=1e-8):
                    issues.append("recorded F1 does not match the curated evidence")
                f4 = score_submission(
                    row["curated_ids"],
                    example.qrels,
                    beta=4,
                    fact_groups=example.fact_groups,
                )
                for name, value in [
                    ("f_beta", f4.f_beta),
                    ("precision", f4.precision),
                    ("recall", f4.recall),
                ]:
                    if name in row["metrics"] and not math.isclose(
                        row["metrics"][name], value, abs_tol=1e-8
                    ):
                        issues.append(
                            f"recorded {name} does not match the curated evidence"
                        )
                if not math.isfinite(row["reward"]):
                    issues.append("non-finite episode reward")
        wanted = {(g, d) for g in range(groups) for d in range(draws)}
        if set(seen) != wanted or any(n != 1 for n in seen.values()):
            issues.append("missing or repeated group/draw pairs")
        if len(set(query_groups.values())) != groups:
            issues.append("incorrect number of distinct questions")
        if expected_ids is not None and query_groups != dict(enumerate(expected_ids)):
            issues.append("training batch does not match the upstream epoch order")
        if expected_dev and query_groups != {
            g: example.query_id for g, example in enumerate(dev)
        }:
            issues.append("evaluation does not use the fixed first 32 dev questions")
        metrics = read_json(folder / "metrics.json")
        if metrics is None:
            issues.append("missing aggregate metrics")
        elif rows:
            prefix = "eval" if expected_dev else "train_raw"
            for key in ("reward", "f1"):
                calculated = sum(
                    r["reward"] if key == "reward" else r["metrics"]["f1"] for r in rows
                ) / len(rows)
                if not math.isclose(
                    metrics.get(f"{prefix}/{key}", math.nan), calculated, abs_tol=1e-8
                ):
                    issues.append(f"aggregate {key} does not match episodes")
        return {
            "count": len(rows),
            "complete": not issues,
            "issues": sorted(set(issues)),
        }

    train_order = _sec_train_order(list(examples.values()), config.get("seed", 42))
    max_steps = int(config.get("max_steps", 0))
    batches_per_epoch = (len(examples) + config["groups_per_batch"] - 1) // config[
        "groups_per_batch"
    ]
    for update in range(1, max_steps + 1):
        folder = directory / f"rollouts-{update:06d}"
        epoch, batch = divmod(update - 1, batches_per_epoch)
        epoch_order = list(train_order)
        if epoch:
            random.Random(config.get("seed", 42) + epoch).shuffle(epoch_order)
        expected_ids = [
            example.query_id for example in epoch_order[batch * 64 : (batch + 1) * 64]
        ]
        result = inspect_episodes(folder, 64, 8, examples, expected_ids=expected_ids)
        counts[str(update)] = result["count"]
        if not result["complete"]:
            failures.append({"update": update, **result})
    check(
        f"{max_steps} complete rollout batches",
        not failures
        and sum(counts.values())
        == max_steps * config["groups_per_batch"] * config["group_size"],
        {"counts": counts, "failures": failures},
    )
    evals = {}
    for update in range(0, max_steps + 1, config["eval_every"]):
        evals[str(update)] = inspect_episodes(
            directory / f"eval-{update:06d}",
            32,
            2,
            {e.query_id: e for e in dev},
            expected_dev=True,
        )
    check(
        "Baseline and all requested checkpoint evaluations: 32 questions x 2 draws",
        all(result["complete"] for result in evals.values()),
        evals,
    )
    completed = [
        i
        for i in range(1, max_steps + 1)
        if (directory / f"checkpoint-{i:06d}/complete").exists()
    ]
    missing_states = [
        i
        for i in completed
        if not all(
            (directory / f"checkpoint-{i:06d}" / suffix).exists()
            for suffix in [
                "optimizer.pt",
                "adapter/adapter_model.safetensors",
                "metrics.json",
            ]
        )
    ]
    check(
        f"{max_steps} saved optimizer and adapter checkpoints",
        len(completed) == max_steps and not missing_states,
        {"completed": completed, "missing_states": missing_states},
    )
    optimizer_metrics = {}
    for update in completed:
        metrics = read_json(directory / f"checkpoint-{update:06d}/metrics.json") or {}
        norm = metrics.get("optim/grad_norm", math.nan)
        optimizer_metrics[str(update)] = (
            metrics.get("optim/world_size") == 8
            and metrics.get("optim/action_tokens", 0) > 0
            and math.isfinite(norm)
            and norm > 0
            and metrics.get("optim/learning_rate") == 1e-4
        )
    check(
        f"{max_steps} finite nonzero gradient updates using all 8 GPUs",
        len(optimizer_metrics) == max_steps and all(optimizer_metrics.values()),
        optimizer_metrics,
    )
    final = directory / f"checkpoint-{max_steps:06d}"
    index = read_json(final / "sampler/model.safetensors.index.json") or {}
    shards = set(index.get("weight_map", {}).values())
    check(
        "Final full model snapshot",
        bool(shards) and all((final / "sampler" / s).exists() for s in shards),
        {"shards": sorted(shards), "tensor_count": len(index.get("weight_map", {}))},
    )
    if inspect_optimizer:
        import torch

        path = final / "optimizer.pt"
        steps = set()
        if path.exists():
            optimizer = torch.load(path, weights_only=True, map_location="cpu")
            steps = {int(value["step"]) for value in optimizer["state"].values()}
        check(
            f"Final optimizer reached exactly update {max_steps}",
            steps == {max_steps},
            {"steps": sorted(steps)},
        )
    else:
        check(
            f"Final optimizer reached exactly update {max_steps}",
            False,
            "Not inspected; use --inspect-optimizer for the final completion audit",
        )
    charts = list(directory.glob("wandb/*/files/media/table/charts/*.table.json"))
    chart_names = {p.name.split("_table_")[0] for p in charts}
    check(
        "W&B native concise chart data",
        {"eval_quality", "training_reward", "optimization_loss"} <= chart_names
        or {"retrieval", "training", "behavior"} <= chart_names,
        {"chart_families": sorted(chart_names), "table_count": len(charts)},
    )
    check(
        f"Controller finished all {max_steps} updates",
        state.get("status") == "completed"
        and state.get("completed_updates") == max_steps,
        state,
    )
    return {
        "audited_at": datetime.now(UTC).isoformat(),
        "run_dir": str(directory),
        "scope": "User-requested Qwen3.5-9B local plain-F4 main training; differs from the blog model/backend",
        "training_and_local_charts_complete": all(item["passed"] for item in evidence),
        "wandb_cloud_sync": "not verified by this local audit",
        "requirements": evidence,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inspect-optimizer", action="store_true")
    args = parser.parse_args()
    report = audit(args.run_dir, inspect_optimizer=args.inspect_optimizer)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(
        json.dumps(
            {
                "complete": report["training_and_local_charts_complete"],
                "passed": sum(r["passed"] for r in report["requirements"]),
                "requirements": len(report["requirements"]),
                "wandb_cloud_sync": report["wandb_cloud_sync"],
            },
            indent=2,
        )
    )
