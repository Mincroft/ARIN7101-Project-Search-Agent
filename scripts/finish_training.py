"""Wait for a training controller, audit its output, and publish the final table."""

from __future__ import annotations

import argparse
import json
import math
import netrc
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx


def write_status(path, status, **fields):
    data = {"status": status, "updated_at": datetime.now(UTC).isoformat(), **fields}
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps(data), flush=True)


def finalize(directory):
    state = json.loads((directory / "state.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    if state.get("status") != "completed":
        raise RuntimeError("Training did not finish; refusing to publish final results")
    status_path = directory / "finalization-state.json"
    root = Path(__file__).resolve().parents[1]
    for label, command in [
        (
            "auditing_tokens",
            [
                "-m",
                "scripts.audit_tokens",
                str(directory),
                "--output",
                str(directory / "token-audit.json"),
            ],
        ),
        (
            "auditing_checkpoints",
            [
                "-m",
                "scripts.audit_run",
                str(directory),
                "--inspect-optimizer",
                "--output",
                str(directory / "completion-audit.json"),
            ],
        ),
        ("summarizing", ["-m", "scripts.summarize_run", str(directory), "--final"]),
    ]:
        write_status(status_path, label)
        subprocess.run([sys.executable, *command], check=True, cwd=root)
    report = json.loads((directory / "results.json").read_text())
    if not report["local_completion_audit"]["training_and_local_charts_complete"]:
        raise RuntimeError("Local completion audit failed")

    import wandb

    write_status(status_path, "publishing_final_summary")
    run = wandb.init(
        entity=os.environ["WANDB_ENTITY"],
        project=config["wandb_project"],
        id=state["wandb_id"],
        resume="must",
        mode="online",
        dir=str(directory.resolve()),
        settings=wandb.Settings(x_disable_stats=True),
    )
    evaluations = [row for row in report["history"] if "eval/f1" in row]
    best = max(evaluations, key=lambda row: row["eval/f1"])
    last = evaluations[-1]
    run.log(
        {
            "evaluation_table": wandb.Table(
                columns=["Update", "F1", "F4", "Precision", "Recall"],
                data=[
                    [
                        r[k]
                        for k in [
                            "update",
                            "eval/f1",
                            "eval/f_beta",
                            "eval/precision",
                            "eval/recall",
                        ]
                    ]
                    for r in evaluations
                ],
            )
        }
    )
    run.summary.update(
        {
            "best_eval_f1": best["eval/f1"],
            "best_eval_f1_update": best["update"],
            "best_eval_f4": max(row["eval/f_beta"] for row in evaluations),
            "final_eval_f1": last["eval/f1"],
            "final_eval_f4": last["eval/f_beta"],
            "training_updates": config["max_steps"],
            "training_episodes": report["training_episodes_committed"],
            "full_training_audit_passed": True,
        }
    )
    artifact = wandb.Artifact(
        config["run_name"] + "-audited-results", type="evaluation"
    )
    for name in [
        "config.json",
        "results.json",
        "RESULTS.md",
        "token-audit.json",
        "completion-audit.json",
        "wandb-dashboard.json",
        "source-snapshot.json",
    ]:
        path = directory / name
        if path.exists():
            artifact.add_file(str(path))
    run.log_artifact(artifact)
    run.finish()
    base = os.environ["WANDB_BASE_URL"].rstrip("/")
    credentials = netrc.netrc().authenticators(urlsplit(base).hostname)
    key = os.getenv("WANDB_API_KEY") or (credentials[2] if credentials else None)
    query = """
    query VerifyRun($entity: String!, $project: String!, $run: String!) {
      project(entityName: $entity, name: $project) {
        run(name: $run) { state summaryMetrics config }
      }
    }
    """
    with httpx.Client(proxy=os.getenv("HTTPS_PROXY"), trust_env=False, timeout=30) as c:
        response = c.post(
            base + "/graphql",
            json={
                "query": query,
                "variables": {
                    "entity": os.environ["WANDB_ENTITY"],
                    "project": config["wandb_project"],
                    "run": state["wandb_id"],
                },
            },
            auth=("api", key),
        )
        response.raise_for_status()
        body = response.json()
        if body.get("errors"):
            raise RuntimeError(body["errors"])
        cloud = body["data"]["project"]["run"]
    summary = json.loads(cloud["summaryMetrics"])
    if cloud["state"] != "finished" or not math.isclose(
        summary["final_eval_f1"], last["eval/f1"], abs_tol=1e-12
    ):
        raise RuntimeError("Cloud completion or final F1 differs from the local report")
    if summary["training_episodes"] != config["max_steps"] * 64 * 8:
        raise RuntimeError(
            "Cloud training episode count differs from the requested budget"
        )
    write_status(
        status_path, "completed", wandb_id=state["wandb_id"], cloud_run_verified=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    status_path = directory / "finalization-state.json"
    state = json.loads((directory / "state.json").read_text())
    try:
        if args.wait:
            pid = state["pid"]
            write_status(status_path, "waiting_for_training", controller_pid=pid)
            while True:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(30)
        finalize(directory)
    except BaseException as exc:
        write_status(status_path, "failed", error_type=type(exc).__name__)
        raise


if __name__ == "__main__":
    main()
