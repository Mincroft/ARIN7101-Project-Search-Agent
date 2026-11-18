"""Manage the complete local run, evaluations, checkpoints and W&B."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import httpx

import wandb
from local_grpo.rollouts import collect

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temporary.replace(path)


class Server:
    def __init__(self, model, log_dir, gpus=8, port=8100):
        self.url = [f"http://127.0.0.1:{port + gpu}" for gpu in range(gpus)]
        self.processes, self.handles = [], []
        command = [
            str(ROOT / ".venv-vllm/bin/python"),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(Path(model).resolve()),
            "--tokenizer",
            str((ROOT / "models/Qwen3.5-9B").resolve()),
            "--served-model-name",
            "local-search-agent",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "30720",
            "--data-parallel-size",
            "1",
            "--tensor-parallel-size",
            "1",
            "--gpu-memory-utilization",
            "0.85",
            "--max-num-seqs",
            "64",
            "--max-num-batched-tokens",
            "8192",
            "--enable-prefix-caching",
            "--enforce-eager",
            "--no-enable-log-requests",
            "--language-model-only",
        ]
        # Independent replicas avoid a vLLM dense-DP rendezvous-port race.
        # Sampling distribution and policy weights are identical on every GPU.
        for gpu in range(gpus):
            handle = (log_dir / f"vllm-gpu-{gpu}.log").open("a")
            self.handles.append(handle)
            gpu_command = list(command)
            gpu_command[gpu_command.index("--port") + 1] = str(port + gpu)
            environment = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "VLLM_PORT": str(22000 + gpu * 100),
                "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "60",
            }
            self.processes.append(
                subprocess.Popen(
                    gpu_command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    cwd=ROOT,
                    env=environment,
                )
            )
        (log_dir / "vllm.pid").write_text(json.dumps([p.pid for p in self.processes]))

    def ready(self):
        deadline = time.monotonic() + 1200
        with httpx.Client(trust_env=False, timeout=3) as client:
            while time.monotonic() < deadline:
                ready = 0
                for gpu, process in enumerate(self.processes):
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"vLLM GPU {gpu} exited with {process.returncode}; see vllm-gpu-{gpu}.log"
                        )
                    try:
                        ready += (
                            client.get(self.url[gpu] + "/health").status_code == 200
                        )
                    except httpx.HTTPError:
                        pass
                if ready == len(self.processes):
                    return
                time.sleep(3)
        raise TimeoutError("vLLM did not become ready within 20 minutes")

    def stop(self):
        for process in self.processes:
            # Clean orphaned workers even if the server parent already failed.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in self.processes:
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
        for handle in self.handles:
            handle.close()


def chart_history(run, history):
    # Keep the dashboard readable: one decision-relevant chart per question.
    charts = {
        "eval_quality": {
            "title": "Evaluation quality · F1 and F4",
            "names": [("F1", "eval/f1"), ("F4", "eval/f_beta")],
            "predicate": lambda row: "eval/f1" in row,
        },
        "training_reward": {
            "title": "Training reward",
            "names": [("Reward", "train_raw/reward")],
            "predicate": lambda row: "train_raw/reward" in row,
        },
        "optimization_loss": {
            "title": "Optimization loss",
            "names": [("Loss", "optim/loss")],
            "predicate": lambda row: "optim/loss" in row,
        },
    }
    for label, spec in charts.items():
        rows = [row for row in history if spec["predicate"](row)]
        if not rows:
            continue
        xs, ys, keys = [], [], []
        for key, name in spec["names"]:
            points = [(row["update"], row[name]) for row in rows if name in row]
            if points:
                keys.append(key)
                xs.append([p[0] for p in points])
                ys.append([p[1] for p in points])
        if keys:
            run.log(
                {
                    f"charts/{label}": wandb.plot.line_series(
                        xs,
                        ys,
                        keys=keys,
                        title=spec["title"],
                        xname="Policy update",
                    )
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen35_9b_f4.json")
    parser.add_argument("--run-dir")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline"], default="offline"
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    config = json.loads(Path(args.config).read_text())
    if args.smoke:
        config = {
            **config,
            "groups_per_batch": 2,
            "group_size": 4,
            "max_steps": 1,
            "eval_queries": 2,
            "eval_draws": 1,
        }
    name = (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        + "-qwen35-9b-sec-search-drgrpo-f4"
        + ("-smoke" if args.smoke else "-main")
    )
    directory = Path(args.run_dir) if args.run_dir else Path("runs") / name
    directory.mkdir(parents=True, exist_ok=True)
    saved_config = directory / "config.json"
    if saved_config.exists():
        config = json.loads(saved_config.read_text())
        name = config["run_name"]
    else:
        config.update(
            run_name=name,
            gpus=args.gpus,
            renderer="qwen3_5",
            smoke=args.smoke,
            reward_standardization=False,
            kl_coefficient=0,
            ratio_clipping=False,
            optimizer_steps_per_batch=1,
            model_dtype="bfloat16",
            model_change="User requested Qwen3.5-9B instead of gpt-oss-20b",
        )
        write_json(saved_config, config)
    existing_state = (
        json.loads((directory / "state.json").read_text())
        if (directory / "state.json").exists()
        else {}
    )
    resume_id = existing_state.get("wandb_id")
    run = wandb.init(
        project="ARIN7101-Project",
        name=name,
        config=config,
        mode=args.wandb_mode,
        dir=str(directory.resolve()),
        id=resume_id,
        resume="must" if resume_id else None,
        tags=["qwen3.5-9b", "sec-search", "dr-grpo", "smoke" if args.smoke else "main"],
        group="qwen35-9b-sec-search",
        job_type="train",
    )
    run.define_metric("update")
    run.define_metric("train_raw/*", step_metric="update")
    run.define_metric("eval/*", step_metric="update")
    run.define_metric("optim/*", step_metric="update")
    state = {
        "run_name": name,
        "pid": os.getpid(),
        "wandb_id": run.id,
        "wandb_mode": args.wandb_mode,
        "status": "starting",
        "completed_updates": 0,
    }
    history_path = directory / "history.jsonl"
    history = (
        [json.loads(line) for line in history_path.read_text().splitlines()]
        if history_path.exists()
        else []
    )

    def log(metrics, update):
        row = {"update": update, **metrics}
        history.append(row)
        with history_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        run.log(row)
        chart_history(run, history)

    def phase(text, **kwargs):
        state.update(status=text, **kwargs, updated_at=datetime.now(UTC).isoformat())
        write_json(directory / "state.json", state)
        print(json.dumps(state), flush=True)

    previous = None
    for update in range(1, config["max_steps"] + 1):
        checkpoint_dir = directory / f"checkpoint-{update:06d}"
        if (checkpoint_dir / "complete").exists():
            previous = checkpoint_dir
            state["completed_updates"] = update
        else:
            break
    server = None
    try:
        artifact = wandb.Artifact(
            "sec-search-reproduction-inputs", type="experiment-config"
        )
        artifact.add_file(str(saved_config))
        for data_dir in [config["data_dir"], config["eval_data_dir"]]:
            artifact.add_file(
                str(Path(data_dir) / "dataset.json"), name=Path(data_dir).name + ".json"
            )
        run.log_artifact(artifact)
        # Recover an interrupted evaluation before deleting its only sampler
        # snapshot or declaring a final checkpoint complete.
        completed = state["completed_updates"]
        if previous and (
            completed % config["eval_every"] == 0 or completed == config["max_steps"]
        ):
            eval_dir = directory / f"eval-{completed:06d}"
            if not (eval_dir / "metrics.json").exists():
                phase("resuming_evaluation", checkpoint=str(previous))
                server = Server(previous / "sampler", directory, args.gpus)
                server.ready()
                log(
                    asyncio.run(
                        collect(
                            config, eval_dir, completed, evaluate=True, url=server.url
                        )
                    ),
                    completed,
                )
        for update in range(state["completed_updates"], config["max_steps"]):
            if server is None:
                phase("starting_sampler", next_update=update + 1)
                server = Server(
                    previous / "sampler" if previous else config["model_path"],
                    directory,
                    args.gpus,
                )
                server.ready()
            if update == 0 and not (directory / "eval-000000/metrics.json").exists():
                phase("baseline_evaluation")
                metrics = asyncio.run(
                    collect(
                        config,
                        directory / "eval-000000",
                        0,
                        evaluate=True,
                        url=server.url,
                    )
                )
                log(metrics, 0)
            batch = directory / f"rollouts-{update + 1:06d}"
            phase("sampling", next_update=update + 1)
            if not (batch / "metrics.json").exists():
                metrics = asyncio.run(collect(config, batch, update, url=server.url))
            else:
                metrics = json.loads((batch / "metrics.json").read_text())
            log(metrics, update + 1)
            server.stop()
            server = None
            phase("optimizing", next_update=update + 1)
            checkpoint_dir = directory / f"checkpoint-{update + 1:06d}"
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={args.gpus}",
                "-m",
                "local_grpo.optimize",
                "--config",
                str(saved_config),
                "--batch",
                str(batch / "packed.pt"),
                "--output",
                str(checkpoint_dir),
            ]
            if previous:
                command += ["--previous", str(previous)]
            with (directory / f"optimize-{update + 1:06d}.log").open("a") as handle:
                subprocess.run(
                    command, check=True, stdout=handle, stderr=subprocess.STDOUT
                )
            log(json.loads((checkpoint_dir / "metrics.json").read_text()), update + 1)
            state["completed_updates"] = update + 1
            if previous and (previous / "sampler").exists():
                shutil.rmtree(previous / "sampler")
            previous = checkpoint_dir
            if (update + 1) % config["eval_every"] == 0 or update + 1 == config[
                "max_steps"
            ]:
                phase("starting_eval_sampler", checkpoint=str(previous))
                server = Server(previous / "sampler", directory, args.gpus)
                server.ready()
                phase("evaluating", checkpoint=str(previous))
                metrics = asyncio.run(
                    collect(
                        config,
                        directory / f"eval-{update + 1:06d}",
                        update + 1,
                        evaluate=True,
                        url=server.url,
                    )
                )
                log(metrics, update + 1)
        artifact = wandb.Artifact(name + "-final-adapter", type="model")
        artifact.add_dir(str(previous / "adapter"))
        run.log_artifact(artifact)
        evaluations = [row for row in history if "eval/f1" in row]
        if evaluations:
            best = max(evaluations, key=lambda row: row["eval/f1"])
            final = evaluations[-1]
            run.summary.update(
                {
                    "best_eval_f1": best["eval/f1"],
                    "best_eval_f1_update": best["update"],
                    "best_eval_f4": max(row["eval/f_beta"] for row in evaluations),
                    "final_eval_f1": final["eval/f1"],
                    "final_eval_f4": final["eval/f_beta"],
                    "training_updates": config["max_steps"],
                    "training_episodes": config["max_steps"]
                    * config["groups_per_batch"]
                    * config["group_size"],
                }
            )
        phase("completed", final_checkpoint=str(previous))
        run.summary["completed_updates"] = config["max_steps"]
        run.finish()
    except BaseException as exc:
        phase("failed", exception=repr(exc))
        traceback.print_exc()
        run.finish(exit_code=1)
        raise
    finally:
        if server:
            server.stop()


if __name__ == "__main__":
    main()
