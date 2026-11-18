"""Run the unchanged SEC environment against a local vLLM token API."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
from collections import defaultdict
from contextlib import AsyncExitStack
from pathlib import Path

import httpx
import torch
from sec_rl.config import HarnessConfig
from sec_rl.data import load_examples
from sec_rl.environment import _sec_train_order, make_environment
from sec_rl.rewards import score_submission
from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.rollouts import do_single_rollout
from transformers import AutoTokenizer

from local_grpo.retrieval_pool import RetrievalPool


class VllmTokenCompleter(TokenCompleter):
    def __init__(self, client, config, seed):
        self.client = client
        self.config = config
        self.seed = seed
        self.calls = 0
        self.retries = 0

    async def __call__(self, model_input, stop, *, max_tokens=None):
        tokens = [token for chunk in model_input.chunks for token in chunk.tokens]
        count = min(
            max_tokens or self.config["max_generation_tokens"],
            self.config["max_generation_tokens"],
            self.config["max_trajectory_tokens"] - len(tokens),
        )
        if count < 1:
            raise ValueError("Environment requested generation beyond its token budget")
        request = {
            "model": "local-search-agent",
            "prompt": tokens,
            "max_tokens": count,
            "temperature": self.config["temperature"],
            "top_p": 1.0,
            "top_k": -1,
            "logprobs": 1,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            "skip_special_tokens": False,
            "include_stop_str_in_output": True,
            "seed": self.seed + self.calls,
        }
        self.calls += 1
        if stop and isinstance(stop[0], int):
            request["stop_token_ids"] = stop
        else:
            request["stop"] = stop
        for attempt in range(4):
            try:
                response = await self.client.post("/v1/completions", json=request)
                response.raise_for_status()
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if isinstance(
                    exc, httpx.HTTPStatusError
                ) and exc.response.status_code not in (429, 502, 503, 504):
                    raise
                if attempt == 3:
                    raise
                self.retries += 1
                # Retry the identical prompt and seed; no extra sampled turn
                # is added to the trajectory on a failed HTTP transport.
                await asyncio.sleep(
                    self.config.get("transport_retry_base_seconds", 0.5) * 2**attempt
                )
        choice = response.json()["choices"][0]
        ids = choice["token_ids"]
        logprobs = choice["logprobs"]["token_logprobs"]
        if len(ids) != len(logprobs) or any(
            p is None or not math.isfinite(p) for p in logprobs
        ):
            raise ValueError(
                "vLLM must return one finite log probability per sampled token"
            )
        return TokensWithLogprobs(
            tokens=ids,
            maybe_logprobs=logprobs,
            stop_reason="length" if choice["finish_reason"] == "length" else "stop",
        )


def pack_trajectory(trajectory):
    """Save raw sampled tokens/logprobs before the complete group is available."""
    return [
        {
            "input_ids": torch.tensor(
                [t for c in datum.model_input.chunks for t in c.tokens]
            ),
            **{key: value.to_torch() for key, value in datum.loss_fn_inputs.items()},
        }
        for datum in trajectory_to_data(trajectory, 1.0)
    ]


async def collect(
    config,
    output,
    update,
    *,
    evaluate=False,
    queries=None,
    draws=None,
    url="http://127.0.0.1:8100",
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / "episode-cache"
    cache_dir.mkdir(exist_ok=True)
    episodes_path = output / "episodes.jsonl"
    if episodes_path.exists() and not (output / "metrics.json").exists():
        # Preserve interrupted attempts without counting them as new episodes.
        episodes_path.replace(output / f"episodes-interrupted-{time.time_ns()}.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"], local_files_only=True
    )
    renderer = get_renderer("qwen3_5", tokenizer)
    data_dir = Path(config["eval_data_dir"] if evaluate else config["data_dir"])
    harness = HarnessConfig(
        data_dir=data_dir,
        **{
            k: config[k]
            for k in (
                "max_turns",
                "max_tool_calls",
                "max_trajectory_tokens",
                "max_generation_tokens",
            )
        },
    )
    examples = load_examples(data_dir, "dev" if evaluate else "train")
    if evaluate:
        examples = examples[: queries or config["eval_queries"]]
        group_size = draws or config["eval_draws"]
    else:
        examples = _sec_train_order(examples, config["seed"])
        epoch, batch = divmod(
            update,
            (len(examples) + config["groups_per_batch"] - 1)
            // config["groups_per_batch"],
        )
        if epoch:
            random.Random(config["seed"] + epoch).shuffle(examples)
        start = batch * config["groups_per_batch"]
        examples = examples[start : start + (queries or config["groups_per_batch"])]
        group_size = draws or config["group_size"]
    semaphore = asyncio.Semaphore(config.get("rollout_concurrency", 128))
    completed = 0
    resumed = 0
    started = time.monotonic()
    async with AsyncExitStack() as stack:
        retrieval_workers = config.get("retrieval_processes", 8)
        if retrieval_workers:
            stack.enter_context(RetrievalPool(retrieval_workers))
        urls = [url] if isinstance(url, str) else url
        clients = [
            await stack.enter_async_context(
                httpx.AsyncClient(
                    base_url=endpoint,
                    timeout=1800,
                    trust_env=False,
                    limits=httpx.Limits(
                        max_connections=128,
                        max_keepalive_connections=64,
                        keepalive_expiry=1.0,
                    ),
                )
            )
            for endpoint in urls
        ]

        async def one(group, draw, example):
            nonlocal completed, resumed
            cache_path = cache_dir / f"group-{group:03d}-draw-{draw:02d}.pt"
            if cache_path.exists():
                cached = torch.load(cache_path, map_location="cpu", weights_only=True)
                summary = cached["summary"]
                if (
                    summary["query_id"] != example.query_id
                    or summary["group"] != group
                    or summary["draw"] != draw
                    or cached["update"] != update
                    or cached["evaluate"] != evaluate
                ):
                    raise ValueError(
                        f"Cached episode does not match this sampling task: {cache_path}"
                    )
                with episodes_path.open("a") as handle:
                    handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
                resumed += 1
                completed += 1
                return summary, cached["datums"]
            async with semaphore:
                policy = VllmTokenCompleter(
                    clients[(group * group_size + draw) % len(clients)],
                    config,
                    config["seed"]
                    + 1000000 * int(evaluate)
                    + 100000 * update
                    + group * 1000
                    + draw * 100,
                )
                env, session = make_environment(
                    example=example,
                    renderer=renderer,
                    harness_config=harness,
                    reward=config["reward"],
                )
                trajectory = await do_single_rollout(policy, env)
            metrics = (
                trajectory.transitions[-1].metrics if trajectory.transitions else {}
            )
            f1 = score_submission(
                session.curated_ids,
                example.qrels,
                beta=1.0,
                fact_groups=example.fact_groups,
            )
            metrics = {
                **metrics,
                "f1": f1.f_beta,
                "turns": len(trajectory.transitions),
                "sampled_tokens": sum(len(t.ac.tokens) for t in trajectory.transitions),
                "prefill_tokens": sum(t.ob.length for t in trajectory.transitions),
                "transport_retries": policy.retries,
            }
            sampled_logprobs = [
                p for t in trajectory.transitions for p in t.ac.logprobs
            ]
            metrics["entropy"] = (
                -sum(sampled_logprobs) / len(sampled_logprobs)
                if sampled_logprobs
                else 0.0
            )
            summary = {
                "group": group,
                "draw": draw,
                "query_id": example.query_id,
                "reward": sum(t.reward for t in trajectory.transitions),
                "metrics": metrics,
                "curated_ids": list(session.curated_ids),
                "stop_reason": str(trajectory.stop_reason),
                "messages": list(env.message_env.history),
            }
            summary = json.loads(json.dumps(summary, ensure_ascii=False, default=str))
            unit_datums = [] if evaluate else pack_trajectory(trajectory)
            temporary = cache_path.with_suffix(".partial")
            torch.save(
                {
                    "summary": summary,
                    "datums": unit_datums,
                    "update": update,
                    "evaluate": evaluate,
                },
                temporary,
            )
            temporary.replace(cache_path)
            with (output / "episodes.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(summary, ensure_ascii=False, default=str) + "\n"
                )
            completed += 1
            if completed % 8 == 0 or completed == len(examples) * group_size:
                print(
                    f"completed {completed}/{len(examples) * group_size} episodes in {time.monotonic() - started:.1f}s",
                    flush=True,
                )
            return summary, unit_datums

        results = await asyncio.gather(
            *(
                one(i, j, example)
                for i, example in enumerate(examples)
                for j in range(group_size)
            )
        )
    grouped = defaultdict(list)
    for summary, unit_datums in results:
        grouped[summary["group"]].append((summary, unit_datums))
    datums, dropped, kept_trajectories = [], 0, 0
    if not evaluate:
        for group in grouped.values():
            rewards = [summary["reward"] for summary, _ in group]
            if max(rewards) == min(rewards):
                dropped += 1
                continue
            mean = sum(rewards) / len(rewards)
            kept_trajectories += len(group)
            for summary, unit_datums in group:
                advantage = summary["reward"] - mean
                for datum in unit_datums:
                    datums.append(
                        {**datum, "advantages": datum["advantages"] * advantage}
                    )
        temporary = output / "packed.partial"
        torch.save(datums, temporary)
        temporary.replace(output / "packed.pt")
    summaries = [s for s, _ in results]
    names = sorted({k for s in summaries for k in s["metrics"]})
    prefix = "eval" if evaluate else "train_raw"
    metrics = {}
    for key in names:
        # Only explicitly conditioned metrics exclude episodes where absent.
        # Missing termination flags mean false, rather than a conditional 100%.
        denominator = (
            sum(key in s["metrics"] for s in summaries)
            if key.endswith("_given_finish")
            else len(summaries)
        )
        metrics[f"{prefix}/{key}"] = (
            sum(float(s["metrics"].get(key, 0)) for s in summaries) / denominator
        )
    metrics.update(
        {
            f"{prefix}/reward": sum(s["reward"] for s in summaries) / len(summaries),
            f"{prefix}/episodes": len(summaries),
            f"{prefix}/resumed_episodes": resumed,
            f"{prefix}/retrieval_processes": retrieval_workers,
            f"{prefix}/rollout_concurrency": config.get("rollout_concurrency", 128),
            f"{prefix}/wall_seconds": time.monotonic() - started,
        }
    )
    if not evaluate:
        metrics.update(
            {
                "train_raw/groups_dropped_constant": dropped,
                "train_raw/groups_total": len(grouped),
                "train_raw/training_sequences": len(datums),
                "train_raw/datums_per_trajectory": len(datums) / kept_trajectories
                if kept_trajectories
                else 0.0,
            }
        )
    action_tokens = sum(s["metrics"]["sampled_tokens"] for s in summaries)
    metrics[f"{prefix}/entropy"] = (
        sum(s["metrics"]["entropy"] * s["metrics"]["sampled_tokens"] for s in summaries)
        / action_tokens
        if action_tokens
        else 0.0
    )
    if evaluate and group_size > 1:
        for name in ["f1", "precision", "recall", "f_beta"]:
            means = [
                sum(
                    float(s["metrics"].get(name, 0))
                    for s in summaries
                    if s["draw"] == d
                )
                / len(examples)
                for d in range(group_size)
            ]
            metrics[f"eval/{name}_draw_std"] = float(torch.tensor(means).std())
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen35_9b_f4.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--update", type=int, default=0)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--queries", type=int)
    parser.add_argument("--draws", type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    asyncio.run(
        collect(
            config,
            args.output,
            args.update,
            evaluate=args.evaluate,
            queries=args.queries,
            draws=args.draws,
        )
    )


if __name__ == "__main__":
    main()
