"""Compare saved training batches with the real per-episode sampling caches."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import torch


def audit_tokens(directory: Path, through_step: int):
    torch.set_num_threads(1)
    config = json.loads((directory / "config.json").read_text())
    model_config = json.loads((Path(config["model_path"]) / "config.json").read_text())
    vocab_size = model_config["text_config"]["vocab_size"]
    if not 1 <= through_step <= config["max_steps"]:
        raise ValueError("Requested update is outside this experiment's budget")
    checks = []
    for update in range(1, through_step + 1):
        batch = directory / f"rollouts-{update:06d}"
        checkpoint = directory / f"checkpoint-{update:06d}"
        if not (checkpoint / "complete").exists():
            raise ValueError(f"Update {update} is not a complete checkpoint")
        rows = [json.loads(line) for line in (batch / "episodes.jsonl").open()]
        if len(rows) != config["groups_per_batch"] * config["group_size"]:
            raise ValueError(f"Incorrect episode count at update {update}")
        groups = defaultdict(list)
        for row in rows:
            groups[row["group"]].append(row)
        if set(groups) != set(range(config["groups_per_batch"])):
            raise ValueError(f"Incorrect group IDs at update {update}")
        packed = torch.load(batch / "packed.pt", map_location="cpu", weights_only=True)
        position, dropped, raw_action_tokens, training_action_tokens = 0, 0, 0, 0
        for group_id in sorted(groups):
            group = sorted(groups[group_id], key=lambda row: row["draw"])
            if [row["draw"] for row in group] != list(range(config["group_size"])):
                raise ValueError(f"Missing or repeated draws at update {update}")
            rewards = [row["reward"] for row in group]
            mean_reward = sum(rewards) / len(rewards)
            constant = max(rewards) == min(rewards)
            dropped += int(constant)
            for row in group:
                cache = (
                    batch
                    / "episode-cache"
                    / f"group-{group_id:03d}-draw-{row['draw']:02d}.pt"
                )
                if not cache.exists():
                    raise FileNotFoundError(cache)
                if constant:
                    continue
                saved = torch.load(cache, map_location="cpu", weights_only=True)
                assert saved["update"] == update - 1 and not saved["evaluate"]
                for key in ("query_id", "reward", "group", "draw"):
                    assert saved["summary"][key] == row[key], (update, group_id, key)
                for unit in saved["datums"]:
                    datum = packed[position]
                    position += 1
                    for key in ("input_ids", "target_tokens", "logprobs", "mask"):
                        assert datum[key].dtype == unit[key].dtype
                        assert torch.equal(datum[key], unit[key]), (
                            update,
                            group_id,
                            key,
                        )
                    expected = unit["advantages"] * (row["reward"] - mean_reward)
                    assert torch.equal(datum["advantages"], expected), (
                        update,
                        group_id,
                        "advantages",
                    )
                    ids, targets, mask = (
                        datum["input_ids"],
                        datum["target_tokens"],
                        datum["mask"],
                    )
                    assert 0 < len(ids) <= config["max_trajectory_tokens"]
                    assert all(tensor.numel() == len(ids) for tensor in datum.values())
                    assert ids.min() >= 0 and ids.max() < vocab_size
                    assert targets.min() >= 0 and targets.max() < vocab_size
                    assert torch.equal(ids[1:], targets[:-1]), (
                        update,
                        group_id,
                        "token shift",
                    )
                    assert torch.isfinite(datum["logprobs"]).all()
                    assert torch.isfinite(datum["advantages"]).all()
                    assert (datum["logprobs"] <= 1e-5).all()
                    assert ((mask == 0) | (mask == 1)).all()
                    assert (datum["advantages"][mask == 0] == 0).all()
                    raw_action_tokens += int(mask.sum())
                    training_action_tokens += int(datum["advantages"].count_nonzero())
        assert position == len(packed), (update, "unexpected training sequences")
        metrics = json.loads((batch / "metrics.json").read_text())
        optimizer = json.loads((checkpoint / "metrics.json").read_text())
        assert position == metrics["train_raw/training_sequences"]
        assert dropped == metrics["train_raw/groups_dropped_constant"]
        assert training_action_tokens == optimizer["optim/action_tokens"]
        check = {
            "update": update,
            "episodes": len(rows),
            "training_sequences": position,
            "constant_reward_groups_dropped": dropped,
            "masked_action_tokens": raw_action_tokens,
            "nonzero_advantage_action_tokens": training_action_tokens,
            "tokens_logprobs_masks_match_sampling_cache": True,
            "advantages_match_group_centered_rewards": True,
            "observation_tokens_masked": True,
            "optimizer_token_count_matches": True,
        }
        checks.append(check)
        print(json.dumps(check), flush=True)
    return {
        "audited_at": datetime.now(UTC).isoformat(),
        "audited_updates": list(range(1, through_step + 1)),
        "full_update_token_audit_complete": through_step == config["max_steps"],
        "full_24_update_token_audit_complete": through_step
        == config["max_steps"]
        == 24,
        "episodes_checked": sum(item["episodes"] for item in checks),
        "updates": checks,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--through-step", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.run_dir / "config.json").read_text())
    through_step = args.through_step or config["max_steps"]
    result = audit_tokens(args.run_dir, through_step)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
