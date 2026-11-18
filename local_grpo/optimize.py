"""One synchronous policy update over a complete, already sampled batch.

No clipping, KL term, reward standardization or within-batch optimizer steps.
Each GPU accumulates a disjoint set of trajectories; gradients are summed once.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5


def importance_loss(logprobs, old_logprobs, advantages):
    return -(torch.exp(logprobs - old_logprobs) * advantages).sum()


def backward_sequence(model, datum, denominator, *, chunk_size=256):
    base = model.get_base_model()
    device = next(base.parameters()).device
    ids = datum["input_ids"].to(device).unsqueeze(0)
    active = datum["advantages"].ne(0).nonzero(as_tuple=True)[0].to(device)
    if not active.numel():
        return 0.0, 0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = base.model.language_model(
            input_ids=ids, use_cache=False
        ).last_hidden_state[0]
        selected = hidden.index_select(0, active)
        targets = datum["target_tokens"].to(device).index_select(0, active)
        old = datum["logprobs"].to(device).index_select(0, active)
        advantages = datum["advantages"].to(device).index_select(0, active)

        def head_loss(h, target, old_logp, advantage):
            logits = base.lm_head(h).float()
            logp = -torch.nn.functional.cross_entropy(logits, target, reduction="none")
            return importance_loss(logp, old_logp, advantage) / denominator

        loss = sum(
            checkpoint(
                head_loss,
                selected[i : i + chunk_size],
                targets[i : i + chunk_size],
                old[i : i + chunk_size],
                advantages[i : i + chunk_size],
                use_reentrant=False,
            )
            for i in range(0, len(selected), chunk_size)
        )
    if not torch.isfinite(loss):
        raise FloatingPointError(
            "Non-finite importance sampling loss; refusing an invalid update"
        )
    loss.backward()
    return float(loss.detach()), len(active)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen35_9b_f4.json")
    parser.add_argument("--batch", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--previous")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    rank, world = (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group(
            "nccl",
            timeout=timedelta(hours=2),
            device_id=torch.device("cuda", local_rank),
        )
    torch.manual_seed(config["seed"])
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        config["model_path"],
        dtype=torch.bfloat16,
        device_map={"": local_rank},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    if args.previous:
        model = PeftModel.from_pretrained(
            base, str(Path(args.previous) / "adapter"), is_trainable=True
        )
    else:
        targets = [
            name
            for name, module in base.named_modules()
            if isinstance(module, torch.nn.Linear)
            and not name.startswith("model.visual")
        ]
        model = get_peft_model(
            base,
            LoraConfig(
                r=config["lora_rank"],
                lora_alpha=config["lora_alpha"],
                lora_dropout=config["lora_dropout"],
                target_modules=targets,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.config.use_cache = False
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=config["learning_rate"],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )
    if args.previous:
        optimizer.load_state_dict(
            torch.load(
                Path(args.previous) / "optimizer.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
    datums = torch.load(args.batch, map_location="cpu", weights_only=True)
    if not datums:
        raise ValueError("No variable-reward groups; no optimizer step can be taken")
    # Fixed-length Dr. GRPO normalization, independent of each sampled length.
    denominator = len(datums) * config["max_trajectory_tokens"]
    optimizer.zero_grad(set_to_none=True)
    loss_sum, tokens, started = 0.0, 0, time.monotonic()
    local_datums = datums[rank::world]
    for i, datum in enumerate(local_datums):
        print(
            f"rank {rank} starting sequence {i + 1}/{len(local_datums)} length={len(datum['input_ids'])}",
            flush=True,
        )
        loss, count = backward_sequence(model, datum, denominator)
        loss_sum += loss
        tokens += count
        print(
            f"rank {rank} sequence {i + 1}/{len(local_datums)} tokens={count} loss={loss:.6f} seconds={time.monotonic() - started:.1f}",
            flush=True,
        )
    # A manual reduction allows uneven trajectory counts without padding fake data.
    grad_flat = torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
            for p in params
        ]
    )
    if world > 1:
        dist.all_reduce(grad_flat, op=dist.ReduceOp.SUM)
    grad_norm = float(grad_flat.float().norm())
    if not torch.isfinite(grad_flat).all():
        raise FloatingPointError("Non-finite gradients")
    offset = 0
    for p in params:
        p.grad = grad_flat[offset : offset + p.numel()].view_as(p)
        offset += p.numel()
    optimizer.step()
    aggregate = torch.tensor([loss_sum, tokens], device=local_rank, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(aggregate)
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output / "adapter", save_embedding_layers=False)
        torch.save(optimizer.state_dict(), output / "optimizer.pt")
        metrics = {
            "optim/loss": float(aggregate[0]),
            "optim/action_tokens": int(aggregate[1]),
            "optim/grad_norm": grad_norm,
            "optim/trainable_parameters": sum(p.numel() for p in params),
            "optim/wall_seconds": time.monotonic() - started,
            "optim/world_size": world,
            "optim/learning_rate": config["learning_rate"],
            "optim/native_causal_conv": float(
                modeling_qwen3_5.causal_conv1d_fn is not None
            ),
        }
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        # A merged snapshot lets vLLM sample every trained linear projection,
        # including GatedDeltaNet and lm_head, without partial LoRA support.
        merged = model.merge_and_unload()
        merged.config.use_cache = True
        merged.save_pretrained(output / "sampler", max_shard_size="5GB")
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"], local_files_only=True
        )
        tokenizer.save_pretrained(output / "sampler")
        (output / "complete").touch()
        print(json.dumps(metrics), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
