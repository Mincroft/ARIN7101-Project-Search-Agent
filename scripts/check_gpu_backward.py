"""Exercise the real hybrid-attention + LoRA loss on a tiny GPU model."""

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from local_grpo.optimize import backward_sequence

torch.cuda.set_device(7)
torch.manual_seed(42)
config = Qwen3_5Config.from_pretrained("models/Qwen3.5-9B", local_files_only=True)
text = config.text_config
text.hidden_size = 128
text.intermediate_size = 256
text.num_hidden_layers = 2
text.num_attention_heads = 2
text.num_key_value_heads = 1
text.head_dim = 64
text.vocab_size = 256
text.layer_types = ["linear_attention", "full_attention"]
text.linear_num_key_heads = 2
text.linear_num_value_heads = 2
text.linear_key_head_dim = 16
text.linear_value_head_dim = 16
config.vision_config.depth = 1
config.vision_config.hidden_size = 128
config.vision_config.intermediate_size = 256
config.vision_config.out_hidden_size = 128
config.vision_config.num_heads = 4
base = Qwen3_5ForConditionalGeneration(config).to(device="cuda", dtype=torch.bfloat16)
targets = [
    name
    for name, module in base.named_modules()
    if isinstance(module, torch.nn.Linear) and not name.startswith("model.visual")
]
model = get_peft_model(
    base, LoraConfig(r=4, lora_alpha=4, target_modules=targets, task_type="CAUSAL_LM")
)
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
model.enable_input_require_grads()
model.train()
datum = {
    "input_ids": torch.randint(0, 256, (64,)),
    "target_tokens": torch.randint(0, 256, (64,)),
    "logprobs": torch.full((64,), -5.6),
    "advantages": torch.cat([torch.zeros(32), torch.ones(32)]),
}
loss, tokens = backward_sequence(model, datum, 64, chunk_size=8)
norm = (
    sum(
        float(p.grad.float().square().sum())
        for p in model.parameters()
        if p.grad is not None
    )
    ** 0.5
)
assert tokens == 32 and norm > 0 and torch.isfinite(torch.tensor(loss))
print(
    {"loss": loss, "action_tokens": tokens, "gradient_norm": norm, "passed": True},
    flush=True,
)
