import asyncio
import json
from pathlib import Path

import httpx
import torch
from sec_rl.config import HarnessConfig
from sec_rl.data import load_examples
from sec_rl.environment import make_environment
from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.rollouts import do_single_rollout
from transformers import AutoTokenizer

from local_grpo.optimize import importance_loss
from local_grpo.retrieval_pool import RetrievalPool
from local_grpo.rollouts import VllmTokenCompleter, collect, pack_trajectory


def test_importance_loss_matches_policy_gradient_and_masks_observations():
    logits = torch.tensor([[0.3, -0.4], [0.2, 0.1], [-0.1, 0.3]], requires_grad=True)
    labels = torch.tensor([0, 1, 0])
    logp = logits.log_softmax(-1).gather(1, labels[:, None]).squeeze(1)
    advantages = torch.tensor([0.7, -0.7, 0.0])
    loss = importance_loss(logp, logp.detach(), advantages)
    actual = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    expected = torch.autograd.grad(-(logp * advantages).sum(), logits)[0]
    torch.testing.assert_close(actual, expected)
    assert actual[0, 0] < 0 and actual[1, 1] > 0
    assert actual[2].abs().sum() == 0


def test_vllm_bridge_preserves_tokens_and_sampling_distribution():
    import tinker

    def handle(request):
        payload = json.loads(request.content)
        assert payload["prompt"] == [12, 34]
        assert payload["stop_token_ids"] == [99]
        assert payload["temperature"] == 1.0
        assert payload["top_p"] == 1.0 and payload["top_k"] == -1
        assert payload["max_tokens"] == 3
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "token_ids": [56, 99],
                        "logprobs": {"token_logprobs": [-0.2, -0.3]},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(handle)
        ) as client:
            policy = VllmTokenCompleter(
                client,
                {
                    "max_generation_tokens": 10,
                    "max_trajectory_tokens": 5,
                    "temperature": 1.0,
                },
                42,
            )
            result = await policy(tinker.ModelInput.from_ints([12, 34]), [99])
            assert result.tokens == [56, 99]
            assert result.logprobs == [-0.2, -0.3]

    asyncio.run(run())


def test_qwen_tool_episode_is_packed_once_and_masks_tool_results():
    tokenizer = AutoTokenizer.from_pretrained(
        "models/Qwen3.5-9B", local_files_only=True
    )
    renderer = get_renderer("qwen3_5", tokenizer)
    example = load_examples(Path("data/sec_subset"), "dev")[0]

    class FinishedPolicy(TokenCompleter):
        calls = 0

        async def __call__(self, model_input, stop, *, max_tokens=None):
            self.calls += 1
            function = (
                "<function=bm25_search>\n<parameter=query>financial company</parameter>\n</function>"
                if self.calls == 1
                else "<function=finish>\n</function>"
            )
            action = tokenizer.encode(
                "</think>\n\n<tool_call>\n" + function + "\n</tool_call><|im_end|>",
                add_special_tokens=False,
            )
            return TokensWithLogprobs(action, [-0.1] * len(action))

    async def run():
        env, session = make_environment(
            example=example,
            renderer=renderer,
            harness_config=HarnessConfig(data_dir=Path("data/sec_subset")),
            reward="f4",
        )
        trajectory = await do_single_rollout(FinishedPolicy(), env)
        assert session.finished
        assert sum(t.reward for t in trajectory.transitions) == -0.2
        packed = trajectory_to_data(trajectory, 0.5)
        assert len(packed) == 1
        weights = packed[0].loss_fn_inputs["advantages"].to_torch()
        assert len(trajectory.transitions) == 2
        assert weights.count_nonzero() == sum(
            len(t.ac.tokens) for t in trajectory.transitions
        )
        assert weights[0] == 0  # System, question and tool schema never train.
        unit = pack_trajectory(trajectory)
        assert len(unit) == len(packed)
        torch.testing.assert_close(unit[0]["advantages"] * 0.5, weights)
        torch.testing.assert_close(
            unit[0]["logprobs"], packed[0].loss_fn_inputs["logprobs"].to_torch()
        )

    asyncio.run(run())


def test_transport_retry_reuses_the_exact_prompt_and_seed():
    import tinker

    payloads = []

    def handle(request):
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            raise httpx.ReadError("temporary connection reset", request=request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "token_ids": [56, 99],
                        "logprobs": {"token_logprobs": [-0.2, -0.3]},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(handle)
        ) as client:
            policy = VllmTokenCompleter(
                client,
                {
                    "max_generation_tokens": 10,
                    "max_trajectory_tokens": 50,
                    "temperature": 1.0,
                    "transport_retry_base_seconds": 0,
                },
                42,
            )
            result = await policy(tinker.ModelInput.from_ints([12, 34]), [99])
            assert result.tokens == [56, 99]
            assert policy.calls == 1 and policy.retries == 1
            assert payloads[0] == payloads[1]

    asyncio.run(run())


def test_interrupted_collection_reuses_saved_episodes_without_sampling(
    tmp_path, monkeypatch
):
    tokenizer = AutoTokenizer.from_pretrained(
        "models/Qwen3.5-9B", local_files_only=True
    )
    action = tokenizer.encode(
        "</think>\n\n<tool_call>\n<function=finish>\n</function>\n</tool_call><|im_end|>",
        add_special_tokens=False,
    )
    calls = []

    class FinishedPolicy(TokenCompleter):
        def __init__(self, *args):
            self.retries = 0

        async def __call__(self, model_input, stop, *, max_tokens=None):
            calls.append(1)
            return TokensWithLogprobs(action, [-0.1] * len(action))

    monkeypatch.setattr("local_grpo.rollouts.VllmTokenCompleter", FinishedPolicy)
    config = json.loads(Path("configs/qwen35_9b_f4.json").read_text())
    first = asyncio.run(collect(config, tmp_path, 0, evaluate=True, queries=2, draws=1))
    assert len(calls) == 2
    (tmp_path / "metrics.json").unlink()  # Simulate interruption before batch commit.
    second = asyncio.run(
        collect(config, tmp_path, 0, evaluate=True, queries=2, draws=1)
    )
    assert len(calls) == 2
    assert second["eval/resumed_episodes"] == 2
    assert first["eval/f1"] == second["eval/f1"]
    assert len((tmp_path / "episodes.jsonl").read_text().splitlines()) == 2
    assert len(list((tmp_path / "episode-cache").glob("*.pt"))) == 2


def test_process_retrieval_matches_original_scores_snippets_and_document_reads():
    from sec_rl import environment
    from sec_rl.retrieval import CorpusIndex

    directory = Path("data/sec_subset").resolve()
    original = CorpusIndex(directory)
    query = "acquisition spring 2024 four facilities annual report"
    expected = original.bm25_search(query, k=5, snippet_chars=220)
    original_factory = environment.get_index
    with RetrievalPool(workers=2):
        pooled = environment.get_index(str(directory))
        assert pooled.bm25_search(query, k=5, snippet_chars=220) == expected
        assert pooled.grep("annual report", k=2, snippet_chars=220) == original.grep(
            "annual report", k=2, snippet_chars=220
        )
        assert pooled.read(expected[0].doc_id) == original.read(expected[0].doc_id)
    assert environment.get_index is original_factory
