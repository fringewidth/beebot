"""The think-then-act voxel policy (voxel/think.py) on a tiny randomly initialised Qwen."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")

from beebot.vla.voxel.lm import PATCH  # noqa: E402
from beebot.vla.voxel.think import ThinkVoxelPolicy, chat, chunked_cross_entropy, pack  # noqa: E402
from beebot.vla.voxel.think_fake import REFUSAL, Example, FakeThinkData  # noqa: E402
from beebot.vla.voxel.world import N_ACTIONS  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"


def tiny(tokenizer, **kwargs):
    config = transformers.AutoConfig.from_pretrained(MODEL, local_files_only=True)
    config.hidden_size, config.intermediate_size, config.num_hidden_layers = 32, 64, 2
    config.num_attention_heads, config.num_key_value_heads, config.head_dim = 4, 2, 8
    torch.manual_seed(0)
    return ThinkVoxelPolicy(transformers.AutoModelForCausalLM.from_config(config).float(), tokenizer, **kwargs)


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except OSError:
        pytest.skip("the Qwen tokenizer is not cached")


@pytest.fixture(scope="module")
def policy(tokenizer):
    return tiny(tokenizer).eval()


def examples(n, seed=0, **kwargs):
    data = FakeThinkData(side=8, instruction_words=3, thought_words=6, robot_share=0.2, **kwargs)
    rng = np.random.default_rng(seed)
    return [data.sample(rng) for _ in range(n)]


def test_pack_fits_rows():
    lengths = [5, 3, 4, 2, 9]
    rows = pack(lengths, 7)
    assert sorted(i for r in rows for i in r) == list(range(5))
    assert all(sum(lengths[i] for i in r) <= 7 for r in rows if len(r) > 1)
    assert pack(lengths, 0) == [[0], [1], [2], [3], [4]]


def test_chunked_cross_entropy_equals_full():
    torch.manual_seed(0)
    head = torch.nn.Linear(16, 50, bias=False)
    states = torch.randn(23, 16, requires_grad=True)
    targets = torch.randint(0, 50, (23,))
    full = F.cross_entropy(head(states), targets, reduction="sum")
    (g_full,) = torch.autograd.grad(full, states)
    chunked = chunked_cross_entropy(head, states, targets, chunk=4)
    (g_chunked,) = torch.autograd.grad(chunked, states)
    assert torch.allclose(full, chunked, atol=1e-4) and torch.allclose(g_full, g_chunked, atol=1e-5)


@torch.no_grad()
def test_packing_and_padding_change_nothing(policy):
    batch = examples(6)
    assert any(e.act for e in batch) and not all(e.act for e in batch)
    policy.pack_length = 0
    padded = policy(batch)
    logits = policy.slot_logits(batch)
    policy.pack_length = 10_000  # every example in one row
    packed = policy(batch)
    packed_logits = policy.slot_logits(batch)
    policy.pack_length = 0
    for key in ("text", "action"):
        assert torch.allclose(padded[key], packed[key], atol=1e-4), key
    assert torch.allclose(logits, packed_logits, atol=1e-4)
    first = next(i for i, e in enumerate(batch) if e.act)
    alone = policy.slot_logits([batch[first]])
    assert torch.allclose(alone, logits[: alone.shape[0]], atol=1e-4)  # next to other examples or alone: the same


@torch.no_grad()
def test_text_loss_is_on_reply_tokens_only(policy, tokenizer):
    """Against a plain causal pass of the LM over the stream built by hand: the text loss is the
    cross-entropy of the reply tokens alone, and the slots' readout matches."""
    for e in examples(4, seed=3):
        out = policy([e])
        layout = policy.layout(e)
        embed = policy.lm.get_input_embeddings()
        obs = e.obs.unsqueeze(0)
        parts = [embed(torch.tensor([layout.before])), policy.patches(obs, 0), embed(torch.tensor([layout.after])), embed(torch.tensor([layout.reply]))]
        if e.act:
            parts.append(policy.patches(obs, 1))
        x = torch.cat(parts, 1)
        h = policy.lm.model(inputs_embeds=x).last_hidden_state[0]
        logits = policy.lm.lm_head(h)
        start = layout.reply_start
        target = torch.full((x.shape[1],), -100)
        target[start - 1 : start - 1 + len(layout.reply)] = torch.tensor(layout.reply)
        expected = F.cross_entropy(logits, target, ignore_index=-100)
        assert torch.allclose(out["text"], expected, atol=1e-4)
        assert out["text_tokens"] == len(layout.reply) == len(tokenizer(e.reply_text, add_special_tokens=False).input_ids)
        if e.act:
            n = layout.n
            slots = policy.readout(h[-n:]).view(n, PATCH**3, N_ACTIONS)
            assert torch.allclose(policy.slot_logits([e]), slots, atol=1e-4)
            assert out["action"] > 0
        else:
            assert out["action"] == 0


def test_context_wrapping():
    assert chat("hi SCENE").startswith("<|im_start|>user\n") and chat("hi SCENE").endswith("<|im_start|>assistant\n")
    assert chat("<|im_start|>user\nhi SCENE<|im_end|>\n") == "<|im_start|>user\nhi SCENE<|im_end|>\n"


@torch.no_grad()
def test_think_then_act_matches_training_stream(policy, tokenizer):
    """Two rows of different context lengths, the LM head scripted: one row thinks and acts, the
    other answers. The acting row's action logits equal those of the training stream on the same
    reply, and the text row's answer is decoded."""
    data = examples(2, seed=5)
    acting = Example(data[0].obs, data[0].context_text, "<think>\nEvery robot takes move 3.\n</think>\n\n<tool_call>", True, torch.full((8, 8, 8), 3), {})
    talking = Example(data[1].obs, "<|im_start|>user\nsing a long song about the sea and the sky SCENE<|im_end|>\n<|im_start|>assistant\n", f"<think>\nnot a move\n</think>\n\n{REFUSAL}<|im_end|>", False, None, {})
    scripts = [policy.encode(acting.reply_text), policy.encode(talking.reply_text)]
    calls = [0]

    def scripted(module, inputs, output):
        k = calls[0]
        calls[0] += 1
        bias = torch.zeros_like(output)
        for row, script in enumerate(scripts):
            if k < len(script):
                bias[row, script[k]] = 1e4
        return output + bias

    hook = policy.lm.lm_head.register_forward_hook(scripted)
    try:
        step = policy.think_then_act(torch.stack([acting.obs, talking.obs]), [acting.context_text, talking.context_text], max_new_tokens=64)
    finally:
        hook.remove()
    assert step.reply_ids == scripts
    assert step.acts.tolist() == [True, False] and step.finished.tolist() == [True, True]
    assert step.thought == ["Every robot takes move 3.", "not a move"]
    assert step.answer == ["", REFUSAL]
    assert torch.allclose(step.logits[0], policy.slot_logits([acting]), atol=1e-4)


def test_unfinished_row_and_no_action(policy):
    e = examples(1, seed=7)[0]
    step = policy.think_then_act(e.obs.unsqueeze(0), [e.context_text], max_new_tokens=3)
    assert len(step.reply_ids[0]) <= 3
    assert step.logits is None or step.acts[0]


def test_overfit_both_losses_fall(tokenizer):
    # a random tiny LM has no useful output head: the (tied) embeddings train here too
    model = tiny(tokenizer, pack_length=512, freeze_embeddings=False).train()
    batch = examples(6, seed=11)
    optimizer = torch.optim.AdamW([{"params": model.lm_parameters(), "lr": 3e-3}, {"params": model.new_parameters(), "lr": 3e-3}])
    first = None
    for _ in range(20):
        out = model(batch)
        optimizer.zero_grad()
        out["loss"].backward()
        optimizer.step()
        first = first or {k: out[k].item() for k in ("text", "action")}
    last = model(batch)
    assert last["text"].item() < first["text"] - 2.0 and last["action"].item() < 0.7 * first["action"], (first, last)


def test_lora_option(tokenizer):
    model = tiny(tokenizer, lora_rank=4)
    lm_trainable = [n for n, p in model.lm.named_parameters() if p.requires_grad]
    assert lm_trainable and all(n.endswith((".a", ".b")) for n in lm_trainable)
    out = model(examples(3))
    out["loss"].backward()
    assert out["loss"].isfinite()


def test_full_weights_default_freezes_embeddings(policy):
    assert not policy.lm.get_input_embeddings().weight.requires_grad
    assert any(p.requires_grad for n, p in policy.lm.named_parameters() if "layers.0.mlp" in n)


def test_trainer_runs_saves_and_resumes(tokenizer, tmp_path, monkeypatch):
    """think_train end to end on the tiny LM (CPU): fake data, distillation on a local text file,
    a validation, a step checkpoint, then a resumed session that continues the step count."""
    import json
    import sys

    from beebot.vla.voxel import think_train
    from beebot.vla.voxel.think import load_policy

    def from_pretrained(name, **kwargs):
        return tiny(tokenizer).lm

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", from_pretrained)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "general.txt").write_text("\n\n".join(" ".join(f"word{i} number {j} of the text" for j in range(40)) for i in range(20)))
    common = ["--name", "t", "--data", "fake", "--data_kwargs", '{"side": 8, "instruction_words": 3, "thought_words": 6}', "--device", "cpu", "--batch", "3", "--accumulate", "2", "--val_examples", "4", "--log_every", "2", "--general", "general.txt", "--lm_length", "32", "--warmup_steps", "2", "--pack_length", "256", "--prefetch", "2"]
    monkeypatch.setattr(sys, "argv", ["think_train", *common, "--steps", "4"])
    think_train.run()
    monkeypatch.setattr(sys, "argv", ["think_train", *common, "--steps", "6", "--resume", "runs/t/resume.pt"])
    think_train.run()
    rows = [json.loads(line) for line in open(tmp_path / "runs/t/history.jsonl")]
    trained = [r for r in rows if "text" in r]
    assert [r["step"] for r in trained] == [1, 2, 4, 5, 6] and all(r["distill_kl"] >= 0 and r["tokens_per_s"] > 0 for r in trained)
    assert [r["step"] for r in rows if "val_loss" in r] == [4, 6]
    assert (tmp_path / "runs/t/policy_step4.pt").exists() and (tmp_path / "runs/t/policy_step6.pt").exists()
    policy, _ = load_policy(tmp_path / "runs/t/policy_step6.pt", "cpu")
    e = examples(1)[0]
    assert policy.think_then_act(e.obs.unsqueeze(0), [e.context_text], max_new_tokens=2).thought is not None
