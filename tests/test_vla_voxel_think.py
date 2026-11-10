"""The think-then-act voxel policy (voxel/think.py) on tiny randomly initialised LMs: Qwen3 (text,
the voxel scene as tokens) and Qwen3.5 (vision-language and hybrid linear attention, the scene as
images)."""

import json
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")

from beebot.vla.voxel.think import IMAGES, ThinkVoxelPolicy, chat, chat_context, chunked_cross_entropy  # noqa: E402
from beebot.vla.voxel.think_fake import REFUSAL, Example, FakeThinkData  # noqa: E402

QWEN3 = "Qwen/Qwen3-0.6B"
QWEN35 = "Qwen/Qwen3.5-4B"


def tiny_lm(name):
    """A tiny randomly initialised LM with the cached model's architecture (fp32)."""
    config = transformers.AutoConfig.from_pretrained(name, local_files_only=True)
    torch.manual_seed(0)
    if name == QWEN3:
        config.hidden_size, config.intermediate_size, config.num_hidden_layers = 32, 64, 2
        config.num_attention_heads, config.num_key_value_heads, config.head_dim = 4, 2, 8
        return transformers.AutoModelForCausalLM.from_config(config).float()
    text, vision = config.text_config, config.vision_config
    text.hidden_size, text.intermediate_size, text.num_hidden_layers = 32, 64, 4
    text.layer_types = ["linear_attention"] * 3 + ["full_attention"]
    text.num_attention_heads, text.num_key_value_heads, text.head_dim = 2, 1, 32
    text.linear_num_key_heads, text.linear_num_value_heads, text.linear_key_head_dim, text.linear_value_head_dim = 2, 4, 8, 8
    text.rope_parameters = {**text.rope_parameters, "partial_rotary_factor": 0.5, "mrope_section": [3, 3, 2]}
    vision.depth, vision.hidden_size, vision.intermediate_size, vision.num_heads, vision.out_hidden_size = 2, 32, 64, 2, 32
    return transformers.AutoModelForImageTextToText.from_config(config).float()


def parts(name):
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
    except OSError:
        pytest.skip(f"{name} is not cached")
    image_processor = None
    if name == QWEN35:
        from transformers.models.qwen2_vl import Qwen2VLImageProcessorPil

        image_processor = Qwen2VLImageProcessorPil.from_pretrained(name, local_files_only=True)
    return tokenizer, image_processor


def tiny(name, **kwargs):
    tokenizer, image_processor = parts(name)
    return ThinkVoxelPolicy(tiny_lm(name), tokenizer, image_processor, **kwargs)


@pytest.fixture(scope="module", params=[QWEN3, QWEN35])
def policy(request):
    return tiny(request.param).eval()


def data(policy, **kwargs):
    vision = policy.visual is not None
    return FakeThinkData(side=8, instruction_words=3, thought_words=6, robot_share=0.2, images=2 if vision else 0, image_size=64, scene=not vision, **kwargs)


def examples(policy, n, seed=0, **kwargs):
    rng = np.random.default_rng(seed)
    source = data(policy, **kwargs)
    return [source.sample(rng) for _ in range(n)]


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


def test_context_wrapping():
    assert chat("hi SCENE").startswith("<|im_start|>user\n") and chat("hi SCENE").endswith("<|im_start|>assistant\n")
    assert chat("<|im_start|>user\nhi SCENE<|im_end|>\n") == "<|im_start|>user\nhi SCENE<|im_end|>\n"
    assert chat_context("go", images=True, scene=False) == f"<|im_start|>user\n{IMAGES}go<|im_end|>\n<|im_start|>assistant\n"


@torch.no_grad()
def test_padding_changes_nothing(policy):
    """An example alone or right-padded next to longer ones: the same reply logits and slots."""
    batch = examples(policy, 5)
    assert any(e.act for e in batch) and not all(e.act for e in batch)
    together = policy.reply_logits(batch)
    slots = policy.slot_logits(batch)
    k = 0
    for e, logits in zip(batch, together):
        assert torch.allclose(policy.reply_logits([e])[0], logits, atol=1e-4)
        if e.act:
            alone = policy.slot_logits([e])
            assert torch.allclose(alone, slots[k : k + len(alone)], atol=1e-4)
            k += len(alone)


@torch.no_grad()
def test_text_loss_is_on_reply_tokens_only(policy):
    """Against the LM's own forward pass over the row's tokens (its own image handling and
    positions for Qwen3.5; the scene embedded by hand for Qwen3): the text loss is the
    cross-entropy of the reply tokens alone."""
    for e in examples(policy, 4, seed=3):
        out = policy([e])
        p = policy.prepare_example(e)
        end = p.reply_start + p.reply_len
        ids = torch.tensor([p.ids[:end]])
        if policy.visual is not None:
            h = policy.lm.model(input_ids=ids, pixel_values=p.pixels, image_grid_thw=p.grid, mm_token_type_ids=(ids == policy.image_id).int()).last_hidden_state[0]
            policy.lm.model.rope_deltas = None
        else:
            x = policy.embed(ids)
            x[0, p.scene_at : p.scene_at + p.n] = policy.patches(e.obs.unsqueeze(0), 0)[0]
            h = policy.lm.model(inputs_embeds=x).last_hidden_state[0]
        target = torch.full((end,), -100)
        target[p.reply_start - 1 : end - 1] = torch.tensor(p.ids[p.reply_start : end])
        expected = F.cross_entropy(policy.lm.lm_head(h), target, ignore_index=-100)
        assert torch.allclose(out["text"], expected, atol=1e-4), (out["text"], expected)
        assert out["text_tokens"] == p.reply_len == len(policy.encode(e.reply_text))
        assert bool(out["action"] > 0) == e.act


@torch.no_grad()
def test_think_then_act_scripted(policy):
    """Two rows (different context lengths and, for Qwen3.5, two images against none), the LM head
    scripted: one row thinks and acts, the other answers. The acting row's action logits equal the
    training stream's on the same reply, and the text row's answer is decoded."""
    base = examples(policy, 2, seed=5)
    vision = policy.visual is not None
    context = chat_context("sing a long song about the sea and the sky", images=False, scene=not vision)
    acting = Example(base[0].obs, base[0].context_text, "<think>\nEvery robot takes move 3.\n</think>\n\n<tool_call>", True, torch.full((8, 8, 8), 3), {}, base[0].images)
    talking = Example(base[1].obs, context, f"<think>\nnot a move\n</think>\n\n{REFUSAL}<|im_end|>", False, None, {}, [])
    scripts = [policy.encode(acting.reply_text), policy.encode(talking.reply_text)]
    calls = [0]

    def scripted(module, inputs, output):
        k = calls[0]
        calls[0] += 1
        bias = torch.zeros_like(output)
        for row, script in enumerate(scripts):
            bias[row, script[min(k, len(script) - 1)]] = 1e4
        return output + bias

    hook = policy.lm.lm_head.register_forward_hook(scripted)
    try:
        step = policy.think_then_act(torch.stack([acting.obs, talking.obs]), [acting.context_text, talking.context_text], [acting.images, talking.images], max_new_tokens=64)
    finally:
        hook.remove()
    assert step.reply_ids == scripts
    assert step.acts.tolist() == [True, False] and step.finished.tolist() == [True, True]
    assert step.thought == ["Every robot takes move 3.", "not a move"]
    assert step.answer == ["", REFUSAL]
    assert torch.allclose(step.logits[0], policy.slot_logits([acting]), atol=1e-4)


@torch.no_grad()
def test_cached_generation_matches_the_stream(policy):
    """Greedy tokens from the KV-cached, left-padded generation are the argmax of the training
    stream's logits on the same reply (positions, padding and cache agree)."""
    batch = examples(policy, 3, seed=9)
    if policy.visual is not None:
        batch[1].images = batch[1].images[:1]  # rows with 2, 1 and 0 images
        batch[2].images = []
        batch[2].context_text = batch[2].context_text.replace(IMAGES, "")
    step = policy.think_then_act(torch.stack([e.obs for e in batch]), [e.context_text for e in batch], [e.images for e in batch], max_new_tokens=6)
    for e, ids in zip(batch, step.reply_ids):
        row = policy.with_reply(policy.prepare(e.context_text, e.obs, e.images), ids, False)
        assert policy.reply_logits([row])[0].argmax(-1).tolist() == ids


def test_unfinished_row_and_no_action(policy):
    e = examples(policy, 1, seed=7)[0]
    step = policy.think_then_act(e.obs.unsqueeze(0), [e.context_text], [e.images], max_new_tokens=3)
    assert len(step.reply_ids[0]) <= 3
    assert step.logits is None or step.acts[0]


@pytest.mark.parametrize("name", [QWEN3, QWEN35])
def test_overfit_both_losses_fall(name):
    # a random tiny LM has no useful output head: the (tied) embeddings train here too
    model = tiny(name, freeze_embeddings=False).train()
    batch = examples(model, 6, seed=11)
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


def test_trainable_parts():
    model = tiny(QWEN35)
    visual = model.visual
    assert not model.lm.get_input_embeddings().weight.requires_grad
    assert all(p.requires_grad for p in visual.merger.parameters())
    assert not any(p.requires_grad for n, p in visual.named_parameters() if not n.startswith("merger"))
    assert all(p.requires_grad for p in model.text_model.layers.parameters())
    assert not any(n.startswith("visual") or n.startswith("text_model") for n, _ in model.named_parameters())  # no second registration
    model = tiny(QWEN35, train_vision=True)
    assert all(p.requires_grad for p in model.visual.parameters())
    model = tiny(QWEN35, train_projector=False)
    assert not any(p.requires_grad for p in model.visual.parameters())
    lora = tiny(QWEN3, lora_rank=4)
    lm_trainable = [n for n, p in lora.lm.named_parameters() if p.requires_grad]
    assert lm_trainable and all(n.endswith((".a", ".b")) for n in lm_trainable)
    out = lora(examples(lora, 3))
    out["loss"].backward()
    assert out["loss"].isfinite()
    lora = tiny(QWEN35, lora_rank=4)  # LoRA reaches the linear-attention projections too; the projector still trains
    names = [n for n, p in lora.lm.named_parameters() if p.requires_grad]
    assert any("linear_attn.in_proj_qkv.a" in n for n in names) and any("self_attn.q_proj.a" in n for n in names)
    assert all(n.endswith((".a", ".b")) or ".merger." in n for n in names)


@pytest.mark.parametrize("name", [QWEN3, QWEN35])
def test_trainer_runs_saves_and_resumes(name, tmp_path, monkeypatch):
    """think_train end to end on the tiny LM (CPU): fake data, distillation on a local text file,
    a validation, a step checkpoint, then a resumed session that continues the step count."""
    from beebot.vla.voxel import think, think_train

    tokenizer, image_processor = parts(name)

    def load_lm(model, dtype=torch.float32):
        return tiny_lm(name).to(dtype), tokenizer, image_processor

    monkeypatch.setattr(think, "load_lm", load_lm)
    monkeypatch.setattr(think_train, "load_lm", load_lm)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "general.txt").write_text("\n\n".join(" ".join(f"word{i} number {j} of the text" for j in range(40)) for i in range(20)))
    vision = name == QWEN35
    fake = {"side": 8, "instruction_words": 3, "thought_words": 6, "images": 2 if vision else 0, "image_size": 64, "scene": not vision}
    common = ["--name", "t", "--model", name, "--data", "fake", "--data_kwargs", json.dumps(fake), "--device", "cpu", "--batch", "3", "--accumulate", "2", "--val_examples", "4", "--log_every", "2", "--general", "general.txt", "--lm_length", "32", "--warmup_steps", "2", "--prefetch", "2"]
    monkeypatch.setattr(sys, "argv", ["think_train", *common, "--steps", "4"])
    think_train.run()
    monkeypatch.setattr(sys, "argv", ["think_train", *common, "--steps", "6", "--resume", "runs/t/resume.pt"])
    think_train.run()
    rows = [json.loads(line) for line in open(tmp_path / "runs/t/history.jsonl")]
    trained = [r for r in rows if "text" in r]
    assert [r["step"] for r in trained] == [1, 2, 4, 5, 6] and all(r["distill_kl"] >= 0 and r["tokens_per_s"] > 0 for r in trained)
    assert [r["step"] for r in rows if "val_loss" in r] == [4, 6]
    assert (tmp_path / "runs/t/policy_step4.pt").exists() and (tmp_path / "runs/t/policy_step6.pt").exists()
    policy, _ = think.load_policy(tmp_path / "runs/t/policy_step6.pt", "cpu")
    e = FakeThinkData(**fake).sample(np.random.default_rng(0))
    assert policy.think_then_act(e.obs.unsqueeze(0), [e.context_text], [e.images], max_new_tokens=2).thought is not None
