"""The plain causal stream layout of the voxel LM policy, on a tiny randomly initialised LM."""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from beebot.vla.voxel.stream import StreamVoxelPolicy, right_align  # noqa: E402
from beebot.vla.voxel.world import N_ACTIONS, N_CHANNELS  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def setup():
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        config = transformers.AutoConfig.from_pretrained(MODEL, local_files_only=True)
    except OSError:
        pytest.skip("the Qwen tokenizer is not cached")
    tokenizer.padding_side = "right"
    config.hidden_size, config.intermediate_size, config.num_hidden_layers = 32, 64, 2
    config.num_attention_heads, config.num_key_value_heads, config.head_dim = 4, 2, 8
    torch.manual_seed(0)
    policy = StreamVoxelPolicy(transformers.AutoModelForCausalLM.from_config(config), tokenizer, lora_rank=4).eval()
    with torch.no_grad():  # LoRA's B starts at zero: make the update live so the test covers it
        for name, p in policy.named_parameters():
            if name.endswith(".b"):
                p.normal_(std=0.02)
    return policy, tokenizer


def scene(seed=0, batch=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(batch, N_CHANNELS, 8, 8, 8, generator=g) > 0.7).float()


def tokens(tokenizer, texts):
    enc = tokenizer(texts, return_tensors="pt", padding=True)
    return enc.input_ids, enc.attention_mask


def test_right_align_keeps_order():
    ids = torch.tensor([[5, 6, 0, 0], [1, 2, 3, 4]])
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    out, m = right_align(ids, mask)
    assert out.tolist() == [[0, 0, 5, 6], [1, 2, 3, 4]] and m.tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]


@torch.no_grad()
def test_shapes_and_padding_invariance(setup):
    policy, tokenizer = setup
    obs = scene(batch=2)
    ids, mask = tokens(tokenizer, ["eat the red cheese", "build a tower five robots tall right now"])
    out = policy.logits(obs, ids, mask)
    assert out.shape == (2, 8, 64, N_ACTIONS)
    alone = policy.logits(obs[:1], *tokens(tokenizer, ["eat the red cheese"]))
    assert torch.allclose(out[:1], alone, rtol=1e-4, atol=1e-4)  # padding next to a longer row changes nothing


@torch.no_grad()
def test_slots_read_the_instruction_and_are_causal(setup):
    policy, tokenizer = setup
    obs = scene()
    a = policy.logits(obs, *tokens(tokenizer, ["eat the red cheese"]))
    b = policy.logits(obs, *tokens(tokenizer, ["eat the blue cheese"]))
    assert not torch.allclose(a, b, atol=1e-5)
    # the scene comes before every slot: changing the last patch changes the first slot
    other = obs.clone()
    other[..., 4:, 4:, 4:] = 1 - other[..., 4:, 4:, 4:]
    c = policy.logits(other, *tokens(tokenizer, ["eat the red cheese"]))
    assert not torch.allclose(a[:, 0], c[:, 0], atol=1e-5)


@torch.no_grad()
def test_null_ignores_the_instruction(setup):
    policy, tokenizer = setup
    obs = scene()
    null = torch.ones(1, dtype=torch.bool)
    a = policy.logits(obs, *tokens(tokenizer, ["eat the red cheese"]), null=null)
    b = policy.logits(obs, *tokens(tokenizer, ["eat the blue cheese"]), null=null)
    assert torch.allclose(a, b, atol=1e-4)


def test_losses_and_reply(setup):
    policy, tokenizer = setup
    obs = scene(batch=2)
    ids, mask = tokens(tokenizer, ["eat the red cheese", "go to the back"])
    codes = torch.randint(0, N_ACTIONS, (2, 8, 8, 8))
    loss = policy.loss(obs, codes, ids, mask, null=torch.tensor([False, True]))
    assert loss.isfinite() and loss.requires_grad
    answers = [tokenizer(t, add_special_tokens=False).input_ids + [policy.end_id] for t in ("red, blue", "none")]
    width = max(len(a) for a in answers)
    a_ids = torch.tensor([a + [policy.end_id] * (width - len(a)) for a in answers])
    a_mask = torch.tensor([[1] * len(a) + [0] * (width - len(a)) for a in answers])
    text = policy.text_loss(obs, ids, mask, a_ids, a_mask)
    assert text.isfinite() and text.requires_grad
    # the answer's padding carries no loss
    longer = torch.cat([a_ids, torch.full((2, 3), policy.end_id)], 1)
    padded = policy.text_loss(obs, ids, mask, longer, torch.cat([a_mask, torch.zeros(2, 3, dtype=torch.long)], 1))
    assert torch.allclose(text, padded, atol=1e-4)
    out = policy.reply(obs, ids, mask, max_new_tokens=4)
    assert out.shape[0] == 2 and out.shape[1] <= 4
