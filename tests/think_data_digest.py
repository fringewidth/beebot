"""Digest of today's default ThinkData examples (token ids, images, observations, codes, texts) on the
synthetic store of test_vla_voxel_think_data.py, for the byte-identical check in test_think_ground.py.

    .venv/bin/python tests/think_data_digest.py --write     # only when a change to the defaults is intended

The fixture tests/fixtures/think_data_default_digest.json was written from think_data.py as it stood
before the grounding kinds (think_ground.py) were added; ThinkData without `ground` must give the same."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

FIXTURE = HERE / "fixtures" / "think_data_default_digest.json"
TOKENIZER = "Qwen/Qwen3.5-0.8B"


def tokenizer_or_none():
    try:
        import transformers

        return transformers.AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    except Exception:
        return None


def example_digest(example, tokenizer=None):
    from beebot.vla.voxel import think_data as T

    h = hashlib.sha256()
    for text in (example.context_text, example.reply_text, example.rung, example.kind, example.family, example.instruction, str(example.turns), f"{example.camera:.6f}", str(example.act), str(example.e), str(example.t)):
        h.update(text.encode())
        h.update(b"\x00")
    for image in example.images:
        h.update(f"{image.mode}{image.size}".encode())
        h.update(image.tobytes())
    h.update(np.ascontiguousarray(example.obs.numpy()).tobytes())
    if example.codes is not None:
        h.update(np.ascontiguousarray(example.codes.numpy()).tobytes())
    if tokenizer is not None:
        ids, labels = T.tokenize(example, tokenizer)
        h.update(ids.numpy().tobytes())
        h.update(labels.numpy().tobytes())
    return h.hexdigest()


def build(tmp, **kwargs):
    """ThinkData on the synthetic store with the haiku labels of test_vla_voxel_think_data.py (stand-in views)."""
    import test_vla_voxel_think_data as TD

    from beebot.vla.voxel import think_data as T

    store = TD.synthetic_store(Path(tmp) / "store")
    n0, n1, n2 = (TD.index_of(store, e) for e in (0, 1, 2))
    rows = [
        TD.pair(TD.SEED, n0, "logic", "eat:ordered", "close", "eat the red one and then the blue one", "you asked: eat the red one and then the blue one. i see red and blue cheeses. red first. first: eat the red cheese.", ["now the blue one"]),
        TD.pair(TD.SEED, n1, "switch", "goto", "close", "head to the front", "you asked: head to the front. i see a flat floor. first: go to the front of the floor.", part=0, parts=2, t=0),
        TD.pair(TD.SEED, n1, "switch", "stay", "indirect", "could you all freeze", "you asked: could you all freeze. nobody moves. first: stay where you are.", part=1, parts=2, t=6),
        TD.pair(TD.SEED, n2, "formation", "mask", "close", "make a wall", "you asked: make a wall. i see robots. first: form a wall."),
        TD.pair(TD.SEED, n2, "formation", "refuse", "refusal", "make the blue cheese fly", "you asked: make the blue cheese fly. gravity is on, so i will not act.", reply="the blue cheese cannot fly, so i will do nothing."),
    ]
    labels = TD.label_dir(Path(tmp) / "haiku" / "run-test", rows)
    return T.ThinkData([store], [labels], held_share=0.2, refusal_share=0.2, **kwargs)


def digests(data, tokenizer=None, n=48):
    rng = np.random.default_rng(5)
    out = [example_digest(data.sample(rng), tokenizer) for _ in range(n)]
    out += [example_digest(ex, tokenizer) for ex in data.validation(12, seed=1)]
    return out


def main():
    import tempfile

    torch.set_num_threads(1)
    tokenizer = tokenizer_or_none()
    assert tokenizer is not None, f"{TOKENIZER} tokenizer not cached"
    with tempfile.TemporaryDirectory() as tmp:
        data = build(tmp)
        rows = digests(data, tokenizer)
    if "--write" in sys.argv:
        FIXTURE.parent.mkdir(exist_ok=True)
        FIXTURE.write_text(json.dumps({"tokenizer": TOKENIZER, "digests": rows}, indent=0) + "\n")
        print(f"wrote {len(rows)} digests to {FIXTURE}")
    else:
        print(json.dumps(rows[:3]))


if __name__ == "__main__":
    main()
