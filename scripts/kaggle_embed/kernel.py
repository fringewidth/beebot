"""Kaggle kernel: embed caption texts with Qwen3-Embedding-0.6B exactly as beebot.vla.text.TextEncoder
does (same instruction prefix, left padding, max length 96, last-token pooling, L2 normalised),
on every GPU the kernel has. Input: texts.json from the attached dataset. Output: embeddings.npy
(N, 1024) float16 in the same order, plus texts.json copied next to it.
"""

import glob
import json
import shutil
import subprocess
import sys
import threading

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.51"], check=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

MODEL = "Qwen/Qwen3-Embedding-0.6B"
INSTRUCTION = "Given a command for a robot swarm on a grid, represent what the swarm should do"
BATCH = 256

source = glob.glob("/kaggle/input/**/texts.json", recursive=True)[0]
texts = json.load(open(source))
print(f"{len(texts)} texts from {source}", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
models = [AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to(d).eval() for d in devices]
order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
batches = [order[i : i + BATCH] for i in range(0, len(order), BATCH)]
out = np.zeros((len(texts), models[0].config.hidden_size), dtype=np.float16)


def work(slot):
    device, model = devices[slot], models[slot]
    for b, index in enumerate(batches[slot :: len(devices)]):
        prompts = [f"Instruct: {INSTRUCTION}\nQuery:{texts[i]}" for i in index]
        inputs = tokenizer(prompts, padding=True, truncation=True, max_length=96, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
        out[index] = F.normalize(hidden[:, -1].float(), dim=-1).cpu().numpy().astype(np.float16)
        if b % 200 == 0:
            print(f"{device}: batch {b}", flush=True)


threads = [threading.Thread(target=work, args=(s,)) for s in range(len(devices))]
for t in threads:
    t.start()
for t in threads:
    t.join()
assert np.isfinite(out).all(), "non-finite embeddings (fp16 overflow?)"
np.save("/kaggle/working/embeddings.npy", out)
shutil.copy(source, "/kaggle/working/texts.json")
print("done", out.shape, flush=True)
