"""The frozen text encoder: a pretrained embedding model, run once over every caption and prompt.

Captions and test prompts get the same instruction prefix, so they land in the same part of the
embedding space.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
INSTRUCTION = "Given a command for a robot swarm on a grid, represent what the swarm should do"


class TextEncoder:
    def __init__(self, model_name=DEFAULT_MODEL, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    def _inputs(self, texts):
        prompts = [f"Instruct: {INSTRUCTION}\nQuery:{t}" for t in texts]
        return self.tokenizer(prompts, padding=True, truncation=True, max_length=96, return_tensors="pt").to(self.device)

    @torch.no_grad()
    def embed(self, texts, batch_size=64):
        """(N, D) float32, L2-normalized, last-token pooled (left padding puts the last token at -1)."""
        out = []
        for i in range(0, len(texts), batch_size):
            inputs = self._inputs(texts[i : i + batch_size])
            hidden = self.model(**inputs).last_hidden_state
            out.append(F.normalize(hidden[:, -1].float(), dim=-1).cpu())
        return torch.cat(out)

    @torch.no_grad()
    def token_states(self, texts, batch_size=64):
        """Per-token final hidden states for the caption part, as a list of (T_i, D) float32 tensors."""
        out = []
        for i in range(0, len(texts), batch_size):
            inputs = self._inputs(texts[i : i + batch_size])
            hidden = self.model(**inputs).last_hidden_state.float().cpu()
            mask = inputs["attention_mask"].bool().cpu()
            for row, keep in zip(hidden, mask):
                out.append(row[keep])
        return out
