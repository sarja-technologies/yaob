"""Frozen text encoder used for routing queries and for initial node state.

This is a conventional pretrained dependency (default: Qwen3-Embedding-0.6B, Apache-2.0).
It is never trained here. Its outputs are inputs to the learned router and memory.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

QUERY_INSTRUCTION = "Given a question, retrieve the knowledge node that answers it"


class TextEncoder:
    def __init__(self, model_dir: str, device: str = "cuda", dtype=torch.bfloat16, max_len: int = 256):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
        self.model = AutoModel.from_pretrained(model_dir, dtype=dtype, device_map={"": device}).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device, self.max_len = device, max_len
        self.dim = self.model.config.hidden_size
        self.params = sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def encode(self, texts: list[str], query: bool = False, batch_size: int = 64) -> torch.Tensor:
        """-> [len(texts), dim] float32 unit vectors on CPU. Empty strings give zero vectors."""
        out = torch.zeros(len(texts), self.dim)
        idx = [i for i, t in enumerate(texts) if t and t.strip()]
        for b in range(0, len(idx), batch_size):
            chunk = idx[b:b + batch_size]
            batch = [(f"Instruct: {QUERY_INSTRUCTION}\nQuery: {texts[i]}" if query else texts[i]) for i in chunk]
            enc = self.tok(batch, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state[:, -1]        # left padding: the last position is the final token
            out[chunk] = F.normalize(h.float(), dim=-1).cpu()
        return out
