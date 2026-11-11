"""The shared cognitive core: one frozen pretrained language model.

The core is replaceable through this contract. Anything the rest of the system
needs from a core is listed in `CoreContract`; a new core is acceptable when it
loads through `Core.load`, reports these capabilities and passes the injection
check (`scripts/core_bench.py`).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

MEM_MARK = "⁣MEM⁣"   # invisible separators around a marker that never occurs in normal text


@dataclass
class CoreContract:
    """What MPN Studio requires of a core. Recorded in every manifest."""
    name: str
    repo: str
    revision: str
    license: str
    architecture: str
    hidden_size: int
    params_total: int
    precision: str = "bf16"
    accepts_inputs_embeds: bool = True      # needed for node-memory injection
    gradient_through_frozen_core: bool = True  # needed to train memory and bridge
    chat_template: bool = True
    tool_calling: bool = True
    max_context: int = 32768
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


class Core:
    def __init__(self, model, tok, contract: CoreContract, device: str = "cuda"):
        self.model, self.tok, self.contract, self.device = model, tok, contract, device
        self.emb = model.get_input_embeddings()
        self.hidden_size = self.emb.embedding_dim
        with torch.no_grad():
            self.embed_norm = float(self.emb.weight[:20000].float().norm(dim=-1).mean())
        self._tmpl_cache: dict = {}

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, model_dir: str | Path, device: str = "cuda", dtype=torch.bfloat16) -> "Core":
        from transformers import AutoConfig, AutoTokenizer
        model_dir = str(model_dir)
        t0 = time.perf_counter()
        cfg = AutoConfig.from_pretrained(model_dir)
        tok = AutoTokenizer.from_pretrained(model_dir)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        arch = (cfg.architectures or ["?"])[0]
        kw = dict(dtype=dtype, device_map={"": device}, low_cpu_mem_usage=True)
        if "ConditionalGeneration" in arch:
            from transformers import AutoModelForImageTextToText as Auto
        else:
            from transformers import AutoModelForCausalLM as Auto
        model = Auto.from_pretrained(model_dir, **kw)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        meta = {}
        mf = Path(model_dir) / "CORE.json"
        if mf.exists():
            meta = json.loads(mf.read_text())
        contract = CoreContract(
            name=meta.get("name", Path(model_dir).name), repo=meta.get("repo", ""), revision=meta.get("revision", ""),
            license=meta.get("license", ""), architecture=arch, hidden_size=model.get_input_embeddings().embedding_dim,
            params_total=sum(p.numel() for p in model.parameters()),
            max_context=int(meta.get("max_context", 32768)), extra={"load_s": round(time.perf_counter() - t0, 2)})
        return cls(model, tok, contract, device)

    # ------------------------------------------------------------------ prompts
    def chat_text(self, messages: list[dict], tools: list | None = None) -> str:
        kw = dict(add_generation_prompt=True, tokenize=False)
        if tools:
            kw["tools"] = tools
        try:
            return self.tok.apply_chat_template(messages, enable_thinking=False, **kw)
        except TypeError:
            return self.tok.apply_chat_template(messages, **kw)

    def ids(self, text: str) -> torch.Tensor:
        return self.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

    def split_on_memory(self, messages: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        """Token ids before and after the memory marker in the rendered chat prompt."""
        text = self.chat_text(messages)
        if MEM_MARK not in text:
            return self.ids(text), torch.empty(0, dtype=torch.long)
        a, b = text.split(MEM_MARK, 1)
        return self.ids(a), self.ids(b)

    def embed_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(ids.to(self.device))

    # ------------------------------------------------------------------ batching helpers
    def build_batch(self, pieces: list[list[torch.Tensor]], labels: list[torch.Tensor | None] | None = None):
        """Left-pad a batch of sequences given as lists of [T_i, H] embedding pieces.

        `labels[i]`, when given, is aligned to the *end* of sequence i (the answer tokens)."""
        seqs = [torch.cat(p, dim=0) for p in pieces]
        T = max(s.shape[0] for s in seqs)
        H = seqs[0].shape[-1]
        x = torch.zeros(len(seqs), T, H, device=self.device, dtype=seqs[0].dtype)
        mask = torch.zeros(len(seqs), T, device=self.device, dtype=torch.long)
        lab = torch.full((len(seqs), T), -100, device=self.device, dtype=torch.long) if labels is not None else None
        for i, s in enumerate(seqs):
            x[i, T - s.shape[0]:] = s
            mask[i, T - s.shape[0]:] = 1
            if lab is not None and labels[i] is not None:
                lab[i, T - labels[i].shape[0]:] = labels[i].to(self.device)
        return x, mask, lab

    @torch.inference_mode()
    def generate_from_embeds(self, x: torch.Tensor, mask: torch.Tensor, max_new_tokens: int = 64, streamer=None,
                             stopping_criteria=None) -> list[str]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model.generate(inputs_embeds=x, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False,
                                      temperature=None, top_p=None, top_k=None, pad_token_id=self.tok.pad_token_id,
                                      streamer=streamer, stopping_criteria=stopping_criteria)
        return [t.strip() for t in self.tok.batch_decode(out, skip_special_tokens=True)]
