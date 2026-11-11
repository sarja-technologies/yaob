"""Training jobs for the learned modular state: the router and the MPN specialists.

Two kinds of learned state, trained separately and stored separately, with the core frozen throughout:

  router       per-node routing keys and gates, MPN prototypes, query/node projections.
               Objective: InfoNCE(question, node key) + multi-label MPN loss + gate loss.
               Needs only the frozen text encoder, so it trains in minutes.
  specialists  one LoRA adapter per MPN on the frozen core, trained only on that MPN's nodes.
               Objective: language-model loss on the node's sourced answers, mostly closed book (no node text
               in the prompt), partly with node text and a citation, so the adapter also learns to use context.
               The gradient runs through the core into the adapter's low-rank matrices; the core's own weights never change.

A job writes a new checkpoint directory, starting from the checkpoint it was based on, so that what it did
not retrain is carried over unchanged. Activation is a separate, explicit step.

Stages of a job: pin revision -> allocate rows -> embed -> train (checkpointing, resumable) -> held-out check
-> package -> register.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from ..model.bank import dir_bytes, package_router, save_tensors, sha256_file
from ..model.core import Core
from ..model.encoder import TextEncoder
from ..model.prompts import QUESTION_PREFIX, SYS_YAOB, render_nodes
from ..model.router import Router, RouterSpec, router_loss
from ..store.db import Store, now
from . import data as D
from .gpulock import GpuLock
from .jobs import JobContext, data_root, job_kind

BANK_ID = "main"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def code_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    f = root / "COMMIT"
    if f.exists():
        return f.read_text().strip()
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def models_root() -> Path:
    return Path(os.environ.get("MPN_MODELS", Path.home() / "mpn-studio" / "models"))


def node_text(title: str, statements: list[str]) -> str:
    return title + ". " + " ".join(statements[:4])


def allocate_rows(store: Store, node_ids: list[int], spec_json: dict, core_name: str) -> dict[int, int]:
    """Stable node -> row mapping. Rows are only ever appended; a deleted node keeps its row."""
    c = store.conn()
    with store._wlock:
        c.execute("BEGIN IMMEDIATE")
        try:
            c.execute("INSERT OR IGNORE INTO banks(id, kind, spec, core, created) VALUES(?,?,?,?,?)", (BANK_ID, "router+adapters", json.dumps(spec_json), core_name, now()))
            have = {r[0]: r[1] for r in c.execute("SELECT node_id, row FROM bank_rows WHERE bank_id=?", (BANK_ID,))}
            nxt = (max(have.values()) + 1) if have else 0
            for nid in node_ids:
                if nid not in have:
                    c.execute("INSERT INTO bank_rows(bank_id, node_id, row, state) VALUES(?,?,?,'allocated')", (BANK_ID, nid, nxt))
                    have[nid] = nxt
                    nxt += 1
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise
    return have


def gather(store: Store, rev: int) -> dict:
    nodes = list(D.iter_nodes(store, rev))
    w, a = store.at(rev)
    primary, shared = {}, {}
    for r in store.q(f"SELECT node_id, mpn_id, role FROM member_v WHERE {w}", a):
        if r["role"] == "primary":
            primary[r["node_id"]] = r["mpn_id"]
        else:
            shared.setdefault(r["node_id"], []).append(r["mpn_id"])
    mp = {int(m["id"]): m for m in store.mpns(rev)}
    return {"nodes": nodes, "primary": primary, "shared": shared, "mpn_ids": sorted(mp), "mpns": mp}


def embed_nodes(enc: TextEncoder, nodes: list[dict], cache_path: Path, log) -> dict[int, torch.Tensor]:
    """node id -> frozen node embedding (float16), cached on disk by content hash."""
    cache: dict = {}
    if cache_path.exists():
        try:
            cache = torch.load(cache_path)
        except Exception:
            cache = {}
    todo = [n for n in nodes if n["content_hash"] not in cache]
    if todo:
        log(f"embedding {len(todo)} nodes ({len(nodes) - len(todo)} cached)")
        for b in range(0, len(todo), 256):
            chunk = todo[b:b + 256]
            ne = enc.encode([node_text(n["title"], [s.get("text", "") for s in n["cells"].get("knowledge", {}).get("statements", [])]) for n in chunk])
            for n, v in zip(chunk, ne):
                cache[n["content_hash"]] = v.half()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        torch.save(cache, tmp)
        tmp.replace(cache_path)
    return {n["id"]: cache[n["content_hash"]] for n in nodes}


def norm_text(s: str) -> str:
    return "".join(ch for ch in s.lower().replace(",", "") if ch.isalnum() or ch in " .").strip()


def fact_correct(reply: str, values: list[str]) -> bool:
    from ..evals.heldout import score_values
    return score_values(reply, values)


def active_checkpoint(store: Store) -> tuple[str | None, Path | None]:
    ck = (store.get_setting("active_checkpoint") or {}).get("id")
    if not ck:
        return None, None
    r = store.q1("SELECT path FROM checkpoints WHERE id=?", (ck,))
    return (ck, Path(r["path"])) if r and Path(r["path"]).exists() else (None, None)


def link_tree(src: Path, dst: Path) -> None:
    """Copy a directory by hard links (same filesystem), falling back to real copies."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.rglob("*"):
        if f.is_file():
            t = dst / f.relative_to(src)
            t.parent.mkdir(parents=True, exist_ok=True)
            if t.exists():
                continue
            try:
                os.link(f, t)
            except OSError:
                shutil.copy2(f, t)


# ============================================================================ router
class RouterTrainer:
    def __init__(self, spec: RouterSpec, g: dict, rows: dict[int, int], emb: dict, device: str, seed: int):
        self.spec, self.rows, self.dev = spec, rows, device
        torch.manual_seed(seed)
        self.router = Router(spec).to(device)
        self.mpn_index = {m: i for i, m in enumerate(g["mpn_ids"])}
        self.node_ids = [n["id"] for n in g["nodes"]]
        self.node_emb = torch.zeros(spec.n_rows, spec.d_enc, dtype=torch.float16, device=device)
        self.row_mpn = torch.zeros(spec.n_rows, dtype=torch.long, device=device)
        self.mpn_targets = torch.zeros(spec.n_rows, spec.n_mpn, device=device)
        by_mpn: dict[int, list[int]] = {}
        for nid in self.node_ids:
            r = rows[nid]
            self.node_emb[r] = emb[nid].to(device)
            m = g["primary"].get(nid)
            if m is None:
                continue
            self.row_mpn[r] = self.mpn_index[m]
            self.mpn_targets[r, self.mpn_index[m]] = 1
            for s in g["shared"].get(nid, []):
                self.mpn_targets[r, self.mpn_index[s]] = 1
            by_mpn.setdefault(m, []).append(r)
        self.same_mpn = {r: torch.tensor(v, device=device) for v in by_mpn.values() for r in v}
        self.all_rows = torch.tensor([rows[n] for n in self.node_ids], device=device)

    def pairs(self, g: dict, examples: dict) -> tuple[list[str], list[int]]:
        """Training queries: the node's training questions, its title and aliases, and its statements."""
        qs, rs = [], []
        for e in examples["train"]:
            qs.append(e["q"])
            rs.append(self.rows[e["node"]])
        for n in g["nodes"]:
            r = self.rows[n["id"]]
            for t in [n["title"]] + list(n["cells"].get("thinking", {}).get("cues", []) or [])[:4]:
                if t:
                    qs.append(t)
                    rs.append(r)
            for s in n["cells"].get("knowledge", {}).get("statements", [])[:4]:
                if s.get("text"):
                    qs.append(s["text"])
                    rs.append(r)
        return qs, rs

    def train(self, q_emb: torch.Tensor, q_rows: torch.Tensor, steps: int, batch: int, lr: float, seed: int, ctx: JobContext, start: int = 0,
              only_rows: list[int] | None = None, ckpt=None):
        r = self.router
        per_node = [r.key_delta, r.gate_bias]
        shared = [p for p in r.parameters() if all(p is not x for x in per_node)]
        mask = None
        if only_rows is not None:
            for p in shared:
                p.requires_grad_(False)
            groups = [{"params": per_node, "lr": lr * 3, "weight_decay": 0.0}]
            mask = torch.zeros(self.spec.n_rows, device=self.dev)
            mask[only_rows] = 1
            keep = mask[q_rows] > 0
            q_emb, q_rows = q_emb[keep], q_rows[keep]
        else:
            groups = [{"params": shared, "lr": lr, "weight_decay": 0.01}, {"params": per_node, "lr": lr * 3, "weight_decay": 0.0}]
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.99))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * max(0.05, 0.5 * (1 + math.cos(math.pi * min(s, steps) / max(1, steps)))))
        if ckpt is not None:
            opt.load_state_dict(ckpt["opt"])
            sched.load_state_dict(ckpt["sched"])
        n = q_emb.shape[0]
        t0 = time.perf_counter()
        parts = {}
        r.train()
        for step in range(start, steps):
            gen = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + step)       # batches depend only on (seed, step): exact resume
            idx = torch.randint(0, n, (min(batch, n),), generator=gen).to(self.dev)
            pos = q_rows[idx]
            neg = self._negatives(pos, gen)
            loss, parts = router_loss(r, q_emb[idx].float(), pos, self.node_emb[pos].float(), neg, self.node_emb[neg].float(), self.mpn_targets[pos])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if mask is not None:
                r.key_delta.grad *= mask[:, None]
                r.gate_bias.grad *= mask
            opt.step()
            sched.step()
            done = step + 1
            if done % 100 == 0 or done == steps:
                dt = time.perf_counter() - t0
                info = {"stage": "router", "step": done, "total": steps, "loss": {k: round(v, 4) for k, v in parts.items()},
                        "steps_per_s": round((done - start) / max(dt, 1e-6), 1), "eta_s": round((steps - done) / max((done - start) / max(dt, 1e-6), 1e-6))}
                ctx.log(info)
                ctx.progress(**info)
            if ctx.should_stop():
                return {"opt": opt.state_dict(), "sched": sched.state_dict()}, done, parts, True
        r.eval()
        return {"opt": opt.state_dict(), "sched": sched.state_dict()}, steps, parts, False

    def _negatives(self, pos: torch.Tensor, gen) -> torch.Tensor:
        """Four hard negatives per positive: other nodes of the same MPN."""
        out = torch.empty(len(pos), 4, dtype=torch.long, device=self.dev)
        m = self.row_mpn[pos]
        for mi in m.unique().tolist():
            sel = (m == mi).nonzero(as_tuple=True)[0]
            pool = self.same_mpn.get(int(pos[sel[0]]), self.all_rows)
            out[sel] = pool[torch.randint(0, len(pool), (len(sel), 4), generator=gen).to(self.dev)]
        return out

    @torch.inference_mode()
    def evaluate(self, enc: TextEncoder, held: list[dict], n: int, seed: int) -> dict:
        """Held-out phrasings: learned router against the frozen-encoder baseline it is built on."""
        rng = random.Random(seed)
        held = [e for e in held if e["node"] in self.rows]
        rng.shuffle(held)
        held = held[:n]
        if not held:
            return {"n": 0}
        qe = enc.encode([e["q"] for e in held], query=True).to(self.dev)
        gold = torch.tensor([self.rows[e["node"]] for e in held], device=self.dev)
        rows = self.all_rows
        out = {"n": len(held)}
        for name, sim in (("learned", self.router.query_keys(qe) @ self.router.node_keys(rows, self.node_emb[rows].float()).T),
                          ("learned_without_node_keys", self.router.query_keys(qe) @ self.router.node_keys(rows, self.node_emb[rows].float(), use_delta=False).T),
                          ("frozen_encoder_baseline", qe @ self.node_emb[rows].float().T)):
            top = rows[sim.topk(min(10, len(rows)), dim=-1).indices]
            out[name] = {f"recall_at_{k}": round(float((top[:, :k] == gold[:, None]).any(-1).float().mean()), 4) for k in (1, 5, 10)}
        mp = torch.sigmoid(self.router.mpn_logits(self.router.query_keys(qe)))
        act = mp >= 0.5
        act[torch.arange(len(held)), mp.argmax(-1)] = True
        tgt = self.mpn_targets[gold] > 0
        out["mpn"] = {"primary_mpn_selected": round(float(act[torch.arange(len(held)), self.row_mpn[gold]].float().mean()), 4),
                      "mpns_selected_mean": round(float(act.sum(-1).float().mean()), 2),
                      "selected_that_are_memberships": round(float((act & tgt).sum().float() / act.sum().float()), 4)}
        return out


# ============================================================================ specialists (LoRA adapters)
def adapter_examples(nodes: list[dict]) -> dict[int, dict[str, list[dict]]]:
    out: dict[int, dict[str, list[dict]]] = {}
    for n in nodes:
        for e in D.node_examples(n)["train"]:
            out.setdefault(n["id"], {}).setdefault(e["kind"], []).append(e)
    return out


MIN_AVAILABLE_GIB = float(os.environ.get("MPN_MIN_AVAILABLE_GIB", "8"))


def memory_guard(ctx: JobContext, log) -> float | None:
    """Before each step: if the machine's available memory is below MIN_AVAILABLE_GIB, wait (checking for cancel)
    until it recovers. Returns seconds waited, or None if the job was cancelled while waiting. The Spark is shared and
    its GPU memory is the same memory as everyone else's; a job must give way rather than push the machine into the
    kernel's OOM killer."""
    t0 = time.perf_counter()
    noted = False
    while psutil.virtual_memory().available / 2**30 < MIN_AVAILABLE_GIB:
        if not noted:
            avail = round(psutil.virtual_memory().available / 2**30, 1)
            log(f"available memory {avail} GiB is below {MIN_AVAILABLE_GIB} GiB; waiting")
            ctx.progress(stage="waiting for memory", system_available_gib=avail, force=True)
            torch.cuda.empty_cache()
            noted = True
        if ctx.should_stop():
            return None
        time.sleep(5)
    return time.perf_counter() - t0


class AdapterTrainer:
    """Trains one MPN's adapter. The PEFT-wrapped core is shared between MPNs within a job."""

    def __init__(self, core: Core, rank: int, log):
        from peft import LoraConfig
        self.core, self.rank, self.log = core, rank, log
        self.cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM")
        self.pm = None

    def begin(self, name: str, init_dir: Path | None = None):
        from peft import PeftModel, get_peft_model
        if self.pm is None:
            if init_dir is not None and (init_dir / "adapter_config.json").exists():
                self.pm = PeftModel.from_pretrained(self.core.model, str(init_dir), adapter_name=name, is_trainable=True)
            else:
                self.pm = get_peft_model(self.core.model, self.cfg, adapter_name=name)
        elif init_dir is not None and (init_dir / "adapter_config.json").exists():
            self.pm.load_adapter(str(init_dir), adapter_name=name, is_trainable=True)
        else:
            self.pm.add_adapter(name, self.cfg)
        self.pm.set_adapter(name)
        if not getattr(self, "_ckpt", False):
            # recompute activations in the backward pass instead of keeping 36 layers of them (peak 62 GiB at batch 8 without)
            self.core.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.core.model.enable_input_require_grads()
            self._ckpt = True
        self.name = name
        self.params = [p for n, p in self.pm.named_parameters() if p.requires_grad]
        return sum(p.numel() for p in self.params)

    def end(self, name: str):
        self.pm.delete_adapter(name)

    def batch(self, items: list[dict], with_answer: bool):
        tok = self.core.tok
        seqs, labels = [], []
        for it in items:
            user = (("Knowledge nodes:\n" + it["text"] + "\n\n") if it.get("text") else "") + QUESTION_PREFIX + it["q"]
            p = self.core.ids(self.core.chat_text([{"role": "system", "content": SYS_YAOB}, {"role": "user", "content": user}]))
            if with_answer:
                a = torch.cat([self.core.ids(it["a"])[:72], torch.tensor([tok.eos_token_id])])
                seqs.append(torch.cat([p, a]))
                labels.append(torch.cat([torch.full((len(p),), -100), a]))
            else:
                seqs.append(p)
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), tok.pad_token_id)
        mask = torch.zeros(len(seqs), T, dtype=torch.long)
        lab = torch.full((len(seqs), T), -100)
        for i, s in enumerate(seqs):                 # left padding: answers end at the last position of every row
            ids[i, T - len(s):] = s
            mask[i, T - len(s):] = 1
            if with_answer:
                lab[i, T - len(s):] = labels[i]
        d = self.core.device
        return ids.to(d), mask.to(d), lab.to(d)

    def sample(self, rng: random.Random, ex: dict, node_by_id: dict, pool: list[int], batch: int, open_book: float, all_ids: list[int]) -> list[dict]:
        items = []
        for _ in range(batch):
            nid = pool[rng.randrange(len(pool))]
            kinds = ex[nid]
            x = rng.random()
            kind = "fact" if ("fact" in kinds and x < 0.55) else ("define" if ("define" in kinds and x < 0.7) else "recite")
            e = rng.choice(kinds.get(kind) or kinds["recite"])
            it = {"q": e["q"], "a": e["a"]}
            if rng.random() < open_book:
                other = all_ids[rng.randrange(len(all_ids))]
                ctx = [nid] + ([other] if other != nid else [])
                rng.shuffle(ctx)
                it["text"] = render_nodes([node_by_id[i] for i in ctx])
                it["a"] = e["a"] + f" [{ctx.index(nid) + 1}]"
            items.append(it)
        return items

    def train(self, ctx: JobContext, lock: GpuLock, ex: dict, node_by_id: dict, pool: list[int], steps: int, batch: int, lr: float, seed: int,
              state_dir: Path, start: int = 0, open_book: float = 0.25, replay_pool: list[int] | None = None, label: str = "") -> tuple[int, bool, dict]:
        opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=0.0, betas=(0.9, 0.99))
        warm = max(1, min(50, steps // 10))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.05, 0.5 * (1 + math.cos(math.pi * min(s, steps) / max(1, steps)))))
        if start and (state_dir / "opt.pt").exists():
            o = torch.load(state_dir / "opt.pt", map_location=self.core.device)
            opt.load_state_dict(o["opt"])
            sched.load_state_dict(o["sched"])
        all_ids = sorted(ex)
        self.pm.train()
        t0 = time.perf_counter()
        tokens, yielded, last = 0, 0.0, 0.0
        every = int(ctx.params.get("ckpt_every", 100))
        for step in range(start, steps):
            rng = random.Random(f"{seed}:{self.name}:{step}")
            src = replay_pool if (replay_pool and rng.random() < 0.5) else pool
            ids, mask, lab = self.batch(self.sample(rng, ex, node_by_id, src, batch, open_book, all_ids), True)
            wait = memory_guard(ctx, self.log)
            if wait is None:                       # cancelled while waiting for memory
                self.save_state(state_dir, opt, sched, step, steps)
                return step, True, {"loss": last}
            yielded += wait
            K = int((lab != -100).sum(1).max()) if (lab != -100).any() else 1
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # logits only for the answer positions: the full [batch, length, 151,936] tensor dominated memory
                logits = self.pm(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=K + 1).logits
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), lab[:, -K:].reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            opt.step()
            sched.step()
            tokens += int(mask.sum())
            last = float(loss.detach())
            done = step + 1
            if done % 20 == 0 or done == steps:
                dt = time.perf_counter() - t0 - yielded
                rate = (done - start) / max(dt, 1e-6)
                info = {"stage": f"specialist {label}", "step": done, "total": steps, "loss": round(last, 4), "lr": sched.get_last_lr()[0],
                        "steps_per_s": round(rate, 3), "train_tokens_per_s": round(tokens / max(dt, 1e-6)), "eta_s": round((steps - done) / max(rate, 1e-6)),
                        "torch_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "system_available_gib": round(psutil.virtual_memory().available / 2**30, 1)}
                ctx.log(info)
                ctx.progress(**info)
            stop = ctx.should_stop()
            if done % every == 0 or done == steps or stop:
                self.save_state(state_dir, opt, sched, done, steps)
                ctx.progress(last_checkpoint={"stage": label, "step": done, "path": str(state_dir), "saved": now()}, force=True)
                if lock.exclusive_waiter() and not stop:
                    ctx.log("an exclusive GPU job is waiting; pausing after this checkpoint")
                    ctx.progress(stage="paused for another GPU job", force=True)
                    yielded += lock.yield_if_wanted()
            if stop:
                return done, True, {"loss": last}
        self.pm.eval()
        dt = time.perf_counter() - t0 - yielded
        return steps, False, {"loss": round(last, 4), "wall_s": round(dt, 1), "paused_s": round(yielded, 1), "train_tokens_per_s": round(tokens / max(dt, 1e-6))}

    def save_state(self, d: Path, opt, sched, step: int, steps: int) -> None:
        """Atomic: write beside, then swap directories; the previous state is kept."""
        tmp = d.with_name(d.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        self.pm.save_pretrained(str(tmp), selected_adapters=[self.name])
        inner = tmp / self.name                      # PEFT writes non-default adapters into a sub-directory
        if inner.exists():
            for f in inner.iterdir():
                f.replace(tmp / f.name)
            inner.rmdir()
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict()}, tmp / "opt.pt")
        (tmp / "state.json").write_text(json.dumps({"step": step, "steps": steps, "adapter": self.name, "saved": now()}))
        prev = d.with_name(d.name + ".prev")
        if d.exists():
            if prev.exists():
                shutil.rmtree(prev)
            d.replace(prev)
        tmp.replace(d)

    @torch.inference_mode()
    def answer(self, items: list[dict], use_adapter: bool, max_new: int = 32) -> list[str]:
        out = []
        for b in range(0, len(items), 16):
            ids, mask, _ = self.batch(items[b:b + 16], False)

            def gen():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return self.pm.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
                                            top_k=None, pad_token_id=self.core.tok.pad_token_id)
            if use_adapter:
                g = gen()
            else:
                with self.pm.disable_adapter():
                    g = gen()
            out += [t.strip() for t in self.core.tok.batch_decode(g[:, ids.shape[1]:], skip_special_tokens=True)]
        return out

    def closed_book_check(self, held: list[dict], n: int, seed: int) -> dict:
        """Held-out phrasings of this MPN's stored facts, no node text in the prompt: with the adapter and without it."""
        rng = random.Random(seed)
        held = list(held)
        rng.shuffle(held)
        held = held[:n]
        if not held:
            return {"n": 0}
        items = [{"q": e["q"]} for e in held]
        with_a = self.answer(items, True)
        without = self.answer(items, False)
        k1 = sum(fact_correct(r, e["values"]) for r, e in zip(with_a, held))
        k0 = sum(fact_correct(r, e["values"]) for r, e in zip(without, held))
        return {"n": len(held), "with_adapter": round(k1 / len(held), 4), "core_alone": round(k0 / len(held), 4),
                "examples": [{"q": e["q"], "gold": e["values"][:2], "with_adapter": a, "core_alone": b} for e, a, b in list(zip(held, with_a, without))[:4]]}


# ============================================================================ the job
def _open_core(ctx: JobContext):
    core_name = ctx.params.get("core") or ctx.store.get_setting("core") or os.environ.get("MPN_CORE", "Qwen3-4B-Instruct-2507")
    t0 = time.perf_counter()
    core = Core.load(models_root() / core_name)
    ctx.log(f"core {core_name} loaded in {time.perf_counter() - t0:.1f} s")
    return core, core_name


@job_kind("train_bank")
def train_bank(ctx: JobContext) -> dict:
    """params: router (bool, default true), adapters ("all" | [slug, ...] | [], default []), only_nodes ([ids]),
    rev, seed, base ("active" | checkpoint id | "none"), router_epochs, router_batch, router_lr, exposures, batch, lr, rank, eval_n."""
    p, store = ctx.params, ctx.store
    lock = GpuLock()
    seed = int(p.get("seed", 20261111))
    resume = ctx.resume_dir()
    prev = json.loads((resume / "state.json").read_text()) if resume and (resume / "state.json").exists() else {}
    rev = int(prev.get("rev") or p.get("rev") or store.head())
    ctx.progress(stage="preparing", rev=rev, force=True)
    g = gather(store, rev)
    if not g["nodes"]:
        raise ValueError("there are no live nodes to train")
    only_nodes = [int(x) for x in p.get("only_nodes") or []]
    node_by_id = {n["id"]: n for n in g["nodes"]}
    waited = lock.acquire()
    if waited > 1:
        ctx.log(f"waited {waited:.0f} s for the GPU (an exclusive media job was running)")
    torch.cuda.reset_peak_memory_stats()
    enc = TextEncoder(str(models_root() / os.environ.get("MPN_ENCODER", "Qwen3-Embedding-0.6B")))
    core_name = p.get("core") or store.get_setting("core") or os.environ.get("MPN_CORE", "Qwen3-4B-Instruct-2507")
    rows = allocate_rows(store, [n["id"] for n in g["nodes"]], {"d_key": 256, "d_enc": enc.dim}, core_name)
    n_rows = max(rows.values()) + 1
    spec = RouterSpec(n_rows=n_rows, n_mpn=len(g["mpn_ids"]), d_enc=enc.dim, d_key=256)
    store.set_setting("bank_spec", spec.to_json())
    emb = embed_nodes(enc, g["nodes"], data_root() / "index" / "emb_cache.pt", ctx.log)
    examples = D.build_examples(store, rev, seed)
    node_hash = {n["id"]: n["content_hash"] for n in g["nodes"]}
    ctx.log(f"revision {rev}: {len(g['nodes'])} nodes, {n_rows} rows, {len(examples['train'])} training and {len(examples['held_out'])} held-out examples; hashes {examples['hashes']}")

    # ---- the checkpoint this job builds on
    base = p.get("base", "active")
    base_id, base_dir = active_checkpoint(store) if base == "active" else (None, None)
    if base not in ("active", "none", None):
        r = store.q1("SELECT path FROM checkpoints WHERE id=?", (base,))
        if r is None:
            raise ValueError(f"base checkpoint {base!r} not found")
        base_id, base_dir = base, Path(r["path"])
    ck_id = prev.get("ck_id") or f"ck-{ctx.id}"
    out = data_root() / "banks" / BANK_ID / "checkpoints" / ck_id
    out.mkdir(parents=True, exist_ok=True)
    base_nodes = json.loads((base_dir / "nodes.json").read_text()) if base_dir and (base_dir / "nodes.json").exists() else {}
    base_manifest = json.loads((base_dir / "manifest.json").read_text()) if base_dir and (base_dir / "manifest.json").exists() else {}
    record = {"id": ck_id, "job_id": ctx.id, "bank": BANK_ID, "rev": rev, "seed": seed, "base": base_id, "code_commit": code_commit(), "torch": torch.__version__,
              "encoder": {"name": "Qwen3-Embedding-0.6B", "params": enc.params, "trained": False}, "dataset": examples["hashes"],
              "example_kinds": examples["kinds"], "started": now(), "router": base_manifest.get("router"), "adapters": dict(base_manifest.get("adapters", {})),
              "core": base_manifest.get("core")}
    state = {"rev": rev, "ck_id": ck_id, "router_done": bool(prev.get("router_done")), "adapters_done": list(prev.get("adapters_done", []))}

    def save_state():
        tmp = ctx.dir / "state.json.tmp"
        tmp.write_text(json.dumps(state))
        tmp.replace(ctx.dir / "state.json")
    if resume and resume != ctx.dir:
        for name in ("router", "adapter_state"):
            if (resume / name).exists() and not (ctx.dir / name).exists():
                shutil.copytree(resume / name, ctx.dir / name)
        ctx.log(f"resuming job {resume.name}: router done {state['router_done']}, specialists done {state['adapters_done']}")
    save_state()

    # ================= stage 1: router
    tr = RouterTrainer(spec, g, rows, emb, "cuda", seed)
    router_ck = ctx.dir / "router"
    want_router = bool(p.get("router", True))
    from safetensors.torch import load_file
    if state["router_done"] and (out / "router.safetensors").exists():
        tr.router.load_state_dict(load_file(str(out / "router.safetensors")))
    elif not want_router or only_nodes:
        if base_dir is None:
            raise ValueError("there is no trained router to build on; run with router: true first")
        _load_router_into(tr.router, load_file(str(base_dir / "router.safetensors")))
    if want_router and not state["router_done"]:
        qs, qrows = tr.pairs(g, examples)
        uniq = list(dict.fromkeys(qs))
        # in slices, so the job reports progress and honours cancel during a stage that takes minutes at N10K
        parts_e = []
        for b in range(0, len(uniq), 4096):
            if ctx.should_stop():
                lock.close()
                save_state()
                return {"cancelled": True, "stage": "router: embedding questions", "step": b, "steps": len(uniq)}
            ctx.progress(stage="router: embedding questions", step=b, total=len(uniq), force=True)
            parts_e.append(enc.encode(uniq[b:b + 4096], query=True, batch_size=256).half())
        ue = torch.cat(parts_e) if parts_e else torch.zeros(0, enc.dim, dtype=torch.float16)
        pos = {q: i for i, q in enumerate(uniq)}
        q_emb = ue[[pos[q] for q in qs]].to("cuda")
        q_rows = torch.tensor(qrows, device="cuda")
        r_batch = int(p.get("router_batch", 256))
        r_steps = int(p.get("router_steps") or math.ceil(len(qs) * float(p.get("router_epochs", 4)) / r_batch))
        r_lr = float(p.get("router_lr", 2e-3))
        start, ck = 0, None
        if (router_ck / "state.json").exists():
            st = json.loads((router_ck / "state.json").read_text())
            _load_router_into(tr.router, load_file(str(router_ck / "router.safetensors")))
            ck = torch.load(router_ck / "opt.pt", map_location="cuda")
            start = int(st["step"])
            ctx.log(f"router: exact resume at step {start} (weights, optimiser and schedule restored)")
        elif only_nodes:
            ctx.log(f"router: retraining the keys of {len(only_nodes)} nodes; shared tensors frozen")
        before = {k: v.detach().clone() for k, v in tr.router.state_dict().items()}
        t0 = time.perf_counter()
        opt_state, done, parts, stopped = tr.train(q_emb, q_rows, r_steps, r_batch, r_lr, seed, ctx, start, [rows[n] for n in only_nodes if n in rows] if only_nodes else None, ck)
        router_ck.mkdir(exist_ok=True)
        save_tensors(router_ck / "router.safetensors", tr.router.state_dict(), {"spec": spec.to_json()})
        torch.save(opt_state, router_ck / "opt.pt")
        (router_ck / "state.json").write_text(json.dumps({"step": done, "steps": r_steps}))
        if stopped:
            lock.close()
            return {"cancelled": True, "stage": "router", "step": done, "steps": r_steps}
        after = tr.router.state_dict()
        pc = tr.router.param_counts()
        metrics = tr.evaluate(enc, examples["held_out"], int(p.get("eval_n", 1000)), seed + 1)
        record["router"] = {"spec": spec.to_json(), "learned_params": pc, "steps": r_steps, "batch": r_batch, "lr_shared": r_lr, "lr_per_node": r_lr * 3,
                            "optimizer": "AdamW betas (0.9, 0.99), weight decay 0.01 on shared tensors, warm-up 100 steps then cosine to 5%",
                            "objective": "InfoNCE over in-batch positives and 4 same-MPN negatives + 0.5 * multi-label MPN BCE + 0.25 * gate BCE",
                            "training_pairs": len(qs), "wall_time_s": round(time.perf_counter() - t0, 1), "final_loss": parts,
                            "parameter_change_l2": {k: round(float((after[k].float() - before[k].float()).norm()), 4) for k in after},
                            "held_out": metrics, "trained_at_rev": rev, "job_id": ctx.id}
        ctx.log({"router_held_out": metrics})
        save_tensors(out / "router.safetensors", tr.router.state_dict(), {"spec": spec.to_json()})
        state["router_done"] = True
        save_state()
    elif not (out / "router.safetensors").exists():
        save_tensors(out / "router.safetensors", tr.router.state_dict(), {"spec": spec.to_json()})
    key_trained = set(node_hash) if (want_router and not only_nodes) else ({int(k) for k, v in base_nodes.items() if v.get("key_trained") and node_hash.get(int(k)) == v["hash"]} | set(only_nodes))

    # ================= stage 2: specialists
    want = p.get("adapters", [])
    if want == "all":
        targets = [m for m in g["mpn_ids"]]
    else:
        slug_to_id = {m["slug"]: mid for mid, m in g["mpns"].items()}
        targets = [slug_to_id[s] if s in slug_to_id else int(s) for s in want]
    if only_nodes and not want:
        targets = sorted({g["primary"][n] for n in only_nodes if n in g["primary"] and base_dir and (base_dir / "adapters" / str(g["primary"][n])).exists()})
    # carry over the base checkpoint's adapters that this job does not retrain
    adapter_nodes: dict[int, dict] = {}
    if base_dir and (base_dir / "adapters").exists():
        for d in (base_dir / "adapters").iterdir():
            if d.is_dir() and d.name.isdigit() and int(d.name) not in targets and not (out / "adapters" / d.name).exists():
                link_tree(d, out / "adapters" / d.name)
    core = trainer = None
    if targets:
        core, core_name = _open_core(ctx)
        record["core"] = core.contract.to_json()
        rank = int(p.get("rank", 16))
        trainer = AdapterTrainer(core, rank, ctx.log)
        batch = int(p.get("batch", 8))
        lr = float(p.get("lr", 5e-4))
        for mid in targets:
            if mid in state["adapters_done"]:
                continue
            m = g["mpns"][mid]
            mnodes = [n for n in g["nodes"] if g["primary"].get(n["id"]) == mid]
            ex = adapter_examples(mnodes)
            if not ex:
                ctx.log(f"specialist {m['slug']}: no trainable nodes; skipped")
                state["adapters_done"].append(mid)
                continue
            pool = [n for n in only_nodes if n in ex] if only_nodes else sorted(ex)
            if not pool:
                continue
            exposures = float(p.get("exposures", 60 if only_nodes else (20 if m["kind"] == "diagnostic" else 6)))
            steps = int(p.get("steps") or math.ceil(len(pool) * exposures / batch))
            sdir = ctx.dir / "adapter_state" / str(mid)
            start = 0
            init = None
            if (sdir / "state.json").exists():
                st = json.loads((sdir / "state.json").read_text())
                start, init = int(st["step"]), sdir
                steps = int(st.get("steps", steps))
                ctx.log(f"specialist {m['slug']}: exact resume at step {start} of {steps}")
            elif base_dir and (base_dir / "adapters" / str(mid) / "adapter_config.json").exists() and (only_nodes or p.get("continue_adapters")):
                init = base_dir / "adapters" / str(mid)
                ctx.log(f"specialist {m['slug']}: continuing from the base checkpoint's adapter")
            name = f"mpn{mid}"
            n_params = trainer.begin(name, init)
            replay = [n for n in sorted(ex) if n not in set(pool)] if only_nodes else None
            ctx.log(f"specialist {m['slug']}: {len(pool)} nodes, {steps} steps of {batch}, {n_params:,} trainable parameters (rank {rank})")
            done, stopped, tm = trainer.train(ctx, lock, ex, node_by_id, pool, steps, batch, lr, seed, sdir, start, replay_pool=replay, label=m["slug"])
            if stopped:
                lock.close()
                save_state()
                return {"cancelled": True, "stage": f"specialist {m['slug']}", "step": done, "steps": steps}
            held = [e for e in examples["held_out"] if e["kind"] == "fact" and g["primary"].get(e["node"]) == mid]
            check = trainer.closed_book_check(held, int(p.get("adapter_eval_n", 96)), seed + mid)
            ctx.log({"specialist": m["slug"], "closed_book_check": {k: v for k, v in check.items() if k != "examples"}})
            adir = out / "adapters" / str(mid)
            if adir.exists():
                shutil.rmtree(adir)
            adir.mkdir(parents=True)
            for f in sdir.iterdir():
                if f.name.startswith("adapter_"):
                    shutil.copy2(f, adir / f.name)
            trained_nodes = {str(n): node_hash[n] for n in (sorted(ex) if not only_nodes else pool)}
            if only_nodes and (base_dir / "adapters" / str(mid) / "adapter.json").exists():
                old = json.loads((base_dir / "adapters" / str(mid) / "adapter.json").read_text()).get("nodes", {})
                trained_nodes = {**{k: v for k, v in old.items() if node_hash.get(int(k)) == v}, **trained_nodes}
            info = {"mpn_id": mid, "slug": m["slug"], "rank": rank, "alpha": 2 * rank, "targets": LORA_TARGETS, "trainable_params": n_params,
                    "bytes": dir_bytes(adir), "sha256": sha256_file(adir / "adapter_model.safetensors"), "steps": steps, "batch": batch, "lr": lr,
                    "exposures_per_node": round(steps * batch / len(pool), 2), "open_book_fraction": 0.25, "max_answer_tokens": 72,
                    "optimizer": "AdamW betas (0.9, 0.99), no weight decay, warm-up then cosine to 5%, grad clip 1.0", "precision": "core bf16 autocast; adapter float32",
                    "objective": "LM loss on node answers, 75% closed book, 25% with node text and citation", "core": core.contract.name, "core_revision": core.contract.revision,
                    "trained_at_rev": rev, "job_id": ctx.id, "training": tm, "closed_book_check": check, "nodes": trained_nodes,
                    "validated": bool(check.get("n") and check["with_adapter"] > check["core_alone"])}
            (adir / "adapter.json").write_text(json.dumps(info, indent=1))
            record["adapters"][str(mid)] = {k: v for k, v in info.items() if k not in ("nodes",)}
            trainer.end(name)
            state["adapters_done"].append(mid)
            save_state()

    # ================= package and register
    ctx.progress(stage="packaging", force=True)
    for d in sorted((out / "adapters").iterdir()) if (out / "adapters").exists() else []:
        if (d / "adapter.json").exists():
            a = json.loads((d / "adapter.json").read_text())
            adapter_nodes[int(d.name)] = a.get("nodes", {})
            record["adapters"].setdefault(d.name, {k: v for k, v in a.items() if k != "nodes"})
    pk = package_router(out, tr.router, rows, node_hash, g["primary"], g["shared"], tr.mpn_index, tr.node_emb.float().cpu())
    nodes_json = {}
    for nid, h in node_hash.items():
        an = adapter_nodes.get(g["primary"].get(nid), {})
        nodes_json[str(nid)] = {"row": rows[nid], "hash": h, "key_trained": nid in key_trained, "adapter_trained": an.get(str(nid)) == h}
    (out / "nodes.json").write_text(json.dumps(nodes_json))
    idx = data_root() / "index"
    idx.mkdir(parents=True, exist_ok=True)
    np.save(out / "node_emb.f16.npy", tr.node_emb.cpu().numpy())
    pc = tr.router.param_counts()
    adapter_params = sum(a.get("trainable_params", 0) for a in record["adapters"].values())
    record.update(pk, finished=now(), nodes=len(node_hash), key_trained_nodes=len(key_trained),
                  adapter_trained_nodes=sum(1 for v in nodes_json.values() if v["adapter_trained"]),
                  learned_params={"router": pc, "adapters_total": adapter_params, "adapters": {k: a.get("trainable_params", 0) for k, a in record["adapters"].items()},
                                  "total": pc["total"] + adapter_params},
                  peak_torch_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2), peak_process_rss_gib=round(psutil.Process().memory_info().rss / 2**30, 2),
                  packaged_at=now())
    tmp = out / "manifest.json.tmp"
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(out / "manifest.json")
    (ctx.dir / "run.json").write_text(json.dumps(record, indent=1))
    metrics = {"router": (record.get("router") or {}).get("held_out"), "adapters": {a["slug"]: {k: v for k, v in a.get("closed_book_check", {}).items() if k != "examples"}
                                                                                    for a in record["adapters"].values() if "slug" in a}}
    store.conn().execute("INSERT OR REPLACE INTO checkpoints(id, bank_id, kind, path, parent, step, rev, metrics, sha256, status, created, job_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                         (ck_id, BANK_ID, "router+adapters", str(out), base_id, 0, rev, json.dumps(metrics), sha256_file(out / "router.safetensors"), "saved", now(), ctx.id))
    lock.close()
    return {"checkpoint": ck_id, "path": str(out), "router": (record.get("router") or {}).get("held_out", {}).get("learned"),
            "specialists": {a["slug"]: a.get("closed_book_check", {}).get("with_adapter") for a in record["adapters"].values() if "slug" in a},
            "learned_params": record["learned_params"]["total"], "note": "Saved and packaged. Activate it to use it."}


def _load_router_into(router: Router, sd: dict) -> None:
    """Load router tensors, tolerating a bank that has grown since (new rows keep their zero initialisation)."""
    cur = router.state_dict()
    for k, v in sd.items():
        if k not in cur:
            continue
        if cur[k].shape == v.shape:
            cur[k].copy_(v)
        elif k in Router.PER_NODE and v.shape[1:] == cur[k].shape[1:] and v.shape[0] <= cur[k].shape[0]:
            cur[k][: v.shape[0]].copy_(v)
        elif k == "mpn_proto" and v.shape[1] == cur[k].shape[1] and v.shape[0] <= cur[k].shape[0]:
            cur[k][: v.shape[0]].copy_(v)          # MPNs added since keep a fresh prototype
        else:
            raise ValueError(f"tensor {k} has shape {tuple(v.shape)}; the router expects {tuple(cur[k].shape)}")


@job_kind("retrain_nodes")
def retrain_nodes(ctx: JobContext) -> dict:
    """Retrain only the given nodes: their routing keys (shared router tensors frozen) and their MPNs' specialists (with replay)."""
    if not ctx.params.get("only_nodes"):
        raise ValueError("retrain_nodes needs only_nodes")
    ctx.params.setdefault("router", True)
    ctx.params.setdefault("router_steps", 200)
    ctx.params.setdefault("ckpt_every", 40)
    return train_bank(ctx)


def apply_checkpoint_state(store: Store, ck_dir: Path, ck_id: str) -> dict:
    """Record, per node, what the newly active checkpoint covers. Called when a checkpoint is activated."""
    nodes = json.loads((ck_dir / "nodes.json").read_text())
    c = store.conn()
    cur = {r[0]: r[1] for r in store.q("SELECT node_id, content_hash FROM node_v WHERE rev_to IS NULL")}
    with store._wlock:
        c.execute("BEGIN IMMEDIATE")
        try:
            c.execute("UPDATE derived_state SET learned_hash=NULL, learned_checkpoint=NULL")
            for nid_s, info in nodes.items():
                nid = int(nid_s)
                learned = info["hash"] if (info.get("key_trained") or info.get("adapter_trained")) else None
                c.execute("INSERT INTO derived_state(node_id, embed_hash, embed_row, learned_hash, learned_checkpoint) VALUES(?,?,?,?,?) "
                          "ON CONFLICT(node_id) DO UPDATE SET embed_row=excluded.embed_row, learned_hash=excluded.learned_hash, learned_checkpoint=excluded.learned_checkpoint",
                          (nid, info["hash"] if cur.get(nid) == info["hash"] else None, info["row"], learned, ck_id if learned else None))
                if cur.get(nid) == info["hash"]:
                    c.execute("UPDATE derived_state SET embed_hash=? WHERE node_id=?", (info["hash"], nid))
                    c.execute("DELETE FROM overlay WHERE node_id=? AND content_hash=?", (nid, info["hash"]))
            c.execute("UPDATE bank_rows SET state='allocated' WHERE bank_id=?", (BANK_ID,))
            c.executemany("UPDATE bank_rows SET state='trained' WHERE bank_id=? AND node_id=?",
                          [(BANK_ID, int(n)) for n, i in nodes.items() if i.get("key_trained") or i.get("adapter_trained")])
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise
    return {"nodes": len(nodes), "key_trained": sum(1 for i in nodes.values() if i.get("key_trained")),
            "adapter_trained": sum(1 for i in nodes.values() if i.get("adapter_trained"))}
