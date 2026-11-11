"""The Yaob runtime: one resident core, selectively loaded MPN state, and an auditable request path.

A request is answered in one of three modes, which are also the evaluation arms:

  core        the unmodified core; no node is read, no adapter is applied
  retrieval   ordinary retrieval: frozen-encoder similarity plus text search over node text; nodes are given as text
  yaob        learned routing (MPN prototypes, then per-node keys and gates inside the selected MPNs), the selected
              MPNs' specialist adapters applied to the core, the best nodes also given as text, procedure nodes
              executed, answers checked against node facts

What is neural and what is not, per request:
  learned      query projection, MPN prototypes, per-node routing keys and gates (the router); per-MPN LoRA specialists
  retrieval    text of selected nodes placed in the prompt; text-search candidates
  tools        procedure results (computed by code), validation checks
  pretrained   the core's generation; the frozen encoder's embeddings

Knowledge-cell content reaches the core in two ways only: as text (retrieval, labelled as such) and through the
specialist adapter of the node's MPN, which is shared by all of that MPN's nodes. There is no per-node learned
memory of content; per-node learned state is the routing key (Thinking cell) and the gate bias (Reasoning cell).

Every request pins one model revision and reads all node content at that revision. The evidence it
returns lists what was actually loaded and read. It is a record of access, not of reasoning.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

import psutil
import torch

from ..model.bank import MpnResidency, SharedRouter, from_f16_blob, to_f16_blob
from ..model.core import Core
from ..model.encoder import TextEncoder
from ..model.prompts import QUESTION_PREFIX, SYS_CORE, SYS_RETRIEVAL, SYS_YAOB, render_nodes
from ..store.db import Store, now, sid
from . import budget as tokbudget
from . import conversation as conv
from .conversation import relations_named
from . import procedures

KV_MIB_PER_TOKEN = 0.22          # measured: 5,616 prompt tokens added 1.19 GiB at peak (docs/core-selection.md)
BASE_REQUEST_MIB = 96


class Cancelled(Exception):
    pass


DEFAULT_OPTS = {"router": "learned",     # "learned" | "baseline" (frozen-encoder + text retrieval chooses the nodes)
                "adapters": True,        # apply the selected MPNs' specialist adapters
                "text": True}            # give the best selected nodes to the core as text


class Request:
    def __init__(self, request_id: str | None, mode: str, rev: int, opts: dict | None = None):
        self.id = request_id or uuid.uuid4().hex[:12]
        self.mode, self.rev = mode, rev
        self.opts = dict(DEFAULT_OPTS)
        self.opts.update({k: v for k, v in (opts or {}).items() if k in DEFAULT_OPTS})
        self.cancel = threading.Event()
        self.pinned: list[int] = []
        self.t0 = time.perf_counter()


class Engine:
    def __init__(self, store: Store, data_root: Path, models_root: Path, device: str = "cuda"):
        self.store, self.data_root, self.models_root, self.device = store, Path(data_root), Path(models_root), device
        self.core: Core | None = None
        self.enc: TextEncoder | None = None
        self.shared: SharedRouter | None = None
        self.cache: MpnResidency | None = None
        self.checkpoint: dict | None = None
        self.ck_nodes: dict[int, dict] = {}       # node id -> {row, hash, key_trained, adapter_trained} of the active checkpoint
        self.pm = None                            # PEFT wrapper around the core, created when the first adapter loads
        self.adapter_lock = threading.RLock()
        self.gen_lock = threading.Lock()
        self.requests: dict[str, Request] = {}
        self.queued = 0
        self.flat_index: dict | None = None
        self.budget: dict = {}
        self.load_info: dict = {}
        self.index_lock = threading.Lock()
        self.last_error: str | None = None
        self.counters = {"requests": 0, "cancelled": 0, "refused": 0}

    # ------------------------------------------------------------------ loading
    def load_core(self, name: str | None = None) -> dict:
        name = name or self.store.get_setting("core") or os.environ.get("MPN_CORE", "Qwen3-4B-Instruct-2507")
        t0 = time.perf_counter()
        before = psutil.virtual_memory().available
        self.core = Core.load(self.models_root / name, self.device)
        self.enc = TextEncoder(str(self.models_root / os.environ.get("MPN_ENCODER", "Qwen3-Embedding-0.6B")), self.device)
        torch.cuda.synchronize()
        self.load_info = {"core_load_s": round(time.perf_counter() - t0, 1),
                          "system_available_drop_gib": round((before - psutil.virtual_memory().available) / 2**30, 2), "loaded_at": now()}
        return self.load_info

    def unload_core(self) -> dict:
        with self.gen_lock:
            if self.cache:
                self.cache.drop()
            self.pm = None
            self.core = None
            self.enc = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        return {"unloaded": True}

    # ------------------------------------------------------------------ specialist adapters (PEFT LoRA on the frozen core)
    def _load_adapter(self, mpn_id: int, adir: Path) -> str:
        from peft import PeftModel
        name = f"mpn{mpn_id}"
        with self.adapter_lock:
            if self.pm is None:
                self.pm = PeftModel.from_pretrained(self.core.model, str(adir), adapter_name=name, is_trainable=False)
                self.pm.eval()
            elif name not in self.pm.peft_config:
                self.pm.load_adapter(str(adir), adapter_name=name, is_trainable=False)
        return name

    def _unload_adapter(self, name: str) -> None:
        with self.adapter_lock:
            if self.pm is None or name not in self.pm.peft_config:
                return
            if len(self.pm.peft_config) == 1:
                # removing the last adapter: take the LoRA layers out of the core again
                self.core.model = self.pm.unload()
                self.pm = None
            else:
                others = [n for n in self.pm.peft_config if n != name]
                self.pm.base_model.set_adapter(others[0])
                self.pm.delete_adapter(name)
        torch.cuda.empty_cache()

    def _adapters(self, names: list[str]):
        """Context in which exactly `names` are active; with no names, every adapter is off (the plain core)."""
        if self.pm is None:
            return contextlib.nullcontext()
        if not names:
            return self.pm.disable_adapter()
        self.pm.base_model.set_adapter(names if len(names) > 1 else names[0])
        return contextlib.nullcontext()

    def _generate(self, **kw):
        """`generate` on whichever object currently owns the core's forward (the PEFT wrapper once adapters exist)."""
        m = self.pm if self.pm is not None else self.core.model
        return m.generate(**kw)

    def activate(self, ck_id: str | None = None) -> dict:
        """Load a packaged checkpoint's shared router and point MPN residency at it. Refuses a checkpoint trained for another core."""
        if ck_id is None:
            ck_id = (self.store.get_setting("active_checkpoint") or {}).get("id")
        if not ck_id:
            if self.cache:
                self.cache.drop()
            self.shared, self.cache, self.checkpoint, self.ck_nodes = None, None, None, {}
            return {"active": None, "note": "No trained checkpoint is active. Yaob mode is unavailable until one is activated; retrieval and core modes work."}
        row = self.store.q1("SELECT * FROM checkpoints WHERE id=?", (ck_id,))
        if row is None:
            raise ValueError(f"unknown checkpoint {ck_id}")
        ck_dir = Path(row["path"])
        manifest = json.loads((ck_dir / "manifest.json").read_text())
        if self.core is not None:
            want = manifest.get("core") or {}
            have = self.core.contract
            if want.get("hidden_size") not in (None, have.hidden_size) or (want.get("revision") and have.revision and want["revision"] != have.revision):
                raise ValueError(f"checkpoint {ck_id} was trained for core {want.get('name')} ({want.get('revision', '')[:8]}); "
                                 f"the loaded core is {have.name} ({have.revision[:8]}). Specialist adapters are tied to the core they were trained on.")
        if self.cache:
            self.cache.drop()
        self.shared = SharedRouter(ck_dir, self.device)
        b = self.get_budget()
        self.cache = MpnResidency(ck_dir, int(b["resident_cache_mib"]) * 2**20, int(b["max_resident_mpns"]),
                                  load_adapter=self._load_adapter if self.core is not None else None, unload_adapter=self._unload_adapter)
        self.ck_nodes = {int(k): v for k, v in json.loads((ck_dir / "nodes.json").read_text()).items()} if (ck_dir / "nodes.json").exists() else {}
        self.checkpoint = {"id": ck_id, "path": str(ck_dir), "manifest": manifest}
        from ..train.bank_jobs import apply_checkpoint_state
        applied = apply_checkpoint_state(self.store, ck_dir, ck_id)
        self.flat_index = None
        prev = self.store.get_setting("active_checkpoint")
        self.store.set_setting("active_checkpoint", {"id": ck_id, "activated": now(), "previous": (prev or {}).get("id") if (prev or {}).get("id") != ck_id else (prev or {}).get("previous")})
        self.store.conn().execute("UPDATE checkpoints SET status='previous' WHERE status='active' AND id<>?", (ck_id,))
        self.store.conn().execute("UPDATE checkpoints SET status='active' WHERE id=?", (ck_id,))
        adapters = sorted(int(d.name) for d in (ck_dir / "adapters").iterdir() if (d / "adapter_config.json").exists()) if (ck_dir / "adapters").exists() else []
        return {"active": ck_id, "shared_bytes": self.shared.bytes, "shards": len(manifest.get("shards", {})), "adapters": [sid(m) for m in adapters], **applied}

    def get_budget(self) -> dict:
        from ..api.app import DEFAULT_BUDGET
        b = dict(DEFAULT_BUDGET)
        b.update(self.store.get_setting("budget") or {})
        b.update(self.budget)
        b, note = tokbudget.repair_budget(b)
        if note:
            b["repair_note"] = note
        return b

    def set_budget(self, b: dict) -> dict:
        self.budget = dict(b)
        evicted = []
        if self.cache:
            evicted = self.cache.set_limits(int(b["resident_cache_mib"]) * 2**20, int(b["max_resident_mpns"]))
        return {"evicted": [sid(m) for m in evicted]}

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        p = psutil.Process()
        st: dict = {"available": True, "core": None, "encoder": None, "checkpoint": None, "learned": None,
                    "requests": {"active": len(self.requests), "queued": self.queued, **self.counters},
                    "memory": {"process_rss_mib": round(p.memory_info().rss / 2**20, 1),
                               "torch_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1) if torch.cuda.is_available() else 0,
                               "torch_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else 0,
                               "system_available_gib": round(psutil.virtual_memory().available / 2**30, 1)},
                    "load": self.load_info, "last_error": self.last_error}
        if self.core:
            c = self.core.contract
            st["core"] = {"name": c.name, "repo": c.repo, "revision": c.revision, "license": c.license, "architecture": c.architecture,
                          "params_total": c.params_total, "hidden_size": c.hidden_size, "precision": c.precision, "loaded": True,
                          "max_context": c.max_context}
            lim = tokbudget.resolve(self.get_budget(), c.max_context)
            st["limits"] = {"context_tokens": lim["context"], "output_tokens": lim["output"], "margin": lim["margin"],
                            "max_input_tokens": lim["context"] - lim["output"] - lim["margin"],
                            "max_active_adapters": int(self.get_budget().get("max_active_adapters", 0))}
            st["encoder"] = ({"name": "Qwen3-Embedding-0.6B", "params_total": self.enc.params, "dim": self.enc.dim, "trained_here": False}
                             if self.enc is not None else {"loaded": False})      # the core loads first; the encoder follows
        else:
            st["core"] = {"loaded": False}
        if self.checkpoint:
            m = self.checkpoint["manifest"]
            st["checkpoint"] = {"id": self.checkpoint["id"], "step": m.get("step"), "rev": m.get("rev"), "nodes": m.get("nodes"),
                                "trained_nodes": m.get("trained_nodes"), "packaged_at": m.get("packaged_at"), "job_id": m.get("job_id")}
            lp = m.get("learned_params") or {}
            rp = lp.get("router") or {}
            st["learned"] = {"router parameters": rp.get("total"), "of which per node": rp.get("per_node_total"),
                             "specialist adapters": len(lp.get("adapters") or {}), "adapter parameters": lp.get("adapters_total"),
                             "total learned": lp.get("total")}
            st["residency"] = {"mpns": self.residency(), "cache_bytes": self.cache.bytes(), "cache_limit_bytes": self.cache.limit,
                               "shared_bytes": self.shared.bytes, "stats": self.cache.stats,
                               "adapters_loaded": sorted(self.pm.peft_config) if self.pm is not None else []}
        pend = self.store.q1("SELECT COUNT(*) FROM node_v n LEFT JOIN derived_state d ON d.node_id=n.node_id "
                             "WHERE n.rev_to IS NULL AND n.status='live' AND (d.embed_hash IS NULL OR d.embed_hash<>n.content_hash)")[0]
        overlay = self.store.q1("SELECT COUNT(*) FROM overlay")[0]
        st["index"] = {"pending": pend, "rows": len(self.ck_nodes) + overlay, "overlay_rows": overlay,
                       "flat_index_loaded": self.flat_index is not None}
        return st

    def residency(self) -> list[dict]:
        names = {m["id"]: m["name"] for m in self.store.mpns()}
        res = {r["mpn_id"]: r for r in (self.cache.residency() if self.cache else [])}
        return [dict(res.get(mid, {"mpn_id": mid, "resident": False, "bytes": 0}), name=n) for mid, n in names.items()]

    # ------------------------------------------------------------------ indexing (text-derived state for new and edited nodes)
    def reindex(self, limit: int = 2000) -> dict:
        """Embed nodes whose content has no current embedding and store them in the overlay."""
        if self.enc is None:
            raise RuntimeError("the encoder is not loaded")
        with self.index_lock:
            rows = self.store.q("SELECT n.node_id, n.title, n.cells, n.content_hash FROM node_v n LEFT JOIN derived_state d ON d.node_id=n.node_id "
                                "WHERE n.rev_to IS NULL AND n.status<>'tombstone' AND (d.embed_hash IS NULL OR d.embed_hash<>n.content_hash) LIMIT ?", (limit,))
            if not rows:
                return {"indexed": 0, "pending": 0}
            t0 = time.perf_counter()
            node_texts = [node_text(r["title"], [s.get("text", "") for s in json.loads(r["cells"]).get("knowledge", {}).get("statements", [])])
                          for r in rows]
            ne = self.enc.encode(node_texts)
            c = self.store.conn()
            with self.store._wlock:
                c.execute("BEGIN IMMEDIATE")
                try:
                    for i, r in enumerate(rows):
                        c.execute("INSERT OR REPLACE INTO overlay(node_id, content_hash, slots, dim, node_emb, stmt_emb, updated) VALUES(?,?,?,?,?,?,?)",
                                  (r["node_id"], r["content_hash"], 0, self.enc.dim, to_f16_blob(ne[i]), b"", now()))
                        c.execute("INSERT INTO derived_state(node_id, embed_hash) VALUES(?,?) ON CONFLICT(node_id) DO UPDATE SET embed_hash=excluded.embed_hash",
                                  (r["node_id"], r["content_hash"]))
                    c.execute("COMMIT")
                except BaseException:
                    c.execute("ROLLBACK")
                    raise
            self.flat_index = None        # the baseline index is rebuilt lazily
            pend = self.store.q1("SELECT COUNT(*) FROM node_v n LEFT JOIN derived_state d ON d.node_id=n.node_id "
                                 "WHERE n.rev_to IS NULL AND n.status<>'tombstone' AND (d.embed_hash IS NULL OR d.embed_hash<>n.content_hash)")[0]
            return {"indexed": len(rows), "pending": pend, "ms": round((time.perf_counter() - t0) * 1000, 1)}

    # ------------------------------------------------------------------ baseline flat index (retrieval mode and semantic search)
    def _flat(self) -> dict:
        """Frozen-encoder embeddings of every indexed node, built from shard files and the overlay. Used only by the
        retrieval baseline and by semantic search; the learned path never loads it."""
        if self.flat_index is not None:
            return self.flat_index
        ids, vecs = [], []
        over = {r["node_id"]: r for r in self.store.q("SELECT node_id, content_hash, dim, node_emb FROM overlay")}
        ck_emb = Path(self.checkpoint["path"]) / "node_emb.f16.npy" if self.checkpoint else None
        if ck_emb is not None and ck_emb.exists() and self.ck_nodes:
            import numpy as np
            arr = np.load(ck_emb, mmap_mode="r")
            cur = {r[0]: r[1] for r in self.store.q("SELECT node_id, content_hash FROM node_v WHERE rev_to IS NULL AND status<>'tombstone'")}
            keep_rows, keep_ids = [], []
            for nid, info in self.ck_nodes.items():
                row, h = info["row"], info["hash"]
                if nid in cur and cur[nid] == h and nid not in over and row < arr.shape[0]:
                    keep_rows.append(row)
                    keep_ids.append(nid)
            if keep_rows:
                vecs.append(torch.from_numpy(arr[keep_rows].astype("float32")))
                ids += keep_ids
        if over:
            vecs.append(torch.stack([from_f16_blob(r["node_emb"], (r["dim"],)).float() for r in over.values()]))
            ids += list(over)
        mat = torch.cat(vecs).to(self.device) if vecs else torch.zeros(0, self.enc.dim, device=self.device)
        self.flat_index = {"ids": ids, "mat": mat, "bytes": mat.numel() * 4}
        return self.flat_index

    def semantic(self, q: str, limit: int = 20, filters: dict | None = None) -> list[dict]:
        if self.enc is None:
            raise RuntimeError("the encoder is not loaded")
        flat = self._flat()
        if not flat["ids"]:
            return []
        qv = self.enc.encode([q], query=True).to(self.device)
        sims = (flat["mat"] @ qv[0])
        top = sims.topk(min(len(flat["ids"]), limit * 4))
        ids = [flat["ids"][i] for i in top.indices.tolist()]
        nodes = {int(n["id"]): n for n in self.store.get_nodes(ids)}
        f = filters or {}
        out = []
        for nid, s in zip(ids, top.values.tolist()):
            n = nodes.get(nid)
            if not n or n["status"] == "tombstone":
                continue
            if any(f.get(k) and n[k] != f[k] for k in ("type", "status", "validation", "origin")):
                continue
            if f.get("mpn_id") and not self.store.q1("SELECT 1 FROM member_v WHERE node_id=? AND mpn_id=? AND rev_to IS NULL", (nid, int(f["mpn_id"]))):
                continue
            out.append({"id": n["id"], "title": n["title"], "type": n["type"], "status": n["status"], "validation": n["validation"],
                        "summary": n["summary"][:240], "score": round(float(s), 4)})
            if len(out) >= limit:
                break
        return out

    # ------------------------------------------------------------------ routing
    def _enabled_mpns(self, rev: int) -> dict[int, dict]:
        return {int(m["id"]): m for m in self.store.mpns(rev) if m["installed"] and m["enabled"]}

    @torch.inference_mode()
    def route(self, req: Request, question: str, budget: dict) -> dict:
        """Hierarchical learned routing. Loads only the shards of the MPNs it selects."""
        t0 = time.perf_counter()
        ev: dict = {"mpns": [], "skipped": [], "shards": [], "nodes": [], "reductions": []}
        mem = self.shared.router
        qe = self.enc.encode([question], query=True).to(self.device)
        qk = mem.query_keys(qe)[0]
        all_mpns = {int(m["id"]): m for m in self.store.mpns(req.rev)}
        enabled = {k: m for k, m in all_mpns.items() if m["installed"] and m["enabled"]}
        probs = torch.sigmoid(mem.mpn_logits(qk[None])[0]).tolist()
        scored = []
        for mid, idx in self.shared.mpn_index.items():
            if mid in all_mpns:
                scored.append((probs[idx], mid))
        scored.sort(reverse=True)
        # exact title matches keep routing robust for named entities
        lex = self.store.search_lexical(question, limit=12, any_terms=True)
        lex_ids = [int(r["id"]) for r in lex]
        title_hit_mpns: dict[int, str] = {}
        ql = question.lower()
        for r in lex[:6]:
            if len(r["title"]) >= 4 and r["title"].lower() in ql:
                for m in self.store.memberships(int(r["id"]), req.rev):
                    title_hit_mpns.setdefault(int(m["mpn_id"]), r["title"])
        active: list[int] = []
        max_active = int(budget["max_active_mpns"])
        for p, mid in scored:
            m = all_mpns[mid]
            wanted = p >= 0.5 or (not active and mid in enabled and p == max(x for x, i in scored if i in enabled)) or mid in title_hit_mpns
            if not wanted:
                continue
            if mid not in enabled:
                ev["skipped"].append({"id": sid(mid), "name": m["name"], "score": round(p, 3), "reason": "disabled" if m["installed"] else "not installed"})
                continue
            if len(active) >= max_active:
                ev["skipped"].append({"id": sid(mid), "name": m["name"], "score": round(p, 3), "reason": "over the active-MPN budget"})
                ev["reductions"].append(f"{m['name']} was relevant but the budget allows {max_active} MPNs per request.")
                continue
            active.append(mid)
        for p, mid in scored:
            ev["mpns"].append({"id": sid(mid), "name": all_mpns[mid]["name"], "score": round(p, 3), "active": mid in active,
                               **({"reason": f"title match: {title_hit_mpns[mid]}"} if mid in title_hit_mpns and p < 0.5 else {})})
        ev["mpns"] = [m for m in ev["mpns"] if m["active"]] + [m for m in ev["mpns"] if not m["active"]][:4]
        # candidates from each active MPN's shard
        cands: dict[int, dict] = {}
        n_adapters = int(budget.get("max_active_adapters", 0)) if req.opts["adapters"] else 0
        for rank_i, mid in enumerate(active):          # active is in routing-score order: specialists for the top MPNs only
            shard, e = self.cache.acquire(mid, want_adapter=rank_i < n_adapters)
            e["name"] = all_mpns[mid]["name"]
            e["mpn_id"] = sid(mid)
            ev["shards"].append(e)
            if shard is None:
                if e["state"] == "over budget":
                    ev["reductions"].append(f"{all_mpns[mid]['name']} could not be loaded within the resident-memory budget.")
                continue
            req.pinned.append(mid)
            t = shard.tensors
            if len(t["node_ids"]):
                sim = t["keys"].to(self.device).float() @ qk
                k = min(len(sim), int(budget["retrieval_candidates"]))
                top = sim.topk(k)
                gate = mem.gate_scale * (top.values - mem.gate_center) + t["gate_bias"].to(self.device)[top.indices]
                for i, s, g in zip(top.indices.tolist(), top.values.tolist(), torch.sigmoid(gate).tolist()):
                    nid = int(t["node_ids"][i])
                    if nid not in cands or s > cands[nid]["score"]:
                        cands[nid] = {"id": nid, "score": s, "gate": g, "via_mpn": mid, "home": mid, "row": i, "hash": shard.hashes[i],
                                      "trained": bool(self.ck_nodes.get(nid, {}).get("key_trained")), "via": "router"}
            if len(t["member_ids"]):
                sim = t["member_keys"].to(self.device).float() @ qk
                top = sim.topk(min(len(sim), max(4, int(budget["retrieval_candidates"]) // 4)))
                gate = mem.gate_scale * (top.values - mem.gate_center) + t["member_gate_bias"].to(self.device)[top.indices]
                for i, s, g in zip(top.indices.tolist(), top.values.tolist(), torch.sigmoid(gate).tolist()):
                    nid = int(t["member_ids"][i])
                    if nid not in cands or s > cands[nid]["score"]:
                        cands[nid] = {"id": nid, "score": s, "gate": g, "via_mpn": mid, "home": int(t["member_home"][i]), "row": None,
                                      "hash": shard.member_hashes[i], "trained": None, "via": "router"}
        # overlay: new and edited nodes that belong to an active MPN (text-derived state only)
        if active:
            ph = ",".join("?" * len(active))
            wm, am = self.store.at(req.rev, "m")
            orows = self.store.q(f"SELECT DISTINCT o.node_id, o.content_hash, o.dim, o.node_emb, m.mpn_id FROM overlay o JOIN member_v m ON m.node_id=o.node_id AND {wm} "
                                 f"WHERE m.mpn_id IN ({ph}) LIMIT 4000", (*am, *active))
            if orows:
                oe = torch.stack([from_f16_blob(r["node_emb"], (r["dim"],)).float() for r in orows]).to(self.device)
                ok = torch.nn.functional.normalize(mem.enc_to_key(oe), dim=-1)
                sim = ok @ qk
                gate = torch.sigmoid(mem.gate_scale * (sim - mem.gate_center))
                for r, s, g in zip(orows, sim.tolist(), gate.tolist()):
                    cands[r["node_id"]] = {"id": r["node_id"], "score": s, "gate": g, "via_mpn": r["mpn_id"], "home": None, "row": None,
                                           "hash": r["content_hash"], "trained": False, "via": "router", "overlay": True}
        # text matches inside the active MPNs join the pool (ordinary retrieval, labelled as such)
        if lex_ids and active:
            ph = ",".join("?" * len(lex_ids))
            pa = ",".join("?" * len(active))
            wm, am = self.store.at(req.rev, "m")
            inside = {r[0]: r[1] for r in self.store.q(f"SELECT m.node_id, m.mpn_id FROM member_v m WHERE {wm} AND m.node_id IN ({ph}) AND m.mpn_id IN ({pa})",
                                                       (*am, *lex_ids, *active))}
            for rank, nid in enumerate(lex_ids):
                if nid in inside and nid not in cands:
                    cands[nid] = {"id": nid, "score": 0.0, "gate": max(0.2, 0.75 - 0.08 * rank), "via_mpn": inside[nid], "home": None, "row": None,
                                  "hash": None, "trained": None, "via": "text match", "lex_rank": rank}
                elif nid in cands:
                    cands[nid]["lex_rank"] = rank
        ranked = sorted(cands.values(), key=lambda c: -(c["gate"] + (0.25 / (1 + c["lex_rank"]) if "lex_rank" in c else 0.0)))
        ev["route_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        ev["candidates"] = len(ranked)
        return {"evidence": ev, "ranked": ranked, "active": active, "qk": qk, "question": question}

    # ------------------------------------------------------------------ reading selected nodes
    def _select(self, req: Request, routed: dict, budget: dict) -> list[dict]:
        """Resolve candidates against the pinned revision: drop deleted, disabled and changed nodes; bounded referral expansion."""
        ev = routed["evidence"]
        want = int(budget["selected_nodes"])
        ids = [c["id"] for c in routed["ranked"][: want * 4]]
        recs = {int(n["id"]): n for n in self.store.get_nodes(ids, req.rev, full=True)}
        chosen: list[dict] = []
        for c in routed["ranked"]:
            if len(chosen) >= want:
                break
            n = recs.get(c["id"])
            if n is None or n["status"] != "live":
                continue
            if c["gate"] < 0.35 and len(chosen) >= 1:
                break
            c["node"] = n
            c["stale"] = c["hash"] is not None and c["hash"] != n["content_hash"]
            chosen.append(c)
        # referrals: follow typed links of the best nodes, one hop per unit of traversal depth.
        # referral_mode "relation" (the default since 12 Nov 2026, chosen on development sets: two-hop 47/60 vs 25/60 for
        # "plain", fact questions 75/80 for both; docs/measurements/multihop-dev-002): read all of a node's links, follow
        # first the links whose relation the question names ("place of birth" -> born_in), let those past the selection
        # cap, and let them enter any enabled MPN. Other links, and every link in "plain" mode, stay inside active MPNs.
        depth = int(budget["traversal_depth"])
        by_relation = budget.get("referral_mode", "relation") == "relation"
        named = relations_named(routed.get("question", "")) if by_relation else set()
        enabled_set = None
        if depth > 0 and chosen:
            seen = {c["id"] for c in chosen}
            frontier = chosen[:2]
            for _ in range(depth):
                nxt = []
                for c in frontier:
                    edges = self.store.edges_of(c["id"], req.rev, limit=200 if by_relation else 12)
                    if by_relation:
                        edges.sort(key=lambda e: e["type"] not in named)      # stable: named relations first, then by weight
                    for e in edges:
                        if e["type"] == "mentions" or e["direction"] != "out":
                            continue
                        oid = int(e["other"]["id"])
                        asked = e["type"] in named
                        if oid in seen or (len(chosen) >= want + 2 and not (asked and len(chosen) < want + 4)):
                            continue
                        mems = [int(m["mpn_id"]) for m in self.store.memberships(oid, req.rev)]
                        via = next((m for m in mems if m in routed["active"]), None)
                        if via is None and asked:
                            # a link the question names may lead into any enabled MPN (its text is read from the store;
                            # no routing shard or specialist is loaded for it). Disabled MPNs stay out of reach.
                            enabled = enabled_set if enabled_set is not None else set(self._enabled_mpns(req.rev))
                            enabled_set = enabled
                            via = next((m for m in mems if m in enabled), None)
                        if via is None:
                            continue          # a referral into an MPN this request did not activate is not followed
                        n = self.store.get_node(oid, req.rev)
                        if n is None or n["status"] != "live":
                            continue
                        seen.add(oid)
                        r = {"id": oid, "score": 0.0, "gate": 0.5, "via_mpn": via, "home": None, "row": None, "hash": None, "trained": None,
                             "via": "referral", "node": n, "stale": False, "relation": e["type"], "from": c["node"]["title"], "asked": asked}
                        chosen.append(r)
                        nxt.append(r)
                frontier = nxt
        return chosen

    # ------------------------------------------------------------------ procedures
    def _run_procedure(self, req: Request, c: dict, question: str, limits: dict | None = None) -> dict | None:
        n = c["node"]
        th = n["cells"].get("thinking", {})
        impl = th.get("impl")
        p = procedures.REGISTRY.get(impl or "")
        if p is None:
            return None
        spec = "; ".join(f"{k}: {v}" for k, v in p.params.items())
        prompt = (f"Extract the arguments for the procedure “{p.title}” from the question.\nParameters: {spec}\n"
                  f"Constraints: {'; '.join(p.constraints) or 'none'}\nReply with only a JSON object of arguments. "
                  f"Use numbers for numeric parameters. If the question does not ask for this procedure or lacks the values, reply with {{}}.\nQuestion: {question}")
        msgs = [{"role": "system", "content": "You extract function arguments. Reply with JSON only."}, {"role": "user", "content": prompt}]
        out = {"node_id": sid(c["id"]), "title": n["title"], "impl": impl}
        if limits is not None:      # this call has its own budget: its prompt plus 96 new tokens must fit the context
            n_in = len(self.core.ids(self.core.chat_text(msgs)))
            if not tokbudget.fits(n_in, 96, limits):
                out.update(args={}, ok=False, skipped=True, error=f"the question is too long to extract arguments ({n_in} tokens)")
                return out
        text = self._generate_text(msgs, 96, req)
        m = re.search(r"\{.*\}", text, flags=re.S)
        try:
            args = json.loads(m.group(0)) if m else {}
        except Exception:
            args = {}
        out["args"] = args
        if not isinstance(args, dict) or not args:
            out.update(ok=False, error="the question did not supply this procedure's arguments", skipped=True)
            return out
        r = procedures.run(impl, {k: v for k, v in args.items() if k in p.params})
        out.update(ok=r["ok"], result=r.get("result"), unit=r.get("unit", ""), error=r.get("error"))
        return out

    # ------------------------------------------------------------------ generation
    def _check(self, req: Request):
        if req.cancel.is_set():
            raise Cancelled()

    @torch.inference_mode()
    def _generate_text(self, messages: list[dict], max_new: int, req: Request) -> str:
        """A short deterministic generation on the plain core (adapters off), used to extract tool arguments."""
        ids = self.core.ids(self.core.chat_text(messages)).to(self.device)[None]
        from transformers import StoppingCriteria, StoppingCriteriaList

        class Stop(StoppingCriteria):
            def __call__(self, *_a, **_k):
                return req.cancel.is_set()
        with torch.autocast("cuda", dtype=torch.bfloat16), self._adapters([]):
            out = self._generate(input_ids=ids, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                                           pad_token_id=self.core.tok.pad_token_id, stopping_criteria=StoppingCriteriaList([Stop()]))
        return self.core.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    def estimate_mib(self, prompt_tokens: int, out_tokens: int) -> float:
        return BASE_REQUEST_MIB + (prompt_tokens + out_tokens) * KV_MIB_PER_TOKEN

    def admit(self, est_mib: float, budget: dict) -> tuple[bool, str]:
        avail = psutil.virtual_memory().available / 2**30
        headroom = avail - float(budget["reserve_gib"])
        total = float(budget.get("total_gib") or 0)
        if total:
            used = psutil.Process().memory_info().rss / 2**30 + torch.cuda.memory_allocated() / 2**30
            headroom = min(headroom, total - used)
        if est_mib / 1024 > headroom:
            return False, (f"This request needs about {est_mib:.0f} MiB. {avail:.1f} GiB is available, {budget['reserve_gib']:.0f} GiB is reserved"
                           + (f", and the total budget is {total:.0f} GiB" if total else "") + ".")
        return True, ""

    def run(self, messages: list[dict], mode: str = "yaob", rev: int | None = None, request_id: str | None = None, budget: dict | None = None,
            options: dict | None = None):
        """Generator of (event, data) pairs for one request."""
        b = dict(self.get_budget())
        b.update(budget or {})
        rev = self.store.head() if rev is None else int(rev)
        req = Request(request_id, mode, rev, options)
        if self.core is None:
            yield "error", {"error": "The core model is not loaded. Load it from the runtime panel.", "status": 503}
            return
        if mode == "yaob" and self.shared is None:
            yield "error", {"error": "No trained checkpoint is active, so Yaob mode cannot run. Use Retrieval only or Core only, or activate a checkpoint.", "status": 409}
            return
        if len(self.requests) + self.queued >= int(b["concurrent_requests"]) + 4:
            self.counters["refused"] += 1
            yield "error", {"error": f"{len(self.requests) + self.queued} requests are already running or waiting; the budget allows {b['concurrent_requests']} at once.", "status": 429}
            return
        self.queued += 1
        acquired = self.gen_lock.acquire(timeout=120)
        self.queued -= 1
        if not acquired:
            yield "error", {"error": "The runtime stayed busy for two minutes. Try again.", "status": 503}
            return
        self.requests[req.id] = req
        self.counters["requests"] += 1
        torch.cuda.reset_peak_memory_stats()
        try:
            yield from self._run_locked(req, messages, b)
        except Cancelled:
            self.counters["cancelled"] += 1
            yield "done", {"text": "", "cancelled": True, "usage": {}, "timing": {"total_ms": round((time.perf_counter() - req.t0) * 1000, 1)},
                           "memory": self._mem(0)}
        except torch.OutOfMemoryError:
            self.last_error = "out of memory during a request"
            torch.cuda.empty_cache()
            yield "error", {"error": "The machine ran out of memory for this request. Its memory was released. Lower the context or selected-nodes budget, or free memory elsewhere.", "status": 507}
        except Exception as e:      # report, release, keep serving
            self.last_error = f"{type(e).__name__}: {e}"[:300]
            yield "error", {"error": f"The request failed: {type(e).__name__}: {e}"[:400], "status": 500}
        finally:
            for mid in req.pinned:
                self.cache.release(mid)
            self.requests.pop(req.id, None)
            self.gen_lock.release()

    def _mem(self, est: float) -> dict:
        return {"process_rss_mib": round(psutil.Process().memory_info().rss / 2**20, 1),
                "torch_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
                "torch_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1), "estimate_mib": round(est, 1),
                "resident_cache_bytes": self.cache.bytes() if self.cache else 0,
                "system_available_gib": round(psutil.virtual_memory().available / 2**30, 2)}

    def _run_locked(self, req: Request, messages: list[dict], b: dict):
        core = self.core
        c = core.contract
        question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        history = [m for m in messages if m["role"] in ("user", "assistant")][:-1][-6:]
        # The client's own instructions (a chat client may join its model, workspace and assistant prompts and any
        # passages it retrieved into one system message). Kept after Yaob's instructions and never cut.
        client_system = "\n\n".join(m["content"] for m in messages if m["role"] == "system" and (m.get("content") or "").strip())
        try:
            limits = tokbudget.resolve(b, c.max_context, b.get("requested_output_tokens"))
        except ValueError as e:
            yield "error", {"error": str(e), "status": 400}
            return
        yield "start", {"request_id": req.id, "rev": req.rev, "mode": req.mode, "options": req.opts, "core": {"name": c.name, "revision": c.revision},
                        "checkpoint": self.checkpoint["id"] if self.checkpoint and req.mode == "yaob" else None}
        ev: dict = {"mpns": [], "skipped": [], "shards": [], "nodes": [], "procedures": [], "reductions": list(limits["notes"]), "adapters": [], "ms": 0.0}
        rq, used_history = conv.routing_query(question, messages)
        small_talk = conv.is_small_talk(question)
        ev["routing_query"] = {"used_history": used_history, "small_talk": small_talk}
        if small_talk:
            ev["reductions"].append("Small talk: no nodes were read.")
        if b.get("repair_note"):
            ev["reductions"].append(b["repair_note"])
        chosen: list[dict] = []
        text_nodes: list[dict] = []
        adapter_names: list[str] = []
        t_route = time.perf_counter()
        load_ms = 0.0
        if small_talk:
            pass
        elif req.mode == "yaob":
            if req.opts["router"] == "baseline":
                # ablation: ordinary retrieval chooses the nodes; their MPNs' specialists are still applied
                chosen = self._baseline_retrieve(req, rq, int(b["selected_nodes"]))
                names = {int(m["id"]): m["name"] for m in self.store.mpns(req.rev)}
                mids = []
                for ch in chosen:
                    if ch["via_mpn"] not in mids and len(mids) < int(b["max_active_mpns"]):
                        mids.append(ch["via_mpn"])
                n_adapters = int(b.get("max_active_adapters", 0)) if req.opts["adapters"] else 0
                for rank_i, mid in enumerate(mids):
                    shard, e = self.cache.acquire(mid, want_adapter=rank_i < n_adapters)
                    e.update(name=names.get(mid, ""), mpn_id=sid(mid))
                    ev["shards"].append(e)
                    if shard is not None:
                        req.pinned.append(mid)
                ev["mpns"] = [{"id": sid(m), "name": names.get(m, ""), "score": None, "active": True} for m in mids]
            else:
                routed = self.route(req, rq, b)
                ev.update(routed["evidence"])
                self._check(req)
                chosen = self._select(req, routed, b)
            load_ms = sum(s.get("ms", 0.0) for s in ev["shards"])
            # specialists of the MPNs this request activated (and pinned); only those, and only if asked for
            if req.opts["adapters"]:
                names = {int(m["id"]): m["name"] for m in self.store.mpns(req.rev)}
                n_adapters = int(b.get("max_active_adapters", 0))
                for mid in req.pinned:
                    r = self.cache.items.get(mid)
                    if r is not None and r.adapter and len(adapter_names) < n_adapters:
                        # Pilot 2: a specialist trained on old content overrides the corrected text in the prompt.
                        # If a selected node of this MPN changed since the specialist was trained, leave the specialist off.
                        trained = (r.adapter_info or {}).get("nodes", {})
                        stale = [ch["node"]["title"] for ch in chosen
                                 if (ch.get("home") if ch.get("home") is not None else ch.get("via_mpn")) == mid
                                 and str(ch["id"]) in trained and trained[str(ch["id"])] != ch["node"]["content_hash"]]
                        if stale:
                            ev["reductions"].append(f"The {names.get(mid, '')} specialist was not applied: it was trained before “{stale[0]}” was edited, "
                                                    "and a stale specialist overrides the corrected text. Retrain this MPN to use it again.")
                            continue
                        adapter_names.append(r.adapter)
                        info = r.adapter_info or {}
                        ev["adapters"].append({"mpn_id": sid(mid), "name": names.get(mid, ""), "bytes": r.adapter_bytes, "rank": info.get("rank"),
                                               "trainable_params": info.get("trainable_params"), "validated": info.get("validated"),
                                               "trained_at_rev": info.get("trained_at_rev")})
            procs = [ch for ch in chosen if ch["node"]["type"] == "procedure"][:1]
            for ch in procs:
                if ch["gate"] >= 0.5 or ch["via"] == "text match":
                    self._check(req)
                    r = self._run_procedure(req, ch, question, limits)
                    if r and not r.get("skipped"):
                        ev["procedures"].append(r)
            if req.opts["text"]:
                text_nodes = [ch for ch in chosen if ch["node"]["type"] != "procedure"][: int(b["text_nodes"])]
                if b.get("referral_mode", "relation") == "relation":
                    # a link the question asks about, from a node given as text: give its target as text too (up to 2).
                    # Appended last, so the context packer drops these first when space is short.
                    srcs = {ch["node"]["title"] for ch in text_nodes}
                    extra = [ch for ch in chosen if ch.get("asked") and ch.get("from") in srcs and ch not in text_nodes][:2]
                    text_nodes += extra
        elif req.mode == "retrieval":
            chosen = self._baseline_retrieve(req, rq, int(b["text_nodes"]) + 2)
            text_nodes = chosen[: max(int(b["text_nodes"]), 1)]
        ev["ms"] = round((time.perf_counter() - t_route) * 1000, 2)

        # ---- assemble, shrinking to the context budget if needed
        focus = conv.properties_named(question) if b.get("fact_focus", True) else None

        def render(tn: list[dict]) -> str:
            return render_nodes([ch["node"] for ch in tn], focus=focus)

        tool_text = ""
        for p in ev["procedures"]:
            if p.get("ok"):
                tool_text += f"\nTool result ({p['title']}): {json.dumps(p['result'])}" + (f" {p['unit']}" if p.get("unit") else "")
            else:
                tool_text += f"\nTool {p['title']} could not run: {p.get('error')}"
        system = {"yaob": SYS_YAOB, "retrieval": SYS_RETRIEVAL, "core": SYS_CORE}[req.mode]
        if client_system:
            system += "\n\nInstructions from the application:\n" + client_system
        packed = tokbudget.pack(core, system, history, text_nodes, tool_text, question, limits, render, specialist=bool(adapter_names))
        ev["reductions"] += packed["reductions"]
        if packed.get("error"):
            yield "error", packed["error"]
            return
        ids, n_prompt, max_out, text_nodes, breakdown = packed["ids"], packed["n_prompt"], packed["output"], packed["text_nodes"], packed["breakdown"]
        max_ctx = limits["context"]
        est = self.estimate_mib(n_prompt, max_out)
        ok, why = self.admit(est, b)
        if not ok and self.cache:
            freed = self.cache.drop()          # evict unpinned MPN state before giving up
            if freed:
                ev["reductions"].append(f"Evicted {len(freed)} unused MPNs from memory to make room.")
            ok, why = self.admit(est, b)
        if not ok:
            self.counters["refused"] += 1
            yield "error", {"error": why + " Lower the context, output or selected-nodes budget, or wait for other work to finish.", "status": 507}
            return
        text_ids = {ch["id"] for ch in text_nodes}
        names = {int(m["id"]): m["name"] for m in self.store.mpns(req.rev)}
        active_adapter_mpns = {int(a["mpn_id"]) for a in ev["adapters"]}
        for ch in chosen:
            home = ch.get("home") if ch.get("home") is not None else ch.get("via_mpn")
            info = self.ck_nodes.get(ch["id"], {})
            cur = ch["node"]["content_hash"]
            ch["in_adapter"] = home in active_adapter_mpns and bool(info.get("adapter_trained")) and info.get("hash") == cur
            ch["state"] = ("trained" if info.get("key_trained") and info.get("hash") == cur else
                           "edited since training" if info else "not trained") if req.mode == "yaob" else "text"
        ev["nodes"] = [{"id": sid(ch["id"]), "title": ch["node"]["title"], "type": ch["node"]["type"], "mpn_id": sid(ch.get("via_mpn")),
                        "mpn": names.get(ch.get("via_mpn"), ""), "score": round(float(ch["score"]), 3), "gate": round(float(ch["gate"]), 3),
                        "read_as": (["text"] if ch["id"] in text_ids else []) + (["specialist"] if ch.get("in_adapter") else []),
                        "via": ch["via"], "state": ch["state"], **({"relation": ch["relation"], "from": ch["from"]} if ch["via"] == "referral" else {})}
                       for ch in chosen]
        ev["budget"] = {"applied": {k: b[k] for k in ("max_active_mpns", "max_active_adapters", "selected_nodes", "text_nodes", "traversal_depth", "resident_cache_mib")},
                        "tokens": {"context": max_ctx, "output": max_out, "output_requested": limits["output_requested"], "output_cap": limits["output_cap"],
                                   "margin": limits["margin"], "input": breakdown,
                                   "note": "input.total is the exact token count of the chat-template text sent to the core; the parts are measured separately and approximate"},
                        "reductions": ev.pop("reductions")}
        yield "route", ev
        self._check(req)

        # ---- generate, streaming, with exactly this request's adapters active
        x = ids.to(self.device)[None]
        mask = torch.ones_like(x)
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        class Stop(StoppingCriteria):
            def __call__(self, *_a, **_k):
                return req.cancel.is_set()
        streamer = TextIteratorStreamer(core.tok, skip_prompt=True, skip_special_tokens=True)
        result: dict = {}
        use = adapter_names if req.mode == "yaob" else []

        def work():
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16), self.adapter_lock, self._adapters(use):
                    result["out"] = self._generate(input_ids=x, attention_mask=mask, max_new_tokens=max_out, do_sample=False,
                                                   temperature=None, top_p=None, top_k=None, pad_token_id=core.tok.pad_token_id,
                                                   streamer=streamer, stopping_criteria=StoppingCriteriaList([Stop()]))
            except BaseException as e:
                result["error"] = e
                streamer.on_finalized_text("", stream_end=True)
        t_gen = time.perf_counter()
        th = threading.Thread(target=work, daemon=True)
        th.start()
        ttft = None
        parts = []
        # citations are checked as they complete: only supplied, supporting node text may be cited
        cf = conv.CitationFilter([ch["node"]["title"] + " " + render([ch]) for ch in text_nodes])
        for piece in streamer:
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - t_gen
                out_piece = cf.feed(piece)
                if out_piece:
                    parts.append(out_piece)
                    yield "token", {"text": out_piece}
        rest = cf.finish()
        if rest:
            parts.append(rest)
            yield "token", {"text": rest}
        th.join()
        if "error" in result:
            raise result["error"]
        text = "".join(parts).strip()
        n_out = int(result["out"].shape[1]) - x.shape[1] if "out" in result else 0
        total_gen = time.perf_counter() - t_gen
        cancelled = req.cancel.is_set()
        validation = self.validate(text, chosen, text_nodes, ev["procedures"], question)
        if cf.removed:
            validation["checks"].append({"kind": "citations_removed", "ok": True, "removed": cf.removed,
                                         "detail": f"{len(cf.removed)} citation(s) removed: no supplied node text supported them."})
        yield "validation", validation
        ev_tokens = len(core.ids(render(text_nodes))) if text_nodes else 0
        yield "done", {
            "text": text, "cancelled": cancelled,
            "usage": {"prompt_tokens": n_prompt, "text_evidence_tokens": ev_tokens, "completion_tokens": n_out,
                      "adapters_applied": len(use)},
            "timing": {"route_ms": ev["ms"], "load_ms": round(load_ms, 2), "ttft_ms": round((ttft or 0) * 1000, 1),
                       "total_ms": round((time.perf_counter() - req.t0) * 1000, 1),
                       # decode rate after the first token; below 8 tokens the interval is too short to mean anything
                       "tokens_per_s": round((n_out - 1) / max(total_gen - (ttft or 0), 1e-6), 1) if n_out >= 8 else None},
            "memory": self._mem(est)}

    @torch.inference_mode()
    def _baseline_retrieve(self, req: Request, question: str, k: int) -> list[dict]:
        """Ordinary retrieval over all enabled nodes: frozen-encoder cosine and BM25, fused by reciprocal rank."""
        enabled = set(self._enabled_mpns(req.rev))
        score: dict[int, float] = {}
        flat = self._flat()
        if flat["ids"]:
            qv = self.enc.encode([question], query=True).to(self.device)
            top = (flat["mat"] @ qv[0]).topk(min(len(flat["ids"]), 40))
            for rank, i in enumerate(top.indices.tolist()):
                score[flat["ids"][i]] = score.get(flat["ids"][i], 0) + 1 / (30 + rank)
        for rank, r in enumerate(self.store.search_lexical(question, limit=40, any_terms=True)):
            score[int(r["id"])] = score.get(int(r["id"]), 0) + 1 / (30 + rank)
        out = []
        for nid in sorted(score, key=lambda i: -score[i]):
            n = self.store.get_node(nid, req.rev)
            if n is None or n["status"] != "live" or n["type"] == "procedure":
                continue
            ms = [int(m["mpn_id"]) for m in n["memberships"]]
            via = next((m for m in ms if m in enabled), None)
            if via is None:
                continue
            prim = next((int(m["mpn_id"]) for m in n["memberships"] if m["role"] == "primary"), via)
            out.append({"id": nid, "score": score[nid], "gate": 1.0, "via_mpn": via, "home": prim, "via": "retrieval", "node": n})
            if len(out) >= k:
                break
        return out

    # ------------------------------------------------------------------ validation (deterministic)
    def validate(self, answer: str, chosen: list[dict], text_nodes: list[dict], procs: list[dict], question: str) -> dict:
        citations = []
        for i, ch in enumerate(text_nodes, start=1):
            n = ch["node"]
            src = next((s for s in n.get("sources", []) if s["kind"] in ("wikipedia", "user", "authored", "synthetic")), None) or (n.get("sources") or [None])[0]
            citations.append({"n": i, "node_id": sid(ch["id"]), "title": n["title"], "cited": f"[{i}]" in answer,
                              "source": {k: src[k] for k in ("title", "url", "license", "revision", "kind")} if src else None})
        checks = []
        evidence_text = question + " " + " ".join(
            " ".join(s.get("text", "") for s in ch["node"]["cells"].get("knowledge", {}).get("statements", [])) + " " +
            " ".join(str(f.get("value", "")) + " " + str(f.get("as_of", "")) for f in ch["node"]["cells"].get("knowledge", {}).get("facts", []))
            for ch in chosen) + " " + " ".join(json.dumps(p.get("result")) + " " + json.dumps(p.get("args")) for p in procs)
        ev_nums = set(re.findall(r"\d+(?:\.\d+)?", evidence_text.replace(",", "")))
        ans_nums = [x for x in re.findall(r"\d+(?:\.\d+)?", re.sub(r"\[\d+\]", "", answer).replace(",", "")) if len(x.replace(".", "")) >= 3]
        unsupported = sorted({x for x in ans_nums if x not in ev_nums and x.rstrip("0").rstrip(".") not in ev_nums})
        if chosen or procs:
            checks.append({"kind": "numbers_supported", "ok": not unsupported,
                           "detail": "Every number of three or more digits in the answer occurs in a node that was read, a tool result or the question." if not unsupported
                           else "Not found in any node read or tool result: " + ", ".join(unsupported[:8])})
        for p in procs:
            if p.get("ok"):
                vals = re.findall(r"-?\d+(?:\.\d+)?", json.dumps(p["result"]))
                hit = any(v.rstrip("0").rstrip(".") in answer.replace(",", "") or v in answer.replace(",", "") or
                          _round_match(v, answer) for v in vals) if vals else str(p["result"]).lower() in answer.lower()
                checks.append({"kind": "tool_result_used", "node_id": p["node_id"], "title": p["title"], "ok": bool(hit),
                               "detail": "The answer states the computed result." if hit else f"The answer does not state the computed result {json.dumps(p['result'])}."})
        for ch in text_nodes:
            for con in ch["node"]["cells"].get("reasoning", {}).get("constraints", [])[:4]:
                if con.get("kind") == "dated_value":
                    checks.append({"kind": "dated_value", "node_id": sid(ch["id"]), "title": ch["node"]["title"], "ok": True, "detail": con["text"]})
        if text_nodes:
            cited = sum(1 for c in citations if c["cited"])
            checks.append({"kind": "sources_cited", "ok": cited > 0, "detail": f"{cited} of {len(citations)} text nodes are cited in the answer."})
        if not chosen and not procs:
            checks.append({"kind": "no_node_support", "ok": True, "detail": "No node was read for this answer; it comes from the core's pretrained knowledge alone."})
        return {"citations": citations, "checks": checks}

    def cancel(self, request_id: str) -> bool:
        r = self.requests.get(request_id)
        if r is None:
            return False
        r.cancel.set()
        return True


def _round_match(v: str, answer: str) -> bool:
    try:
        f = float(v)
    except ValueError:
        return False
    a = answer.replace(",", "")
    return any(f"{f:.{d}f}" in a for d in (0, 1, 2, 3)) if abs(f) >= 1 else any(f"{f:.{d}f}" in a for d in (2, 3, 4))


def node_text(title: str, statements: list[str]) -> str:
    return title + ". " + " ".join(statements[:4])
