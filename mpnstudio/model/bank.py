"""On-disk form of trained modular state, and its selective loading.

A checkpoint directory holds:

  router.safetensors           the whole learned router (per-node keys and gates, prototypes, projections)
  nodes.json                   node id -> {row, hash, key_trained, adapter_trained}
  manifest.json                run records and parameter counts
  shards/shared.safetensors    the router's shared tensors. Always resident: every request needs them.
  shards/mpn_<id>.safetensors  per MPN: its nodes' routing keys and gate biases, and routing entries for nodes
                               shared into it from other MPNs
  adapters/<id>/               per MPN, optional: the MPN's specialist, a LoRA adapter on the frozen core
                               (adapter_model.safetensors, adapter_config.json, adapter.json)

`MpnResidency` keeps MPN state in memory under a byte budget with least-recently-used eviction. One
entry is one MPN: its routing shard and, if it has one, its adapter. An MPN used by a request in flight
is pinned. Nothing about an MPN the router did not select is read from disk for that request.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .router import Router, RouterSpec

SHARED_KEYS = ("enc_to_key.weight", "enc_to_key.bias", "query_to_key.weight", "query_to_key.bias", "mpn_proto", "logit_scale", "gate_scale", "gate_center")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def save_tensors(path: Path, tensors: dict, metadata: dict | None = None) -> None:
    """Atomic write: temp file then rename."""
    from safetensors.torch import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(tmp), metadata={k: json.dumps(v) for k, v in (metadata or {}).items()})
    tmp.replace(path)


def load_tensors(path: Path) -> tuple[dict, dict]:
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(str(path), "pt") as f:
        meta = {k: json.loads(v) for k, v in (f.metadata() or {}).items()}
    return load_file(str(path)), meta


def package_router(ck_dir: Path, router: Router, node_rows: dict[int, int], node_hash: dict[int, str], primary_mpn: dict[int, int],
                   shared_mpns: dict[int, list[int]], mpn_index: dict[int, int], node_emb: torch.Tensor) -> dict:
    """Write shared + per-MPN routing shards. Tensors are indexed by bank row."""
    shards = ck_dir / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    router = router.cpu()
    sd = {k: v.detach() for k, v in router.state_dict().items()}
    save_tensors(shards / "shared.safetensors", {k: sd[k] for k in SHARED_KEYS},
                 {"spec": router.spec.to_json(), "mpn_index": {str(k): v for k, v in mpn_index.items()}})
    with torch.no_grad():
        keys_all = router.node_keys(torch.arange(router.spec.n_rows), node_emb.float())       # final normalised keys, residual included
    by_mpn: dict[int, list[int]] = {}
    for nid, m in primary_mpn.items():
        if nid in node_rows:
            by_mpn.setdefault(m, []).append(nid)
    members: dict[int, list[int]] = {}
    for nid, ms in shared_mpns.items():
        if nid in node_rows:
            for m in ms:
                members.setdefault(m, []).append(nid)
    files = {}
    for m in mpn_index:
        ids = sorted(by_mpn.get(m, []))
        rows = torch.tensor([node_rows[i] for i in ids], dtype=torch.long)
        mem_ids = sorted(members.get(m, []))
        mrows = torch.tensor([node_rows[i] for i in mem_ids], dtype=torch.long)
        t = {"node_ids": torch.tensor(ids, dtype=torch.long), "keys": keys_all[rows].half(), "gate_bias": sd["gate_bias"][rows].float(),
             "member_ids": torch.tensor(mem_ids, dtype=torch.long),
             "member_keys": keys_all[mrows].half() if mem_ids else torch.zeros(0, router.spec.d_key).half(),
             "member_gate_bias": sd["gate_bias"][mrows].float() if mem_ids else torch.zeros(0),
             "member_home": torch.tensor([primary_mpn[i] for i in mem_ids], dtype=torch.long)}
        path = shards / f"mpn_{m}.safetensors"
        save_tensors(path, t, {"mpn_id": m, "hashes": [node_hash[i] for i in ids], "member_hashes": [node_hash[i] for i in mem_ids]})
        files[str(m)] = {"file": path.name, "nodes": len(ids), "shared_members": len(mem_ids), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    sp = shards / "shared.safetensors"
    return {"shards": files, "shared": {"file": sp.name, "bytes": sp.stat().st_size, "sha256": sha256_file(sp)}}


@dataclass
class Resident:
    mpn_id: int
    tensors: dict
    hashes: list[str]
    member_hashes: list[str]
    shard_bytes: int
    adapter: str | None            # name under which the adapter is loaded in the core, if any
    adapter_bytes: int
    adapter_info: dict
    load_s: float
    last_used: float = field(default_factory=time.time)
    pins: int = 0
    row_of: dict = field(default_factory=dict)

    def __post_init__(self):
        self.row_of = {int(n): i for i, n in enumerate(self.tensors["node_ids"].tolist())}

    @property
    def bytes(self) -> int:
        return self.shard_bytes + self.adapter_bytes


class MpnResidency:
    """Byte-budgeted, pinned, least-recently-used residency of MPN state (routing shard + specialist adapter)."""

    def __init__(self, ck_dir: Path, limit_bytes: int, max_resident: int = 64,
                 load_adapter: Callable[[int, Path], str] | None = None, unload_adapter: Callable[[str], None] | None = None):
        self.dir, self.limit, self.max_resident = Path(ck_dir), int(limit_bytes), int(max_resident)
        self.items: "OrderedDict[int, Resident]" = OrderedDict()
        self.lock = threading.RLock()
        self._load_adapter, self._unload_adapter = load_adapter, unload_adapter
        self.stats = {"loads": 0, "hits": 0, "evictions": 0, "load_s": 0.0, "refused": 0}

    def bytes(self) -> int:
        return sum(s.bytes for s in self.items.values())

    def size_on_disk(self, mpn_id: int) -> tuple[int, int]:
        shard = self.dir / "shards" / f"mpn_{mpn_id}.safetensors"
        return (shard.stat().st_size if shard.exists() else 0), dir_bytes(self.dir / "adapters" / str(mpn_id))

    def _evict_one(self, mid: int) -> None:
        r = self.items.pop(mid)
        if r.adapter and self._unload_adapter:
            self._unload_adapter(r.adapter)
        self.stats["evictions"] += 1

    def _evict_to_fit(self, incoming: int) -> list[int]:
        evicted = []
        for mid in list(self.items):
            if self.bytes() + incoming <= self.limit and len(self.items) + (1 if incoming else 0) <= self.max_resident:
                break
            if self.items[mid].pins == 0:
                self._evict_one(mid)
                evicted.append(mid)
        return evicted

    def set_limits(self, limit_bytes: int, max_resident: int) -> list[int]:
        with self.lock:
            self.limit, self.max_resident = int(limit_bytes), int(max_resident)
            return self._evict_to_fit(0)

    def acquire(self, mpn_id: int, want_adapter: bool = True) -> tuple[Resident | None, dict]:
        """Get an MPN's state, loading it if needed, and pin it. The caller must `release`. Returns (resident, event)."""
        with self.lock:
            r = self.items.get(mpn_id)
            if r is not None and want_adapter and r.adapter is None and self._load_adapter:
                # resident for routing only; this request also wants its specialist
                adir = self.dir / "adapters" / str(mpn_id)
                ab = dir_bytes(adir) if (adir / "adapter_config.json").exists() else 0
                if ab:
                    r.pins += 1                      # keep it while making room
                    self._evict_to_fit(ab)
                    r.pins -= 1
                    if self.bytes() + ab <= self.limit:
                        t0 = time.perf_counter()
                        r.adapter = self._load_adapter(mpn_id, adir)
                        r.adapter_bytes = ab
                        r.adapter_info = json.loads((adir / "adapter.json").read_text()) if (adir / "adapter.json").exists() else {}
                        r.pins += 1
                        self.items.move_to_end(mpn_id)
                        self.stats["loads"] += 1
                        return r, {"mpn_id": mpn_id, "state": "adapter loaded now", "bytes": r.bytes, "ms": round((time.perf_counter() - t0) * 1000, 2), "adapter": True}
            if r is not None:
                self.items.move_to_end(mpn_id)
                r.pins += 1
                r.last_used = time.time()
                self.stats["hits"] += 1
                return r, {"mpn_id": mpn_id, "state": "already resident", "bytes": r.bytes, "ms": 0.0, "adapter": bool(r.adapter)}
            shard = self.dir / "shards" / f"mpn_{mpn_id}.safetensors"
            if not shard.exists():
                return None, {"mpn_id": mpn_id, "state": "no trained state", "bytes": 0, "ms": 0.0, "adapter": False}
            sb, ab = self.size_on_disk(mpn_id)
            ab = ab if (want_adapter and self._load_adapter) else 0
            evicted = self._evict_to_fit(sb + ab)
            if self.bytes() + sb + ab > self.limit or len(self.items) + 1 > self.max_resident:
                self.stats["refused"] += 1
                return None, {"mpn_id": mpn_id, "state": "over budget", "bytes": sb + ab, "ms": 0.0, "evicted": evicted, "adapter": False}
            t0 = time.perf_counter()
            tensors, meta = load_tensors(shard)
            name, info = None, {}
            adir = self.dir / "adapters" / str(mpn_id)
            if ab and (adir / "adapter_config.json").exists():
                name = self._load_adapter(mpn_id, adir)
                info = json.loads((adir / "adapter.json").read_text()) if (adir / "adapter.json").exists() else {}
            dt = time.perf_counter() - t0
            # account the same figure that admission checked (file size, slightly above the tensor bytes), so the budget holds
            nbytes = max(sb, sum(t.numel() * t.element_size() for t in tensors.values()))
            r = Resident(mpn_id, tensors, meta.get("hashes", []), meta.get("member_hashes", []), nbytes, name, ab if name else 0, info, dt)
            r.pins = 1
            self.items[mpn_id] = r
            self.stats["loads"] += 1
            self.stats["load_s"] += dt
            return r, {"mpn_id": mpn_id, "state": "loaded now", "bytes": r.bytes, "ms": round(dt * 1000, 2), "evicted": evicted, "adapter": bool(name)}

    def release(self, mpn_id: int) -> None:
        with self.lock:
            r = self.items.get(mpn_id)
            if r is not None and r.pins > 0:
                r.pins -= 1

    def drop(self, mpn_id: int | None = None) -> list[int]:
        """Evict one MPN or every unpinned MPN."""
        with self.lock:
            out = []
            for mid in list(self.items):
                if (mpn_id is None or mid == mpn_id) and self.items[mid].pins == 0:
                    self._evict_one(mid)
                    self.stats["evictions"] -= 1
                    out.append(mid)
            return out

    def residency(self) -> list[dict]:
        with self.lock:
            return [{"mpn_id": str(m), "resident": True, "bytes": r.bytes, "shard_bytes": r.shard_bytes, "adapter_bytes": r.adapter_bytes,
                     "adapter": bool(r.adapter), "pins": r.pins, "last_used": r.last_used, "load_ms": round(r.load_s * 1000, 2)}
                    for m, r in self.items.items()]


class SharedRouter:
    """The always-resident shared router tensors (a Router with zero per-node rows)."""

    def __init__(self, ck_dir: Path, device: str):
        tensors, meta = load_tensors(Path(ck_dir) / "shards" / "shared.safetensors")
        self.spec = RouterSpec(**meta["spec"])
        self.mpn_index = {int(k): v for k, v in meta["mpn_index"].items()}
        self.router = Router(RouterSpec(**{**meta["spec"], "n_rows": 1}))
        missing = self.router.load_state_dict(tensors, strict=False)
        assert set(missing.missing_keys) <= {"key_delta", "gate_bias"}, missing
        self.router.to(device).eval()
        self.bytes = sum(t.numel() * t.element_size() for t in tensors.values())
        self.params = sum(t.numel() for t in tensors.values())


def to_f16_blob(t: torch.Tensor) -> bytes:
    return t.detach().cpu().half().numpy().tobytes()


def from_f16_blob(b: bytes, shape: tuple) -> torch.Tensor:
    return torch.from_numpy(np.frombuffer(b, dtype=np.float16).reshape(shape).copy())
