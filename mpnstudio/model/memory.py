"""Learned modular state of a Yaob network: node memory bank, bridge, MPN adapters and router.

Every tensor here is learned by gradient descent with the core frozen. Shapes use:
  N  rows in the bank (one per node; row index comes from the store's `bank_rows`)
  S  memory slots per node
  De dimension of the frozen text encoder (1024 for Qwen3-Embedding-0.6B)
  Dv dimension of a stored slot value
  Dk dimension of a routing key
  H  hidden size of the core
  M  number of MPNs

Per-node tensors (the cell-node parameters, CNP):
  slot_delta  [N, S, Dv]  Knowledge-cell residual memory, one vector per statement slot
  key_delta   [N, Dk]     Thinking-cell routing key residual
  gate_bias   [N]         Reasoning-cell relevance threshold offset

Shared tensors:
  enc_to_value  Linear(De -> Dv)        turns a frozen statement embedding into a slot value
  slot_pos      [S, Dv]                 slot position embedding
  bridge        MLP(Dv -> H)            projects slot values into the core's input-embedding space
  enc_to_key    Linear(De -> Dk)        node side of the router
  query_to_key  Linear(De -> Dk)        query side of the router
  mpn_proto     [M, Dk]                 one routing prototype per MPN (hierarchical first stage)

MPN-scoped tensors (loaded only when the MPN is active):
  mpn_u, mpn_v  [M, Dv, R]              low-rank adapter applied to slot values of nodes read through that MPN

A slot value is  v = enc_to_value(e_stmt) + slot_pos + slot_delta ;  the soft token fed to the core is
bridge(v + U_m (V_m^T v)).  With slot_delta = 0 a node still has usable state computed from its text, which is
what a newly added, not yet trained node uses.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BankSpec:
    n_rows: int
    n_mpn: int
    slots: int = 6
    d_enc: int = 1024
    d_value: int = 384
    d_key: int = 256
    d_core: int = 2560
    adapter_rank: int = 8
    bridge_hidden: int = 1536

    def to_json(self) -> dict:
        return asdict(self)


class NodeMemory(nn.Module):
    def __init__(self, spec: BankSpec, embed_norm: float = 1.0):
        super().__init__()
        self.spec = spec
        s = spec
        # per-node (CNP)
        self.slot_delta = nn.Parameter(torch.zeros(s.n_rows, s.slots, s.d_value))
        self.key_delta = nn.Parameter(torch.zeros(s.n_rows, s.d_key))
        self.gate_bias = nn.Parameter(torch.zeros(s.n_rows))
        # shared
        self.enc_to_value = nn.Linear(s.d_enc, s.d_value)
        self.slot_pos = nn.Parameter(torch.randn(s.slots, s.d_value) * 0.02)
        self.bridge = nn.Sequential(nn.LayerNorm(s.d_value), nn.Linear(s.d_value, s.bridge_hidden), nn.GELU(),
                                    nn.Linear(s.bridge_hidden, s.d_core), nn.LayerNorm(s.d_core))
        self.out_scale = nn.Parameter(torch.tensor(float(embed_norm) / math.sqrt(s.d_core)))
        self.node_sep = nn.Parameter(torch.randn(s.d_core) * 0.02)
        self.enc_to_key = nn.Linear(s.d_enc, s.d_key)
        self.query_to_key = nn.Linear(s.d_enc, s.d_key)
        self.mpn_proto = nn.Parameter(torch.randn(s.n_mpn, s.d_key) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0)))
        self.gate_scale = nn.Parameter(torch.tensor(8.0))
        self.gate_center = nn.Parameter(torch.tensor(0.35))
        # MPN-scoped
        self.mpn_u = nn.Parameter(torch.zeros(s.n_mpn, s.d_value, s.adapter_rank))
        self.mpn_v = nn.Parameter(torch.randn(s.n_mpn, s.d_value, s.adapter_rank) * 0.02)

    # ------------------------------------------------------------------ parameter groups
    PER_NODE = ("slot_delta", "key_delta", "gate_bias")
    PER_MPN = ("mpn_u", "mpn_v", "mpn_proto")

    def param_counts(self) -> dict:
        per_node = sum(getattr(self, n).numel() for n in self.PER_NODE)
        per_mpn = sum(getattr(self, n).numel() for n in self.PER_MPN)
        total = sum(p.numel() for p in self.parameters())
        s = self.spec
        return {"total": total, "per_node_total": per_node, "per_node_each": s.slots * s.d_value + s.d_key + 1,
                "per_mpn_total": per_mpn, "per_mpn_each": 2 * s.d_value * s.adapter_rank + s.d_key,
                "shared": total - per_node - per_mpn,
                "by_tensor": {n: p.numel() for n, p in self.named_parameters()}}

    # ------------------------------------------------------------------ knowledge-cell memory -> soft tokens
    def slot_values(self, rows: torch.Tensor, stmt_emb: torch.Tensor, use_delta: bool = True) -> torch.Tensor:
        """rows [B]; stmt_emb [B, S, De] frozen statement embeddings (zero rows for empty slots) -> [B, S, Dv]."""
        v = self.enc_to_value(stmt_emb) + self.slot_pos
        if use_delta:
            v = v + self.slot_delta[rows]
        return v

    def soft_tokens(self, rows: torch.Tensor, stmt_emb: torch.Tensor, mpn: torch.Tensor | None = None,
                    use_delta: bool = True, use_adapter: bool = True) -> torch.Tensor:
        """-> [B, S, H] soft tokens for B nodes, each read through MPN index mpn[b] (or no adapter)."""
        v = self.slot_values(rows, stmt_emb, use_delta)
        if mpn is not None and use_adapter:
            u, w = self.mpn_u[mpn], self.mpn_v[mpn]                    # [B, Dv, R]
            v = v + torch.einsum("bdr,bsr->bsd", u, torch.einsum("bsd,bdr->bsr", v, w))
        return self.bridge(v) * self.out_scale

    # ------------------------------------------------------------------ router
    def node_keys(self, rows: torch.Tensor, node_emb: torch.Tensor, use_delta: bool = True) -> torch.Tensor:
        k = self.enc_to_key(node_emb)
        if use_delta:
            k = k + self.key_delta[rows]
        return F.normalize(k, dim=-1)

    def query_keys(self, q_emb: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_to_key(q_emb), dim=-1)

    def mpn_logits(self, qk: torch.Tensor) -> torch.Tensor:
        return self.logit_scale.exp() * qk @ F.normalize(self.mpn_proto, dim=-1).T

    def gate(self, sim: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Reasoning-cell relevance: probability that a node with similarity `sim` should be read."""
        return torch.sigmoid(self.gate_scale * (sim - self.gate_center) + self.gate_bias[rows])


def router_loss(mem: NodeMemory, q_emb, pos_rows, pos_node_emb, neg_rows, neg_node_emb, mpn_targets):
    """InfoNCE over in-batch positives plus hard negatives, a multi-label MPN loss and a gate loss.

    q_emb [B, De]; pos_rows [B]; pos_node_emb [B, De]; neg_rows [B, K]; neg_node_emb [B, K, De]; mpn_targets [B, M] in {0,1}.
    """
    qk = mem.query_keys(q_emb)
    pk = mem.node_keys(pos_rows, pos_node_emb)
    B, K = neg_rows.shape
    nk = mem.node_keys(neg_rows.reshape(-1), neg_node_emb.reshape(B * K, -1)).reshape(B, K, -1)
    scale = mem.logit_scale.exp().clamp(max=100.0)
    sim_pos = qk @ pk.T                                        # [B, B]; diagonal = positives
    same = pos_rows[:, None] == pos_rows[None, :]
    sim_pos = sim_pos.masked_fill(same & ~torch.eye(B, dtype=torch.bool, device=q_emb.device), -1e4)
    sim_neg = torch.einsum("bd,bkd->bk", qk, nk)               # [B, K]
    sim_neg = sim_neg.masked_fill(neg_rows == pos_rows[:, None], -1e4)
    logits = torch.cat([sim_pos, sim_neg], dim=1) * scale
    nce = F.cross_entropy(logits, torch.arange(B, device=q_emb.device))
    mpn = F.binary_cross_entropy_with_logits(mem.mpn_logits(qk), mpn_targets.float())
    g_pos = mem.gate(sim_pos.diagonal(), pos_rows)
    g_neg = mem.gate(sim_neg.reshape(-1), neg_rows.reshape(-1))
    valid = (neg_rows != pos_rows[:, None]).reshape(-1)
    gate = F.binary_cross_entropy(g_pos, torch.ones_like(g_pos)) + F.binary_cross_entropy(g_neg[valid], torch.zeros_like(g_neg[valid]))
    return nce + 0.5 * mpn + 0.25 * gate, {"nce": float(nce.detach()), "mpn": float(mpn.detach()), "gate": float(gate.detach())}
