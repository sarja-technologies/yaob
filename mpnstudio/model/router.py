"""Learned hierarchical router: which MPNs, then which nodes, a question should reach.

All tensors are learned by gradient descent on a routing objective. The text encoder that produces the
input embeddings is frozen and is not part of the learned state.

  N  rows (one per node; row index is stable and comes from the store's `bank_rows`)
  M  MPNs     De  encoder dimension (1024)     Dk  key dimension (256)

Per-node tensors (cell-node parameters):
  key_delta   [N, Dk]   Thinking cell: residual on the node's routing key
  gate_bias   [N]       Reasoning cell: offset of the node's relevance gate

Shared tensors:
  enc_to_key    Linear(De -> Dk)   node side
  query_to_key  Linear(De -> Dk)   query side
  mpn_proto     [M, Dk]            one prototype per MPN: the first, coarse routing stage
  logit_scale, gate_scale, gate_center   scalars

A node's key is normalize(enc_to_key(e_node) + key_delta). With key_delta = 0 a node still has a key
derived from its text, which is what a new or edited node uses until the router is retrained.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RouterSpec:
    n_rows: int
    n_mpn: int
    d_enc: int = 1024
    d_key: int = 256

    def to_json(self) -> dict:
        return asdict(self)


class Router(nn.Module):
    PER_NODE = ("key_delta", "gate_bias")
    PER_MPN = ("mpn_proto",)

    def __init__(self, spec: RouterSpec):
        super().__init__()
        self.spec = spec
        self.key_delta = nn.Parameter(torch.zeros(spec.n_rows, spec.d_key))
        self.gate_bias = nn.Parameter(torch.zeros(spec.n_rows))
        self.enc_to_key = nn.Linear(spec.d_enc, spec.d_key)
        self.query_to_key = nn.Linear(spec.d_enc, spec.d_key)
        self.mpn_proto = nn.Parameter(torch.randn(spec.n_mpn, spec.d_key) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0)))
        self.gate_scale = nn.Parameter(torch.tensor(8.0))
        self.gate_center = nn.Parameter(torch.tensor(0.35))

    def param_counts(self) -> dict:
        s = self.spec
        per_node = sum(getattr(self, n).numel() for n in self.PER_NODE)
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "per_node_total": per_node, "per_node_each": s.d_key + 1, "per_mpn_total": self.mpn_proto.numel(),
                "per_mpn_each": s.d_key, "shared": total - per_node - self.mpn_proto.numel(),
                "by_tensor": {n: p.numel() for n, p in self.named_parameters()}}

    def node_keys(self, rows: torch.Tensor, node_emb: torch.Tensor, use_delta: bool = True) -> torch.Tensor:
        k = self.enc_to_key(node_emb)
        if use_delta:
            k = k + self.key_delta[rows]
        return F.normalize(k, dim=-1)

    def query_keys(self, q_emb: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_to_key(q_emb), dim=-1)

    def mpn_logits(self, qk: torch.Tensor) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0) * qk @ F.normalize(self.mpn_proto, dim=-1).T

    def gate(self, sim: torch.Tensor, rows: torch.Tensor | None = None, bias: torch.Tensor | None = None) -> torch.Tensor:
        b = self.gate_bias[rows] if rows is not None else (bias if bias is not None else 0.0)
        return torch.sigmoid(self.gate_scale * (sim - self.gate_center) + b)


def router_loss(r: Router, q_emb, pos_rows, pos_node_emb, neg_rows, neg_node_emb, mpn_targets):
    """InfoNCE over in-batch positives and hard negatives, a multi-label MPN loss, and a gate loss.

    q_emb [B, De]; pos_rows [B]; pos_node_emb [B, De]; neg_rows [B, K]; neg_node_emb [B, K, De]; mpn_targets [B, M] in {0, 1}.
    """
    qk = r.query_keys(q_emb)
    pk = r.node_keys(pos_rows, pos_node_emb)
    B, K = neg_rows.shape
    nk = r.node_keys(neg_rows.reshape(-1), neg_node_emb.reshape(B * K, -1)).reshape(B, K, -1)
    scale = r.logit_scale.exp().clamp(max=100.0)
    sim_pos = qk @ pk.T
    same = pos_rows[:, None] == pos_rows[None, :]
    sim_pos_m = sim_pos.masked_fill(same & ~torch.eye(B, dtype=torch.bool, device=q_emb.device), -1e4)
    sim_neg = torch.einsum("bd,bkd->bk", qk, nk)
    sim_neg_m = sim_neg.masked_fill(neg_rows == pos_rows[:, None], -1e4)
    nce = F.cross_entropy(torch.cat([sim_pos_m, sim_neg_m], dim=1) * scale, torch.arange(B, device=q_emb.device))
    mpn = F.binary_cross_entropy_with_logits(r.mpn_logits(qk), mpn_targets.float())
    g_pos = r.gate(sim_pos.diagonal(), pos_rows)
    valid = (neg_rows != pos_rows[:, None]).reshape(-1)
    g_neg = r.gate(sim_neg.reshape(-1), neg_rows.reshape(-1))[valid]
    gate = F.binary_cross_entropy(g_pos, torch.ones_like(g_pos)) + (F.binary_cross_entropy(g_neg, torch.zeros_like(g_neg)) if valid.any() else 0.0)
    return nce + 0.5 * mpn + 0.25 * gate, {"nce": float(nce.detach()), "mpn": float(mpn.detach()), "gate": float(gate.detach() if torch.is_tensor(gate) else gate)}
