"""Pilot: does a learned node-memory bank work at all with this core, on this machine?

A small network of *canary* nodes: invented entities whose facts cannot be in the
pretrained core. Each node has a few statements. We train only the modular state
(per-node memory and keys, shared bridge and router, MPN adapters) with the core
frozen, then measure:

  core_only          the core with no node access            (should fail: facts are invented)
  text_retrieval     gold node text placed in the prompt     (ordinary retrieval, upper reference)
  memory_gold        gold node read as learned soft tokens    (the learned path, oracle routing)
  memory_routed      learned router picks the nodes           (the full learned path)
  ablations          bridge untrained / per-node state zeroed / rows shuffled to other nodes
  reload             a fresh process-independent load of the saved checkpoint must reproduce memory_gold
  edit               change one fact: stale state, text-derived state, then retrain only that node

Question sets are disjoint by construction:
  train          facts asked with training phrasings
  paraphrase     the same facts asked with phrasings never used in training
  recite_only    facts that occur in a node's statements but were never the answer to a training question
  composition    comparisons needing two nodes at once (never trained)

Usage: python -m mpnstudio.train.pilot --core models/Qwen3.5-4B --encoder models/Qwen3-Embedding-0.6B --out runs/pilot-001
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from ..model.core import MEM_MARK, Core
from ..model.encoder import TextEncoder
from ..model.memory import BankSpec, NodeMemory, router_loss

SYL_A = ["vel", "or", "tam", "kes", "dru", "il", "sar", "nym", "quo", "bra", "zen", "hul", "pir", "ost", "eff", "yar", "gwi", "lom", "tev", "ush"]
SYL_B = ["mar", "eth", "ond", "ix", "ava", "urn", "eck", "oli", "ane", "ugh", "ira", "osk", "emn", "adz", "ilt", "orv", "uma", "eph", "aro", "ist"]
FIRST = ["Ilse", "Corvin", "Maud", "Tobiah", "Senna", "Radek", "Yvaine", "Osric", "Liesl", "Bram", "Nadir", "Petra", "Hollis", "Ximena", "Jorah", "Katri"]
FIELDS = ["tidal cartography", "glacier acoustics", "seed archiving", "bridge metallurgy", "moth migration", "salt chemistry", "clockwork repair", "river law"]
COLOURS = ["slate green", "pale violet", "rust orange", "ink blue", "bone white", "amber", "charcoal", "sea grey"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

KINDS = ["institute", "river", "mineral", "festival"]     # the four pilot MPNs

# attribute -> (statement template, training phrasings, held-out phrasings)
TEMPLATES = {
    "institute": {
        "founded": ("The {name} was founded in {v}.", ["In what year was the {name} founded?", "When was the {name} established?"],
                    ["The {name} dates back to which year?", "What is the founding year of the {name}?"]),
        "town": ("The {name} is located in the town of {v}.", ["In which town is the {name} located?", "Where is the {name}?"],
                 ["Which town hosts the {name}?", "The {name} can be found in what town?"]),
        "founder": ("The {name} was founded by {v}.", ["Who founded the {name}?", "Who established the {name}?"],
                    ["Which person started the {name}?", "The {name} owes its founding to whom?"]),
        "field": ("The {name} studies {v}.", ["What does the {name} study?", "What is the research field of the {name}?"],
                  ["Which subject is the {name} devoted to?", "The {name} specialises in what?"]),
        "staff": ("The {name} employs {v} researchers.", ["How many researchers does the {name} employ?", "How many researchers work at the {name}?"],
                  ["What is the number of researchers at the {name}?", "The {name} has how many researchers?"]),
    },
    "river": {
        "length": ("The {name} is {v} kilometres long.", ["How long is the {name} in kilometres?", "What is the length of the {name} in kilometres?"],
                   ["The {name} runs for how many kilometres?", "How many kilometres does the {name} cover?"]),
        "source": ("The {name} rises on {v}.", ["Where does the {name} rise?", "What is the source of the {name}?"],
                   ["On which mountain does the {name} begin?", "The {name} starts where?"]),
        "mouth": ("The {name} empties into the {v}.", ["Into which sea does the {name} empty?", "Where does the {name} end?"],
                  ["Which sea receives the {name}?", "The {name} flows out into what?"]),
        "bridges": ("The {name} is crossed by {v} bridges.", ["How many bridges cross the {name}?", "How many bridges does the {name} have?"],
                    ["What number of bridges span the {name}?", "The {name} is spanned by how many bridges?"]),
        "explorer": ("The {name} was first charted by {v}.", ["Who first charted the {name}?", "Who mapped the {name} first?"],
                     ["Which person made the first chart of the {name}?", "The first chart of the {name} is owed to whom?"]),
    },
    "mineral": {
        "colour": ("{name} is {v} in colour.", ["What colour is {name}?", "What is the colour of {name}?"],
                   ["{name} appears in which colour?", "Which colour does {name} have?"]),
        "hardness": ("{name} has a hardness of {v} on the Mohs scale.", ["What is the Mohs hardness of {name}?", "How hard is {name} on the Mohs scale?"],
                     ["Where does {name} sit on the Mohs scale?", "{name} rates what on the Mohs scale?"]),
        "discoverer": ("{name} was first described by {v}.", ["Who first described {name}?", "Who discovered {name}?"],
                       ["Which person first described {name}?", "The first description of {name} is by whom?"]),
        "year": ("{name} was first described in {v}.", ["In what year was {name} first described?", "When was {name} first described?"],
                 ["{name} was first reported in which year?", "What year does the first description of {name} date from?"]),
        "locality": ("{name} is found mainly near {v}.", ["Where is {name} mainly found?", "Near which place is {name} found?"],
                     ["What is the main locality of {name}?", "{name} mostly occurs near where?"]),
    },
    "festival": {
        "month": ("The {name} is held every {v}.", ["In which month is the {name} held?", "When is the {name} held?"],
                  ["Which month hosts the {name}?", "The {name} takes place in what month?"]),
        "town": ("The {name} takes place in {v}.", ["In which town does the {name} take place?", "Where is the {name} held?"],
                 ["Which town holds the {name}?", "The {name} happens in what town?"]),
        "days": ("The {name} lasts {v} days.", ["How many days does the {name} last?", "How long is the {name} in days?"],
                 ["What is the length of the {name} in days?", "The {name} runs for how many days?"]),
        "founder": ("The {name} was started by {v}.", ["Who started the {name}?", "Who founded the {name}?"],
                    ["Which person began the {name}?", "The {name} was begun by whom?"]),
        "first": ("The {name} was first held in {v}.", ["In what year was the {name} first held?", "When did the {name} begin?"],
                  ["The first {name} was in which year?", "What year saw the first {name}?"]),
    },
}
NUMERIC = {("institute", "founded"), ("institute", "staff"), ("river", "length"), ("river", "bridges"),
           ("mineral", "hardness"), ("mineral", "year"), ("festival", "days"), ("festival", "first")}
COMPARE = {"institute": ("founded", "Which was founded earlier, the {a} or the {b}?", min),
           "river": ("length", "Which is longer, the {a} or the {b}?", max),
           "festival": ("days", "Which lasts longer, the {a} or the {b}?", max),
           "mineral": ("hardness", "Which is harder, {a} or {b}?", max)}


def make_network(n_per_kind: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    used: set[str] = set()

    def word():
        while True:
            w = (rng.choice(SYL_A) + rng.choice(SYL_B) + (rng.choice(SYL_B) if rng.random() < 0.35 else "")).capitalize()
            if w not in used:
                used.add(w)
                return w

    def person():
        return f"{rng.choice(FIRST)} {word()}"

    nodes = []
    for kind in KINDS:
        for _ in range(n_per_kind):
            w = word()
            if kind == "institute":
                name = f"{w} Institute"
                vals = {"founded": str(rng.randint(1640, 1985)), "town": word(), "founder": person(), "field": rng.choice(FIELDS), "staff": str(rng.randint(12, 940))}
            elif kind == "river":
                name = f"River {w}"
                vals = {"length": str(rng.randint(40, 2900)), "source": f"Mount {word()}", "mouth": f"{word()} Sea", "bridges": str(rng.randint(2, 95)), "explorer": person()}
            elif kind == "mineral":
                name = f"{w}ite"
                vals = {"colour": rng.choice(COLOURS), "hardness": f"{rng.randint(2, 18) / 2:g}", "discoverer": person(), "year": str(rng.randint(1710, 2015)), "locality": f"Lake {word()}"}
            else:
                name = f"{w} Festival"
                vals = {"month": rng.choice(MONTHS), "town": word(), "days": str(rng.randint(2, 31)), "founder": person(), "first": str(rng.randint(1300, 1990))}
            attrs = list(vals)
            recite_only = rng.choice(attrs)       # stated in the node, never the answer to a training question
            nodes.append({"row": len(nodes), "kind": kind, "mpn": KINDS.index(kind), "title": name, "values": vals,
                          "recite_only": recite_only,
                          "statements": [TEMPLATES[kind][a][0].format(name=name, v=vals[a]) for a in attrs]})
    return nodes


def make_questions(nodes: list[dict], seed: int, unseen_frac: float = 0.0) -> dict[str, list[dict]]:
    """`unseen_frac` of the nodes are never trained on at all: no question, no recitation. Their state can only come
    from the shared text-derived path, so they measure what an added-but-untrained node gets."""
    rng = random.Random(seed + 1)
    sets = {"train": [], "paraphrase": [], "recite_only": [], "composition": [], "unseen_nodes": []}
    for n in nodes:
        n["unseen"] = rng.random() < unseen_frac
    for n in nodes:
        if n["unseen"]:
            for a, v in n["values"].items():
                for q in TEMPLATES[n["kind"]][a][2]:
                    sets["unseen_nodes"].append({"rows": [n["row"]], "q": q.format(name=n["title"]), "a": v, "attr": a})
            continue
        for a, v in n["values"].items():
            _, train_q, held_q = TEMPLATES[n["kind"]][a]
            if a == n["recite_only"]:
                for q in held_q:
                    sets["recite_only"].append({"rows": [n["row"]], "q": q.format(name=n["title"]), "a": v, "attr": a})
            else:
                for q in train_q:
                    sets["train"].append({"rows": [n["row"]], "q": q.format(name=n["title"]), "a": v, "attr": a})
                for q in held_q:
                    sets["paraphrase"].append({"rows": [n["row"]], "q": q.format(name=n["title"]), "a": v, "attr": a})
    for kind in KINDS:
        attr, tmpl, pick = COMPARE[kind]
        group = [n for n in nodes if n["kind"] == kind and not n["unseen"]]
        for _ in range(min(len(group), 60)):
            a, b = rng.sample(group, 2)
            va, vb = float(a["values"][attr]), float(b["values"][attr])
            if va == vb:
                continue
            win = a if pick(va, vb) == va else b
            sets["composition"].append({"rows": [a["row"], b["row"]], "q": tmpl.format(a=a["title"], b=b["title"]), "a": win["title"], "attr": attr})
    return sets


SYSTEM_MEM = "You are Yaob. Knowledge nodes are supplied to you as memory. Answer the question from them. Reply with only the answer."
SYSTEM_TXT = "You are Yaob. Knowledge nodes are supplied to you as text. Answer the question from them. Reply with only the answer."
SYSTEM_NONE = "You are Yaob. Answer the question. Reply with only the answer. If you do not know, reply: unknown."
RECITE_Q = "State everything this node says about {title}."


def norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch == " " or ch == ".").strip().rstrip(".")


def is_correct(reply: str, gold: str) -> bool:
    r, g = norm(reply), norm(gold)
    if not g:
        return False
    if g.replace(".", "").isdigit():
        import re
        return g in re.findall(r"\d+(?:\.\d+)?", r)
    return g in r


class Pilot:
    def __init__(self, core: Core, enc: TextEncoder, nodes: list[dict], spec: BankSpec, seed: int):
        self.core, self.enc, self.nodes, self.spec = core, enc, nodes, spec
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        self.mem = NodeMemory(spec, core.embed_norm).to(core.device)
        self.refresh_embeddings()
        a, b = core.split_on_memory([{"role": "system", "content": SYSTEM_MEM}, {"role": "user", "content": MEM_MARK + "\nQuestion: ⁣Q⁣"}])
        self.pre_ids = a
        tail = core.tok.decode(b)
        self.mid_text, self.post_text = tail.split("⁣Q⁣")
        self.q_emb_cache: dict[str, torch.Tensor] = {}

    def refresh_embeddings(self, rows: list[int] | None = None):
        S = self.spec.slots
        if rows is None:
            self.stmt_emb = torch.zeros(len(self.nodes), S, self.spec.d_enc)
            self.node_emb = torch.zeros(len(self.nodes), self.spec.d_enc)
            rows = list(range(len(self.nodes)))
        flat, where = [], []
        for r in rows:
            for j, s in enumerate(self.nodes[r]["statements"][:S]):
                flat.append(s)
                where.append((r, j))
        e = self.enc.encode(flat)
        for r in rows:
            self.stmt_emb[r] = 0
        for (r, j), v in zip(where, e):
            self.stmt_emb[r, j] = v
        self.node_emb[rows] = self.enc.encode([self.nodes[r]["title"] + ". " + " ".join(self.nodes[r]["statements"]) for r in rows])
        self.stmt_emb_d = self.stmt_emb.to(self.core.device)
        self.node_emb_d = self.node_emb.to(self.core.device)

    def q_emb(self, qs: list[str]) -> torch.Tensor:
        miss = [q for q in qs if q not in self.q_emb_cache]
        if miss:
            for q, v in zip(miss, self.enc.encode(miss, query=True)):
                self.q_emb_cache[q] = v
        return torch.stack([self.q_emb_cache[q] for q in qs]).to(self.core.device)

    # ------------------------------------------------------------------ prompt assembly
    def memory_tokens(self, rows: list[int], **kw) -> torch.Tensor:
        """Soft tokens for a list of nodes, each followed by the learned node separator: [len(rows) * (S + 1), H]."""
        r = torch.tensor(rows, device=self.core.device)
        mpn = torch.tensor([self.nodes[i]["mpn"] for i in rows], device=self.core.device)
        emb = kw.pop("stmt_emb", self.stmt_emb_d)[r]
        tok = self.mem.soft_tokens(r, emb, mpn, **kw)                                     # [n, S, H]
        sep = (self.mem.node_sep * self.mem.out_scale).expand(len(rows), 1, -1)
        return torch.cat([tok, sep], dim=1).reshape(-1, tok.shape[-1])

    def assemble(self, items: list[dict], with_answer: bool, **kw):
        pieces, labels = [], []
        dt = torch.bfloat16
        pre = self.core.embed_ids(self.pre_ids).to(dt)
        for it in items:
            mem = self.memory_tokens(it["ctx"], **kw).to(dt)
            q_ids = self.core.ids(self.mid_text + it["q"] + self.post_text)
            p = [pre, mem, self.core.embed_ids(q_ids).to(dt)]
            if with_answer:
                a_ids = self.core.ids(it["a"] + self.core.tok.eos_token)
                p.append(self.core.embed_ids(a_ids).to(dt))
                labels.append(a_ids)
            pieces.append(p)
        return self.core.build_batch(pieces, labels if with_answer else None)

    def context_for(self, rows: list[int], distractors: int) -> list[int]:
        pool = [i for i in range(len(self.nodes)) if i not in rows]
        ctx = list(rows) + self.rng.sample(pool, min(distractors, len(pool)))
        self.rng.shuffle(ctx)
        return ctx

    # ------------------------------------------------------------------ training
    def train(self, train_qs: list[dict], steps: int, batch: int, lr: float, log, node_lr_mult: float = 3.0,
              only_rows: list[int] | None = None, start_step: int = 0, opt_state: dict | None = None):
        mem = self.mem
        per_node = [getattr(mem, n) for n in mem.PER_NODE]
        ids = {id(p) for p in per_node}
        shared = [p for p in mem.parameters() if id(p) not in ids]
        if only_rows is not None:            # "retrain this node": shared tensors frozen, other rows masked
            groups = [{"params": per_node, "lr": lr * node_lr_mult, "weight_decay": 0.0}]
        else:
            groups = [{"params": shared, "lr": lr, "weight_decay": 0.01}, {"params": per_node, "lr": lr * node_lr_mult, "weight_decay": 0.0}]
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.99))
        if opt_state:
            opt.load_state_dict(opt_state)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.1, 1 - s / max(1, steps)), last_epoch=start_step - 1 if start_step else -1)
        recite = [{"rows": [n["row"]], "q": RECITE_Q.format(title=n["title"]), "a": " ".join(n["statements"])} for n in self.nodes if not n.get("unseen")]
        if only_rows is not None:
            keep = set(only_rows)
            train_qs = [q for q in train_qs if set(q["rows"]) <= keep]
            recite = [q for q in recite if set(q["rows"]) <= keep]
            mask = torch.zeros(len(self.nodes), device=self.core.device)
            mask[only_rows] = 1
        mem.train()
        t0 = time.perf_counter()
        tokens = 0
        for step in range(start_step, steps):
            items = []
            for _ in range(batch):
                src = recite if self.rng.random() < 0.3 else train_qs
                it = dict(self.rng.choice(src))
                it["ctx"] = self.context_for(it["rows"], self.rng.choice([0, 1, 2, 3]))
                items.append(it)
            x, mask_att, lab = self.assemble(items, with_answer=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.core.model(inputs_embeds=x, attention_mask=mask_att, labels=lab)
            lm = out.loss.float()
            tokens += int(mask_att.sum())
            # router: every item with a single gold node
            single = [it for it in items if len(it["rows"]) == 1 and it["q"] != RECITE_Q.format(title=self.nodes[it["rows"][0]]["title"])]
            rl, parts = torch.zeros((), device=self.core.device), {}
            if len(single) >= 2:
                pos = torch.tensor([it["rows"][0] for it in single], device=self.core.device)
                negs = []
                for it in single:                # hard negatives: same MPN, different node
                    same = [n["row"] for n in self.nodes if n["mpn"] == self.nodes[it["rows"][0]]["mpn"] and n["row"] != it["rows"][0]]
                    negs.append(self.rng.sample(same, 4) if len(same) >= 4 else [self.rng.choice(same) for _ in range(4)])
                neg = torch.tensor(negs, device=self.core.device)
                tgt = torch.zeros(len(single), self.spec.n_mpn, device=self.core.device)
                tgt[torch.arange(len(single)), torch.tensor([self.nodes[it["rows"][0]]["mpn"] for it in single])] = 1
                rl, parts = router_loss(mem, self.q_emb([it["q"] for it in single]), pos, self.node_emb_d[pos], neg, self.node_emb_d[neg], tgt)
            loss = lm + 0.5 * rl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if only_rows is not None:
                mem.slot_delta.grad *= mask[:, None, None]
                mem.key_delta.grad *= mask[:, None]
                mem.gate_bias.grad *= mask
            torch.nn.utils.clip_grad_norm_(mem.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % 20 == 0 or step == steps - 1:
                dt = time.perf_counter() - t0
                log({"step": step, "lm_loss": round(float(lm), 4), "router": {k: round(v, 4) for k, v in parts.items()},
                     "lr": sched.get_last_lr()[0], "elapsed_s": round(dt, 1), "train_tokens_per_s": round(tokens / max(dt, 1e-6), 0),
                     "torch_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
        mem.eval()
        return opt

    # ------------------------------------------------------------------ evaluation
    @torch.inference_mode()
    def route(self, qs: list[str], top_k: int, use_delta: bool = True):
        """Hierarchical routing: MPN prototypes first, then node keys inside the selected MPNs."""
        qk = self.mem.query_keys(self.q_emb(qs))
        all_rows = torch.arange(len(self.nodes), device=self.core.device)
        keys = self.mem.node_keys(all_rows, self.node_emb_d, use_delta)
        mpn_of = torch.tensor([n["mpn"] for n in self.nodes], device=self.core.device)
        mpn_p = torch.sigmoid(self.mem.mpn_logits(qk))
        active = mpn_p >= 0.5
        active[torch.arange(len(qs)), mpn_p.argmax(-1)] = True       # at least the best MPN
        sim = qk @ keys.T
        sim = sim.masked_fill(~active[:, mpn_of], -2.0)
        top = sim.topk(top_k, dim=-1)
        gate = self.mem.gate(top.values, top.indices)
        return top.indices.tolist(), top.values.tolist(), gate.tolist(), active.tolist()

    @torch.inference_mode()
    def answer(self, qs: list[dict], mode: str, batch: int = 16, top_k: int = 4, **kw) -> list[str]:
        out = []
        for b in range(0, len(qs), batch):
            chunk = qs[b:b + batch]
            if mode in ("core_only", "text_retrieval"):
                msgs = []
                for it in chunk:
                    if mode == "core_only":
                        msgs.append([{"role": "system", "content": SYSTEM_NONE}, {"role": "user", "content": "Question: " + it["q"]}])
                    else:
                        text = "\n".join(f"- {self.nodes[r]['title']}: " + " ".join(self.nodes[r]["statements"]) for r in it["ctx"])
                        msgs.append([{"role": "system", "content": SYSTEM_TXT}, {"role": "user", "content": "Knowledge nodes:\n" + text + "\nQuestion: " + it["q"]}])
                pieces = [[self.core.embed_ids(self.core.ids(self.core.chat_text(m))).to(torch.bfloat16)] for m in msgs]
                x, mask, _ = self.core.build_batch(pieces)
            else:
                x, mask, _ = self.assemble(chunk, with_answer=False, **kw)
            out += self.core.generate_from_embeds(x, mask, max_new_tokens=24)
        return out

    def evaluate(self, sets: dict[str, list[dict]], mode: str, limit: int = 0, seed: int = 7, **kw) -> dict:
        rng = random.Random(seed)
        res = {}
        for name, qs in sets.items():
            qs = list(qs)
            rng.shuffle(qs)
            if limit:
                qs = qs[:limit]
            items = []
            routed_hit = 0
            if mode == "memory_routed":
                idx, _, _, _ = self.route([q["q"] for q in qs], top_k=4)
                for q, top in zip(qs, idx):
                    items.append(dict(q, ctx=top))
                    routed_hit += all(r in top for r in q["rows"])
            else:
                save = self.rng
                self.rng = rng
                for q in qs:
                    items.append(dict(q, ctx=self.context_for(q["rows"], 4 - len(q["rows"]))))
                self.rng = save
            if kw.get("shuffle_rows"):
                n = len(self.nodes)
                items = [dict(it, ctx=[(r + n // 2) % n for r in it["ctx"]]) for it in items]
            kws = {k: v for k, v in kw.items() if k != "shuffle_rows"}
            replies = self.answer(items, "memory" if mode.startswith("memory") else mode, **kws)
            ok = [is_correct(r, it["a"]) for r, it in zip(replies, items)]
            res[name] = {"n": len(items), "correct": sum(ok), "accuracy": round(sum(ok) / max(1, len(items)), 4),
                         "examples": [{"q": it["q"], "gold": it["a"], "reply": r} for r, it in list(zip(replies, items))[:3]]}
            if mode == "memory_routed":
                res[name]["router_recall_at_4"] = round(routed_hit / max(1, len(items)), 4)
        return res


def save_checkpoint(path: Path, mem: NodeMemory, extra: dict, opt=None):
    """Atomic: write to a temp name, then rename. The previous checkpoint is kept as `.prev`."""
    from safetensors.torch import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    save_file({k: v.detach().cpu().contiguous() for k, v in mem.state_dict().items()}, str(tmp),
              metadata={"spec": json.dumps(mem.spec.to_json()), "extra": json.dumps(extra)})
    if path.exists():
        path.replace(path.with_suffix(".prev.safetensors"))
    tmp.replace(path)
    if opt is not None:
        torch.save(opt.state_dict(), path.with_suffix(".opt.pt"))


def load_checkpoint(path: Path, device: str, embed_norm: float) -> tuple[NodeMemory, dict]:
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(str(path), "pt") as f:
        meta = f.metadata()
    spec = BankSpec(**json.loads(meta["spec"]))
    mem = NodeMemory(spec, embed_norm)
    mem.load_state_dict(load_file(str(path)))
    return mem.to(device).eval(), json.loads(meta["extra"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nodes-per-kind", type=int, default=24)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--slots", type=int, default=6)
    ap.add_argument("--d-value", type=int, default=384)
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--eval-limit", type=int, default=120)
    ap.add_argument("--unseen-frac", type=float, default=0.0)
    ap.add_argument("--skip-edit", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / "train.jsonl").open("a")

    def log(d):
        logf.write(json.dumps(d) + "\n")
        logf.flush()
        print(json.dumps(d), flush=True)

    nodes = make_network(args.nodes_per_kind, args.seed)
    sets = make_questions(nodes, args.seed, args.unseen_frac)
    (out / "network.json").write_text(json.dumps({"nodes": nodes, "question_counts": {k: len(v) for k, v in sets.items()}}, indent=1))
    torch.cuda.reset_peak_memory_stats()
    core = Core.load(args.core)
    enc = TextEncoder(args.encoder)
    spec = BankSpec(n_rows=len(nodes), n_mpn=len(KINDS), slots=args.slots, d_enc=enc.dim, d_value=args.d_value, d_core=core.hidden_size)
    pilot = Pilot(core, enc, nodes, spec, args.seed)
    report = {"core": core.contract.to_json(), "encoder_params": enc.params, "spec": spec.to_json(),
              "learned_params": pilot.mem.param_counts(), "nodes": len(nodes), "seed": args.seed,
              "question_counts": {k: len(v) for k, v in sets.items()}, "args": vars(args)}
    evalsets = {k: sets[k] for k in ("paraphrase", "recite_only", "composition", "train", "unseen_nodes") if sets[k]}
    report["unseen_nodes"] = sum(1 for n in nodes if n["unseen"])
    report["exposures_per_trained_node"] = round(args.steps * args.batch / max(1, len(nodes) - report["unseen_nodes"]), 1)
    L = args.eval_limit

    print("== before training", flush=True)
    report["core_only"] = pilot.evaluate(evalsets, "core_only", L)
    report["text_retrieval"] = pilot.evaluate(evalsets, "text_retrieval", L)
    report["memory_untrained"] = pilot.evaluate(evalsets, "memory_gold", L)
    before = {k: v.detach().clone() for k, v in pilot.mem.state_dict().items()}

    print("== training", flush=True)
    t0 = time.perf_counter()
    opt = pilot.train(sets["train"], args.steps, args.batch, args.lr, log)
    report["train_wall_s"] = round(time.perf_counter() - t0, 1)
    report["train_peak_torch_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    after = pilot.mem.state_dict()
    report["parameter_change_l2"] = {k: round(float((after[k].float() - before[k].float()).norm()), 4) for k in after}
    ck = out / "memory.safetensors"
    save_checkpoint(ck, pilot.mem, {"step": args.steps, "seed": args.seed, "core": core.contract.name}, opt)

    print("== after training", flush=True)
    report["memory_gold"] = pilot.evaluate(evalsets, "memory_gold", L)
    report["memory_routed"] = pilot.evaluate(evalsets, "memory_routed", L)
    report["ablation_no_node_state"] = pilot.evaluate(evalsets, "memory_gold", L, use_delta=False)
    report["ablation_no_mpn_adapter"] = pilot.evaluate(evalsets, "memory_gold", L, use_adapter=False)
    report["ablation_rows_shuffled"] = pilot.evaluate(evalsets, "memory_gold", L, shuffle_rows=True)

    print("== reload checkpoint into a new module", flush=True)
    trained = pilot.mem
    pilot.mem, extra = load_checkpoint(ck, core.device, core.embed_norm)
    report["reload"] = {"extra": extra, "result": pilot.evaluate(evalsets, "memory_gold", L)}
    report["reload"]["identical_to_trained"] = report["reload"]["result"]["paraphrase"]["correct"] == report["memory_gold"]["paraphrase"]["correct"]
    pilot.mem = NodeMemory(spec, core.embed_norm).to(core.device).eval()      # fresh, untrained module: the checkpoint is what matters
    report["fresh_module_without_checkpoint"] = pilot.evaluate({"paraphrase": sets["paraphrase"]}, "memory_gold", L)
    pilot.mem = trained

    if args.skip_edit:
        (out / "report.json").write_text(json.dumps(report, indent=1))
        summary = {k: {s: v[s]["accuracy"] for s in v if isinstance(v[s], dict) and "accuracy" in v[s]} for k, v in report.items()
                   if isinstance(v, dict) and "paraphrase" in v}
        print(json.dumps({"summary": summary, "router_recall_at_4": {k: v.get("router_recall_at_4") for k, v in report["memory_routed"].items()},
                          "learned_params": report["learned_params"]["total"], "train_wall_s": report["train_wall_s"],
                          "exposures_per_trained_node": report["exposures_per_trained_node"]}, indent=1))
        return
    print("== edit one node", flush=True)
    target = nodes[0]
    attr = next(a for a in target["values"] if a != target["recite_only"] and (target["kind"], a) in NUMERIC)
    old, new = target["values"][attr], str(int(float(target["values"][attr])) + 137)
    q = {"rows": [0], "q": TEMPLATES[target["kind"]][attr][2][0].format(name=target["title"]), "a": new, "ctx": [0]}
    edit = {"node": target["title"], "attribute": attr, "old": old, "new": new, "question": q["q"]}
    edit["before_edit_reply"] = pilot.answer([q], "memory")[0]
    target["values"][attr] = new
    idx = list(target["values"]).index(attr)
    target["statements"][idx] = TEMPLATES[target["kind"]][attr][0].format(name=target["title"], v=new)
    pilot.refresh_embeddings([0])
    edit["stale_state_reply"] = pilot.answer([q], "memory")[0]                       # new text embedding + old learned residual
    edit["text_derived_state_reply"] = pilot.answer([q], "memory", use_delta=False)[0]  # learned residual ignored
    new_train = [{"rows": [0], "q": t.format(name=target["title"]), "a": target["values"][a]}
                 for a in target["values"] if a != target["recite_only"] for t in TEMPLATES[target["kind"]][a][1]]
    t0 = time.perf_counter()
    pilot.train(new_train, 80, 8, args.lr, log, only_rows=[0])
    edit["retrain_node_s"] = round(time.perf_counter() - t0, 1)
    edit["after_node_retrain_reply"] = pilot.answer([q], "memory")[0]
    edit["after_node_retrain_correct"] = is_correct(edit["after_node_retrain_reply"], new)
    edit["other_nodes_after_retrain"] = pilot.evaluate({"paraphrase": [x for x in sets["paraphrase"] if x["rows"] != [0]]}, "memory_gold", 60)
    report["edit"] = edit

    (out / "report.json").write_text(json.dumps(report, indent=1))
    summary = {k: {s: v[s]["accuracy"] for s in v if isinstance(v[s], dict) and "accuracy" in v[s]} for k, v in report.items()
               if isinstance(v, dict) and "paraphrase" in v}
    print(json.dumps({"summary": summary, "reload_identical": report["reload"]["identical_to_trained"], "edit": edit,
                      "learned_params": report["learned_params"]["total"], "train_wall_s": report["train_wall_s"],
                      "train_peak_torch_gib": report["train_peak_torch_gib"]}, indent=1))


if __name__ == "__main__":
    main()
