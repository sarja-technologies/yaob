"""Pilot 2: MPN specialists as low-rank adapters on the frozen core.

Pilot 1 showed that one learned memory token per statement does not carry invented facts through the
frozen core at an affordable training budget. This pilot tests the other candidate: each MPN owns a
LoRA adapter trained only on that MPN's nodes. The core's own weights stay frozen and unmerged, an
adapter is a separate file, and the runtime applies only the adapters of the MPNs it routes to.

Same canary network as pilot 1 (invented facts the core cannot know), four MPNs. Measured:

  no_adapter            core alone                                   (must fail)
  own_adapter           the question's MPN adapter is active          (closed book: no node text in the prompt)
  wrong_adapter         a different MPN's adapter is active           (must fail: knowledge is in the specific specialist)
  all_adapters          all four active at once                       (do specialists interfere when composed?)
  own_adapter_with_text adapter + node text                           (does the adapter harm ordinary grounded answering?)
  text_only             node text, no adapter                         (reference)
  reload                adapters saved, deleted, loaded from disk     (must equal own_adapter)
  edit                  change one fact; retrain that MPN's adapter on the edited node with replay

Also: a short diagnostic of pilot 1's idea with far more capacity (free soft tokens, four per statement,
no bridge), to record whether the failure was capacity or approach.

Usage: python -m mpnstudio.train.pilot_adapters --core models/Qwen3-4B-Instruct-2507 --out runs/pilot-adapters-001
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch

from ..model.core import Core
from .pilot import KINDS, NUMERIC, TEMPLATES, is_correct, make_network, make_questions

SYS = "You are Yaob, a model by Fylume. Answer from the knowledge nodes you are given or have learned. Cite text nodes as [1], [2]. If nothing covers the question, say so. Be concise."
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def prompt(core: Core, q: str, text: str | None = None) -> str:
    user = (f"Knowledge nodes:\n{text}\n\n" if text else "") + "Question: " + q
    return core.chat_text([{"role": "system", "content": SYS}, {"role": "user", "content": user}])


def node_text(n: dict, i: int = 1) -> str:
    return f"[{i}] {n['title']}\n" + " ".join(n["statements"])


def examples_for(nodes: list[dict]) -> list[dict]:
    """Closed-book training examples for a set of nodes: QA on trained attributes, each statement, and the whole node."""
    out = []
    for n in nodes:
        for a, v in n["values"].items():
            if a == n["recite_only"]:
                continue
            for q in TEMPLATES[n["kind"]][a][1]:
                out.append({"row": n["row"], "q": q.format(name=n["title"]), "a": v})
        for j, s in enumerate(n["statements"]):
            out.append({"row": n["row"], "q": f"What is point {j + 1} of the node on {n['title']}?", "a": s})
        out.append({"row": n["row"], "q": f"State what the node on {n['title']} says.", "a": " ".join(n["statements"])})
    return out


class AdapterPilot:
    def __init__(self, core: Core, nodes: list[dict], rank: int, seed: int):
        from peft import LoraConfig, get_peft_model
        self.core, self.nodes, self.rank = core, nodes, rank
        self.rng = random.Random(seed)
        self.cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=TARGETS, bias="none", task_type="CAUSAL_LM")
        self.pm = get_peft_model(core.model, self.cfg, adapter_name=KINDS[0])
        for k in KINDS[1:]:
            self.pm.add_adapter(k, self.cfg)
        self.params_per_adapter = sum(p.numel() for n, p in self.pm.named_parameters() if f".{KINDS[0]}." in n and "lora_" in n)

    def batch(self, items: list[dict], with_answer: bool):
        tok = self.core.tok
        seqs, labels = [], []
        for it in items:
            p = self.core.ids(prompt(self.core, it["q"], it.get("text")))
            if with_answer:
                a = torch.cat([self.core.ids(it["a"])[:96], torch.tensor([tok.eos_token_id])])
                seqs.append(torch.cat([p, a]))
                labels.append(torch.cat([torch.full((len(p),), -100), a]))
            else:
                seqs.append(p)
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), tok.pad_token_id)
        mask = torch.zeros(len(seqs), T, dtype=torch.long)
        lab = torch.full((len(seqs), T), -100)
        for i, s in enumerate(seqs):
            if with_answer:                     # right padding for training
                ids[i, : len(s)] = s
                mask[i, : len(s)] = 1
                lab[i, : len(s)] = labels[i]
            else:                               # left padding for generation
                ids[i, T - len(s):] = s
                mask[i, T - len(s):] = 1
        d = self.core.device
        return ids.to(d), mask.to(d), lab.to(d)

    def train_adapter(self, name: str, examples: list[dict], steps: int, batch: int, lr: float, log, open_book: float = 0.25, replay: list[dict] | None = None):
        self.pm.set_adapter(name)
        params = [p for n, p in self.pm.named_parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.99))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.05, 0.5 * (1 + math.cos(math.pi * s / max(1, steps)))))
        self.pm.train()
        t0 = time.perf_counter()
        tokens = 0
        for step in range(steps):
            items = []
            for _ in range(batch):
                src = replay if (replay and self.rng.random() < 0.5) else examples
                it = dict(self.rng.choice(src))
                if self.rng.random() < open_book:        # some examples carry the node text, so the adapter also learns to use context
                    n = self.nodes[it["row"]]
                    others = self.rng.sample([m for m in self.nodes if m["kind"] == n["kind"] and m["row"] != n["row"]], 1)
                    ctx = [n] + others
                    self.rng.shuffle(ctx)
                    it["text"] = "\n\n".join(node_text(m, i + 1) for i, m in enumerate(ctx))
                items.append(it)
            ids, mask, lab = self.batch(items, True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.pm(input_ids=ids, attention_mask=mask, labels=lab)
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tokens += int(mask.sum())
            if step % 25 == 0 or step == steps - 1:
                dt = time.perf_counter() - t0
                log({"adapter": name, "step": step, "loss": round(float(out.loss), 4), "elapsed_s": round(dt, 1),
                     "train_tokens_per_s": round(tokens / max(dt, 1e-6)), "torch_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
        self.pm.eval()
        return {"steps": steps, "wall_s": round(time.perf_counter() - t0, 1), "examples": len(examples)}

    @torch.inference_mode()
    def answer(self, items: list[dict], batch: int = 16) -> list[str]:
        out = []
        for b in range(0, len(items), batch):
            ids, mask, _ = self.batch(items[b:b + batch], False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gen = self.pm.generate(input_ids=ids, attention_mask=mask, max_new_tokens=28, do_sample=False, temperature=None, top_p=None, top_k=None,
                                       pad_token_id=self.core.tok.pad_token_id)
            out += [t.strip() for t in self.core.tok.batch_decode(gen[:, ids.shape[1]:], skip_special_tokens=True)]
        return out

    def evaluate(self, sets: dict, mode: str, limit: int, seed: int = 7) -> dict:
        """mode: none | own | wrong | all, optionally suffixed '+text'."""
        rng = random.Random(seed)
        which, with_text = (mode.split("+") + [""])[:2]
        res = {}
        for name, qs in sets.items():
            qs = list(qs)
            rng.shuffle(qs)
            qs = qs[:limit]
            ok_n, ex = 0, []
            by_kind: dict[str, list[dict]] = {}
            for q in qs:
                by_kind.setdefault(self.nodes[q["rows"][0]]["kind"], []).append(q)
            for kind, group in by_kind.items():
                items = []
                for q in group:
                    it = {"q": q["q"]}
                    if with_text:
                        it["text"] = "\n\n".join(node_text(self.nodes[r], i + 1) for i, r in enumerate(q["rows"]))
                    items.append(it)
                if which == "none":
                    with self.pm.disable_adapter():
                        replies = self.answer(items)
                else:
                    if which == "own":
                        self.pm.base_model.set_adapter([kind])
                    elif which == "wrong":
                        self.pm.base_model.set_adapter([KINDS[(KINDS.index(kind) + 1) % len(KINDS)]])
                    else:
                        self.pm.base_model.set_adapter(list(KINDS))
                    replies = self.answer(items)
                for q, r in zip(group, replies):
                    c = is_correct(r, q["a"])
                    ok_n += c
                    if len(ex) < 3:
                        ex.append({"q": q["q"], "gold": q["a"], "reply": r})
            res[name] = {"n": len(qs), "correct": ok_n, "accuracy": round(ok_n / max(1, len(qs)), 4), "examples": ex}
        return res


def soft_token_diagnostic(core: Core, nodes: list[dict], sets: dict, steps: int, batch: int, per_statement: int, lr: float, log, seed: int) -> dict:
    """Pilot 1's idea with much more capacity and no bridge: free soft tokens per node, trained directly."""
    rng = random.Random(seed)
    dev, H = core.device, core.hidden_size
    S = len(nodes[0]["statements"]) * per_statement
    tok = torch.nn.Parameter(torch.randn(len(nodes), S, H, device=dev) * (core.embed_norm / math.sqrt(H)))
    opt = torch.optim.Adam([tok], lr=lr)
    ex = examples_for(nodes)
    pre = core.ids(core.chat_text([{"role": "system", "content": SYS}, {"role": "user", "content": "⁣M⁣"}]).split("⁣M⁣")[0])
    post_text = core.chat_text([{"role": "system", "content": SYS}, {"role": "user", "content": "⁣M⁣"}]).split("⁣M⁣")[1]

    def build(items, with_answer):
        pieces, labels = [], []
        for it in items:
            q = core.ids("\nQuestion: " + it["q"] + post_text)
            p = [core.embed_ids(pre).to(torch.bfloat16), tok[it["row"]].to(torch.bfloat16), core.embed_ids(q).to(torch.bfloat16)]
            if with_answer:
                a = torch.cat([core.ids(it["a"])[:96], torch.tensor([core.tok.eos_token_id])])
                p.append(core.embed_ids(a).to(torch.bfloat16))
                labels.append(a)
            pieces.append(p)
        return core.build_batch(pieces, labels if with_answer else None)
    t0 = time.perf_counter()
    for step in range(steps):
        x, mask, lab = build([rng.choice(ex) for _ in range(batch)], True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = core.model(inputs_embeds=x, attention_mask=mask, labels=lab).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            log({"soft_tokens": True, "step": step, "loss": round(float(loss), 4), "elapsed_s": round(time.perf_counter() - t0, 1)})
    res = {}
    for name in ("train", "paraphrase", "recite_only"):
        qs = list(sets[name])
        random.Random(7).shuffle(qs)
        qs = qs[:96]
        ok = 0
        with torch.inference_mode():
            for b in range(0, len(qs), 16):
                x, mask, _ = build([{"row": q["rows"][0], "q": q["q"]} for q in qs[b:b + 16]], False)
                for q, r in zip(qs[b:b + 16], core.generate_from_embeds(x, mask, 28)):
                    ok += is_correct(r, q["a"])
        res[name] = round(ok / len(qs), 4)
    return {"tokens_per_node": S, "params_per_node": S * H, "steps": steps, "exposures_per_node": round(steps * batch / len(nodes), 1), "lr": lr,
            "wall_s": round(time.perf_counter() - t0, 1), "accuracy": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nodes-per-kind", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--eval-limit", type=int, default=96)
    ap.add_argument("--soft-steps", type=int, default=400)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / "train.jsonl").open("a")

    def log(d):
        logf.write(json.dumps(d) + "\n")
        logf.flush()
        print(json.dumps(d), flush=True)

    nodes = make_network(args.nodes_per_kind, args.seed)
    sets = make_questions(nodes, args.seed)
    evalsets = {k: sets[k] for k in ("train", "paraphrase", "recite_only", "composition")}
    core = Core.load(args.core)
    report = {"core": core.contract.to_json(), "nodes": len(nodes), "seed": args.seed, "args": vars(args), "question_counts": {k: len(v) for k, v in sets.items()}}

    print("== soft-token diagnostic (pilot 1's idea, 4 free tokens per statement, no bridge)", flush=True)
    torch.cuda.reset_peak_memory_stats()
    report["soft_token_diagnostic"] = soft_token_diagnostic(core, nodes, sets, args.soft_steps, args.batch, 4, 2e-2, log, args.seed)
    print(json.dumps(report["soft_token_diagnostic"]), flush=True)

    print("== adapters", flush=True)
    torch.cuda.reset_peak_memory_stats()
    pilot = AdapterPilot(core, nodes, args.rank, args.seed)
    report["adapter"] = {"rank": args.rank, "alpha": 2 * args.rank, "targets": TARGETS, "params_per_adapter": pilot.params_per_adapter, "adapters": len(KINDS)}
    L = args.eval_limit
    report["no_adapter"] = pilot.evaluate(evalsets, "none", L)
    report["text_only"] = pilot.evaluate(evalsets, "none+text", L)
    before = {n: p.detach().clone() for n, p in pilot.pm.named_parameters() if "lora_B" in n and f".{KINDS[0]}." in n}
    report["training"] = {}
    for kind in KINDS:
        ex = examples_for([n for n in nodes if n["kind"] == kind])
        steps = math.ceil(len(ex) * args.epochs / args.batch)
        report["training"][kind] = pilot.train_adapter(kind, ex, steps, args.batch, args.lr, log)
    after = {n: p for n, p in pilot.pm.named_parameters() if n in before}
    report["parameter_change_l2_lora_B_first_adapter"] = round(float(sum((after[n].float() - before[n].float()).pow(2).sum() for n in before).sqrt()), 4)
    report["train_peak_torch_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    print("== evaluation", flush=True)
    for name, mode in (("own_adapter", "own"), ("wrong_adapter", "wrong"), ("all_adapters", "all"), ("own_adapter_with_text", "own+text")):
        report[name] = pilot.evaluate(evalsets, mode, L)
    with pilot.pm.disable_adapter():
        pass
    report["no_adapter_after_training"] = pilot.evaluate({"paraphrase": sets["paraphrase"]}, "none", L)       # the core itself is unchanged

    print("== save, delete, reload from disk", flush=True)
    ad = out / "adapters"
    if ad.exists():
        shutil.rmtree(ad)
    pilot.pm.save_pretrained(str(ad), selected_adapters=list(KINDS))
    sizes = {k: sum(f.stat().st_size for f in (ad / k).rglob("*") if f.is_file()) for k in KINDS}
    for k in KINDS[1:]:
        pilot.pm.delete_adapter(k)
    t0 = time.perf_counter()
    for k in KINDS[1:]:
        pilot.pm.load_adapter(str(ad / k), adapter_name=k, is_trainable=False)
    load_s = (time.perf_counter() - t0) / (len(KINDS) - 1)
    report["reload"] = {"bytes_per_adapter": sizes, "load_s_per_adapter": round(load_s, 3), "result": pilot.evaluate(evalsets, "own", L)}
    report["reload"]["identical_to_trained"] = all(report["reload"]["result"][s]["correct"] == report["own_adapter"][s]["correct"] for s in evalsets)

    print("== edit one node, retrain its MPN's adapter", flush=True)
    target = nodes[0]
    kind = target["kind"]
    attr = next(a for a in target["values"] if a != target["recite_only"] and (kind, a) in NUMERIC)
    old, new = target["values"][attr], str(int(float(target["values"][attr])) + 137)
    q = {"rows": [0], "q": TEMPLATES[kind][attr][2][0].format(name=target["title"]), "a": new}
    pilot.pm.base_model.set_adapter([kind])
    edit = {"node": target["title"], "attribute": attr, "old": old, "new": new, "question": q["q"], "before_edit_reply": pilot.answer([{"q": q["q"]}])[0]}
    target["values"][attr] = new
    target["statements"][list(target["values"]).index(attr)] = TEMPLATES[kind][attr][0].format(name=target["title"], v=new)
    edit["stale_adapter_reply_closed_book"] = pilot.answer([{"q": q["q"]}])[0]
    edit["stale_adapter_reply_with_new_text"] = pilot.answer([{"q": q["q"], "text": node_text(target)}])[0]
    same = [n for n in nodes if n["kind"] == kind]
    t0 = time.perf_counter()
    pilot.train_adapter(kind, examples_for([target]), 80, 8, args.lr, log, replay=examples_for([n for n in same if n["row"] != 0]))
    edit["retrain_s"] = round(time.perf_counter() - t0, 1)
    pilot.pm.base_model.set_adapter([kind])
    edit["after_retrain_reply"] = pilot.answer([{"q": q["q"]}])[0]
    edit["after_retrain_correct"] = is_correct(edit["after_retrain_reply"], new)
    edit["other_nodes_of_that_mpn_after_retrain"] = pilot.evaluate({"paraphrase": [x for x in sets["paraphrase"] if x["rows"] != [0] and nodes[x["rows"][0]]["kind"] == kind]}, "own", 64)["paraphrase"]["accuracy"]
    report["edit"] = edit

    (out / "report.json").write_text(json.dumps(report, indent=1))
    summary = {k: {s: v[s]["accuracy"] for s in v} for k, v in report.items() if isinstance(v, dict) and isinstance(v.get("paraphrase"), dict) and "accuracy" in v["paraphrase"]}
    print(json.dumps({"summary": summary, "soft_tokens": report["soft_token_diagnostic"], "reload_identical": report["reload"]["identical_to_trained"],
                      "adapter": report["adapter"], "reload": {k: report["reload"][k] for k in ("bytes_per_adapter", "load_s_per_adapter")},
                      "training": report["training"], "edit": edit, "train_peak_torch_gib": report["train_peak_torch_gib"]}, indent=1))


if __name__ == "__main__":
    main()
