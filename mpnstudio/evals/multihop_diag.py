"""Two-fact diagnosis on a development set (not the held-out suite).

For each two-hop question "What is the <property> of the <relation> of <A>?" (A links to B, B holds the property):
  1. represented  - A's node text names B, and B's stored facts hold the gold value
  2. rendered     - the gold value survives node rendering (render_node shows at most 14 facts and 6 statements)
  3. routed       - the learned router selected A, B and both (joint recall), and which were given as text
  4. answered     - the default configuration (routing + node text, no specialist) answered correctly
  5. oracle       - the plain core with exactly A's and B's rendered text supplied answers correctly
  6. oracle_full  - the same with B's gold fact guaranteed visible (B rendered with the gold fact first)
  7. missing_b    - with only A's text supplied, the reply abstains or is correct (otherwise it asserts something unsupported)

Tasks are built with the held-out generator under a different seed, and every task that shares a node with the held-out
suite's two-hop tasks is dropped, so nothing here is tuned on locked test items.

  python -m mpnstudio.evals.multihop_diag --heldout ~/mpn-studio/eval/heldout_v1.json --out ~/mpn-studio/runs/multihop-dev-001
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from ..model.prompts import render_node, render_nodes
from ..store.db import Store
from ..train import data as D
from .heldout import abstains, build_suite, norm, post_stream, score_task

DEV_SEED = 777001


def run_mode(runtime: str, messages: list[dict], mode: str, budget: dict, options: dict | None = None) -> dict:
    out = {"reply": "", "route": None, "error": None}
    for ev, d in post_stream(runtime + "/run", {"messages": messages, "mode": mode, "budget": budget, "options": options or {}}):
        if ev == "route":
            out["route"] = d
        elif ev == "done":
            out.update(reply=d["text"], usage=d["usage"])
        elif ev == "error":
            out["error"] = d.get("error")
    return out


def oracle_messages(nodes: list[dict], question: str) -> list[dict]:
    return [{"role": "system", "content": "Answer from these knowledge nodes. Cite them as [1], [2]. If they do not cover the question, say so.\n\n"
                                          "Knowledge nodes:\n" + render_nodes(nodes)},
            {"role": "user", "content": question}]


def gold_first(n: dict, prop: str) -> dict:
    """A copy of node n whose facts list the gold property first, so rendering cannot cut it."""
    n = json.loads(json.dumps(n))
    k = n["cells"].setdefault("knowledge", {})
    facts = k.get("facts", [])
    k["facts"] = [f for f in facts if f.get("property") == prop] + [f for f in facts if f.get("property") != prop]
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db")
    ap.add_argument("--heldout", required=True, help="the locked suite; its two-hop nodes are excluded")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--runtime", default="http://127.0.0.1:8841")
    ap.add_argument("--budget", default="{}", help='JSON overrides for the routed request, e.g. {"referral_mode": "relation"}')
    ap.add_argument("--skip-oracle", action="store_true", help="only the routed request (for comparing routing settings)")
    a = ap.parse_args()
    from ..train.jobs import db_path
    store = Store(a.db or db_path())
    rev = store.head()
    held = json.loads(Path(a.heldout).read_text())
    locked = {u for t in held["tasks"] if t["family"] == "two_hop" for u in t["expect_nodes"]}
    zero = {k: 0 for k in ("fact_paraphrase", "canary_paraphrase", "canary_statement", "procedure", "unsupported", "adversarial")}
    suite = build_suite(store, seed=DEV_SEED, sizes=dict(zero, two_hop=a.n + 40))
    tasks = [t for t in suite["tasks"] if t["family"] == "two_hop" and not (set(t["expect_nodes"]) & locked)][: a.n]
    nodes = {n["uid"]: n for n in D.iter_nodes(store, rev)}
    from ..api.app import get_budget
    budget = dict(get_budget(store), output_tokens=128, max_active_adapters=0)
    routed_budget = dict(budget, **json.loads(a.budget))
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    for i, t in enumerate(tasks):
        A, B = (nodes[u] for u in t["expect_nodes"])
        gold = t["gold"]
        b_facts = [str(f["value"]) for f in B["cells"].get("knowledge", {}).get("facts", []) if f.get("property") == t["prop"]]
        rA, rB = render_node(A, 1), render_node(B, 2)
        row = {"id": t["id"], "question": t["question"], "gold": gold, "prop": t["prop"], "via": t["via"], "A": A["title"], "B": B["title"],
               "represented": {"A_names_B": norm(B["title"]) in norm(rA) or any(f.get("node") == B["uid"] for f in A["cells"].get("knowledge", {}).get("facts", [])),
                               "B_holds_gold": any(norm(g) in norm(v) for g in gold for v in b_facts)},
               "rendered": {"A_text_names_B": norm(B["title"]) in norm(rA), "B_text_has_gold": any(norm(g) in norm(rB) for g in gold),
                            "B_fact_count": len(B["cells"].get("knowledge", {}).get("facts", []))}}
        r = run_mode(a.runtime, [{"role": "user", "content": t["question"]}], "yaob", routed_budget)
        route = r["route"] or {}
        sel = {n["title"]: n for n in route.get("nodes", [])}
        row["routed"] = {"A_selected": A["title"] in sel, "B_selected": B["title"] in sel,
                         "A_text": "text" in sel.get(A["title"], {}).get("read_as", []), "B_text": "text" in sel.get(B["title"], {}).get("read_as", []),
                         "B_via": sel.get(B["title"], {}).get("via"), "mpns": [m["name"] for m in route.get("mpns", []) if m.get("active")],
                         "selected": len(sel), "reductions": (route.get("budget") or {}).get("reductions")}
        row["answered"] = {"ok": score_task(t, r["reply"])[0] if r["reply"] else False, "reply": r["reply"][:300], "error": r["error"]}
        if a.skip_oracle:
            row["oracle"] = row["oracle_full"] = {"ok": False, "reply": "", "skipped": True}
            row["missing_b"] = {"abstained": False, "correct": False, "reply": "", "skipped": True}
        else:
            o = run_mode(a.runtime, oracle_messages([A, B], t["question"]), "core", budget)
            row["oracle"] = {"ok": score_task(t, o["reply"])[0] if o["reply"] else False, "reply": o["reply"][:300]}
            of = run_mode(a.runtime, oracle_messages([A, gold_first(B, t["prop"])], t["question"]), "core", budget)
            row["oracle_full"] = {"ok": score_task(t, of["reply"])[0] if of["reply"] else False, "reply": of["reply"][:300]}
            m = run_mode(a.runtime, oracle_messages([A], t["question"]), "core", budget)
            row["missing_b"] = {"abstained": abstains(m["reply"]), "correct": score_task(t, m["reply"])[0] if m["reply"] else False, "reply": m["reply"][:300]}
        rows.append(row)
        if i % 10 == 0:
            print(f"{i + 1}/{len(tasks)}  {time.perf_counter() - t0:.0f} s", flush=True)
    (out_dir / "rows.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    n = len(rows)

    def pct(k):
        return {"n": k, "pct": round(100 * k / n, 1) if n else 0.0}
    c = Counter()
    for x in rows:
        c["A_names_B"] += x["represented"]["A_names_B"]
        c["B_holds_gold"] += x["represented"]["B_holds_gold"]
        c["B_text_has_gold"] += x["rendered"]["B_text_has_gold"]
        c["A_selected"] += x["routed"]["A_selected"]
        c["B_selected"] += x["routed"]["B_selected"]
        c["both_selected"] += x["routed"]["A_selected"] and x["routed"]["B_selected"]
        c["both_text"] += x["routed"]["A_text"] and x["routed"]["B_text"]
        c["answered"] += x["answered"]["ok"]
        c["oracle"] += x["oracle"]["ok"]
        c["oracle_full"] += x["oracle_full"]["ok"]
        c["missing_b_safe"] += x["missing_b"]["abstained"] or x["missing_b"]["correct"]
        c["answered_given_both_text"] += x["answered"]["ok"] and x["routed"]["A_text"] and x["routed"]["B_text"]
    # first failing stage per task, in pipeline order
    stage = Counter()
    for x in rows:
        if x["answered"]["ok"]:
            stage["correct"] += 1
        elif not (x["represented"]["A_names_B"] and x["represented"]["B_holds_gold"]):
            stage["1 fact not represented"] += 1
        elif not x["rendered"]["B_text_has_gold"]:
            stage["2 gold fact cut by rendering"] += 1
        elif not (x["routed"]["A_selected"] and x["routed"]["B_selected"]):
            stage["3 routing missed a node"] += 1
        elif not (x["routed"]["A_text"] and x["routed"]["B_text"]):
            stage["4 packing left a node's text out"] += 1
        elif not x["oracle"]["ok"]:
            stage["5 core failed to combine even with both texts"] += 1
        else:
            stage["6 other (both texts given and the oracle succeeds)"] += 1
    rep = {"set": "development two-hop set, seed %d, held-out two-hop nodes excluded" % DEV_SEED, "rev": rev, "n": n,
           "budget": {k: routed_budget.get(k) for k in ("context_tokens", "output_tokens", "max_active_adapters", "max_active_mpns", "selected_nodes", "text_nodes",
                                                     "traversal_depth", "referral_mode", "fact_focus")},
           "rates": {k: pct(v) for k, v in c.items()}, "first_failure": dict(sorted(stage.items())), "wall_s": round(time.perf_counter() - t0, 1)}
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
