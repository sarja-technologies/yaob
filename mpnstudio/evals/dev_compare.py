"""Compare routing settings on development questions (never the locked held-out tasks).

  python -m mpnstudio.evals.dev_compare --heldout <suite> --out <dir> --family fact_paraphrase --n 80 \
      --variants '{"plain": {"referral_mode": "plain"}, "relation": {"referral_mode": "relation"}}'

Questions come from the held-out generator under a development seed; any task sharing a node with a held-out task is
dropped. Each variant is the default routed request (routing + node text, no specialist) with the given overrides."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..store.db import Store
from .heldout import build_suite, score_task
from .multihop_diag import DEV_SEED, run_mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db")
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", default="fact_paraphrase")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--runtime", default="http://127.0.0.1:8841")
    a = ap.parse_args()
    from ..api.app import get_budget
    from ..train.jobs import db_path
    store = Store(a.db or db_path())
    held = json.loads(Path(a.heldout).read_text())
    locked = {u for t in held["tasks"] for u in t["expect_nodes"]}
    fams = ("fact_paraphrase", "canary_paraphrase", "canary_statement", "two_hop", "procedure", "unsupported", "adversarial")
    suite = build_suite(store, seed=DEV_SEED, sizes={f: (a.n + 60 if f == a.family else 0) for f in fams})
    tasks = [t for t in suite["tasks"] if t["family"] == a.family and not (set(t["expect_nodes"]) & locked)][: a.n]
    base = dict(get_budget(store), output_tokens=128, max_active_adapters=0)
    variants = json.loads(a.variants)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, score = [], {k: 0 for k in variants}
    for t in tasks:
        row = {"id": t["id"], "question": t["question"], "gold": t["gold"]}
        for name, over in variants.items():
            r = run_mode(a.runtime, [{"role": "user", "content": t["question"]}], "yaob", dict(base, **over))
            ok = score_task(t, r["reply"])[0] if r["reply"] else False
            score[name] += ok
            row[name] = {"ok": ok, "reply": r["reply"][:240]}
        rows.append(row)
    (out / "rows.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    disc = {f"{x} only": sum(1 for r in rows if r[x]["ok"] and not all(r[y]["ok"] for y in variants)) for x in variants}
    rep = {"family": a.family, "n": len(tasks), "seed": DEV_SEED, "correct": score, "discordant": disc, "variants": variants}
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
