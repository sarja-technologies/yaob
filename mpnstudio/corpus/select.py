"""Stage 3: choose which listed topics become knowledge nodes, per family, to an exact total.

Per family: knowledge target = family target - number of procedure nodes in that family.
Level-4 topics come first. If a family has more than its target, the topics with the widest
language coverage are kept. If it has fewer, level-5 topics are added in the same order.
A reserve of extra candidates per family is kept so that topics whose text later proves
unusable can be replaced without changing the total.

Usage: python -m mpnstudio.corpus.select --out data/corpus
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..runtime.procedures import REGISTRY
from .families import FAMILY_SLUGS, TARGETS, family_of

RESERVE = 0.06


def candidates(out: Path) -> list[dict]:
    info = json.loads((out / "pageinfo.json").read_text())
    seen_pages: set[int] = set()
    cands = []
    for level in (4, 5):
        data = json.loads((out / f"vital_level{level}.json").read_text())
        for page in data["pages"]:
            for e in page["entries"]:
                pi = info.get(e["title"])
                if not pi or pi.get("missing") or pi.get("disambiguation") or not pi.get("qid"):
                    continue
                if pi["pageid"] in seen_pages:      # the same article listed twice (or reached through a redirect)
                    continue
                seen_pages.add(pi["pageid"])
                fam = family_of(page["page"], e["path"])
                cands.append({"title": pi["title"], "listed_as": e["title"], "pageid": pi["pageid"], "qid": pi["qid"],
                              "level": level, "family": fam, "path": e["path"], "list_page": page["page"],
                              "list_rev": page["revid"], "langlinks": pi.get("langlinks", 0), "length": pi.get("length", 0)})
    return cands


def select(out: Path) -> dict:
    cands = candidates(out)
    proc_count = Counter(p.family for p in REGISTRY.values())
    by_fam = defaultdict(list)
    for c in cands:
        by_fam[c["family"]].append(c)
    plan, chosen, reserve = {}, [], []
    for fam in FAMILY_SLUGS:
        k = TARGETS[fam] - proc_count.get(fam, 0)
        pool = sorted(by_fam[fam], key=lambda c: (c["level"], -c["langlinks"], c["title"]))
        l4 = [c for c in pool if c["level"] == 4]
        if len(l4) >= k:
            # level 4 alone over-fills the family: rank level 4 by reach, level 5 only as reserve
            ranked = sorted(l4, key=lambda c: (-c["langlinks"], c["title"])) + [c for c in pool if c["level"] == 5]
        else:
            ranked = pool
        n_res = max(8, int(k * RESERVE))
        take, spare = ranked[:k], ranked[k:k + n_res]
        for rank, c in enumerate(take):
            c["rank"] = rank
        chosen += take
        reserve += spare
        plan[fam] = {"target": TARGETS[fam], "procedures": proc_count.get(fam, 0), "knowledge_target": k,
                     "level4_available": len(l4), "level5_available": len(pool) - len(l4),
                     "selected_level4": sum(1 for c in take if c["level"] == 4), "selected_level5": sum(1 for c in take if c["level"] == 5),
                     "reserve": len(spare), "short": max(0, k - len(take)),
                     "min_langlinks_selected": min((c["langlinks"] for c in take), default=0)}
    return {"plan": plan, "selected": chosen, "reserve": reserve,
            "totals": {"knowledge": len(chosen), "procedures": sum(proc_count.values()),
                       "nodes": len(chosen) + sum(proc_count.values())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus")
    args = ap.parse_args()
    out = Path(args.out)
    sel = select(out)
    (out / "selection.json").write_text(json.dumps(sel, ensure_ascii=False))
    for fam, p in sel["plan"].items():
        print(f"{fam:26s} target {p['target']:4d} = {p['knowledge_target']:4d} knowledge (L4 {p['selected_level4']:4d}, L5 {p['selected_level5']:4d}) "
              f"+ {p['procedures']:3d} procedures   short {p['short']}  min reach {p['min_langlinks_selected']}")
    print(sel["totals"])


if __name__ == "__main__":
    main()
