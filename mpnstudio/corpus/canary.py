"""Diagnostic canary nodes: invented entities whose facts cannot be in any pretrained model.

They exist to answer one question honestly: does the network supply knowledge, or does the core merely
recall what it already knew? A canary fact answered correctly can only have come through the network.

Canary nodes live in their own MPN (kind `diagnostic`, origin `diagnostic`), disabled by default, and are
never counted among the model's knowledge nodes. Evaluation enables that MPN at a pinned revision.

Each canary node has five facts in its statements. Four are also listed as structured facts (and so get
training questions). One appears only in a statement: no training question ever has it as the answer, so
answering it shows the node's memory holds the statement itself, not a memorised question-answer pair.

Usage: python -m mpnstudio.corpus.canary --db data/yaob-n10k.sqlite [--per-kind 40]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..store.db import Store

SLUG = "diagnostic-canary"
SYL_A = ["vel", "or", "tam", "kes", "dru", "il", "sar", "nym", "quo", "bra", "zen", "hul", "pir", "ost", "eff", "yar", "gwi", "lom", "tev", "ush"]
SYL_B = ["mar", "eth", "ond", "ix", "ava", "urn", "eck", "oli", "ane", "ugh", "ira", "osk", "emn", "adz", "ilt", "orv", "uma", "eph", "aro", "ist"]
FIRST = ["Ilse", "Corvin", "Maud", "Tobiah", "Senna", "Radek", "Yvaine", "Osric", "Liesl", "Bram", "Nadir", "Petra", "Hollis", "Ximena", "Jorah", "Katri"]
FIELDS = ["tidal cartography", "glacier acoustics", "seed archiving", "bridge metallurgy", "moth migration", "salt chemistry", "clockwork repair", "river law"]
COLOURS = ["slate green", "pale violet", "rust orange", "ink blue", "bone white", "amber", "charcoal", "sea grey"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

# kind -> [(property, statement template, held-out question for the statement-only case)]
KINDS = {
    "institute": [("founding year", "The {n} was founded in {v}.", "In which year was the {n} founded?"),
                  ("town", "The {n} is located in the town of {v}.", "Which town hosts the {n}?"),
                  ("founder", "The {n} was founded by {v}.", "Which person started the {n}?"),
                  ("research field", "The {n} studies {v}.", "Which subject is the {n} devoted to?"),
                  ("number of researchers", "The {n} employs {v} researchers.", "How many researchers does the {n} employ?")],
    "river": [("length in kilometres", "The {n} is {v} kilometres long.", "How many kilometres long is the {n}?"),
              ("source", "The {n} rises on {v}.", "On which mountain does the {n} rise?"),
              ("mouth", "The {n} empties into the {v}.", "Into which sea does the {n} empty?"),
              ("number of bridges", "The {n} is crossed by {v} bridges.", "How many bridges cross the {n}?"),
              ("first charted by", "The {n} was first charted by {v}.", "Who first charted the {n}?")],
    "mineral": [("colour", "{n} is {v} in colour.", "What colour is {n}?"),
                ("Mohs hardness", "{n} has a hardness of {v} on the Mohs scale.", "How hard is {n} on the Mohs scale?"),
                ("first described by", "{n} was first described by {v}.", "Who first described {n}?"),
                ("year first described", "{n} was first described in {v}.", "In which year was {n} first described?"),
                ("main locality", "{n} is found mainly near {v}.", "Near which place is {n} mainly found?")],
    "festival": [("month held", "The {n} is held every {v}.", "In which month is the {n} held?"),
                 ("town", "The {n} takes place in {v}.", "In which town does the {n} take place?"),
                 ("duration in days", "The {n} lasts {v} days.", "How many days does the {n} last?"),
                 ("started by", "The {n} was started by {v}.", "Who started the {n}?"),
                 ("year first held", "The {n} was first held in {v}.", "In which year was the {n} first held?")],
}
DOMAIN_NAMES = {"institute": "Invented institutes", "river": "Invented rivers", "mineral": "Invented minerals", "festival": "Invented festivals"}


def generate(per_kind: int, seed: int) -> list[dict]:
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

    out = []
    for kind, props in KINDS.items():
        for _ in range(per_kind):
            w = word()
            if kind == "institute":
                name, vals = f"{w} Institute", [str(rng.randint(1640, 1985)), word(), person(), rng.choice(FIELDS), str(rng.randint(12, 940))]
            elif kind == "river":
                name, vals = f"River {w}", [str(rng.randint(40, 2900)), f"Mount {word()}", f"{word()} Sea", str(rng.randint(2, 95)), person()]
            elif kind == "mineral":
                name, vals = f"{w}ite", [rng.choice(COLOURS), f"{rng.randint(2, 18) / 2:g}", person(), str(rng.randint(1710, 2015)), f"Lake {word()}"]
            else:
                name, vals = f"{w} Festival", [rng.choice(MONTHS), word(), str(rng.randint(2, 31)), person(), str(rng.randint(1300, 1990))]
            hidden = rng.randrange(len(props))
            out.append({"kind": kind, "title": name, "hidden": hidden,
                        "statements": [tmpl.format(n=name, v=v) for (_, tmpl, _), v in zip(props, vals)],
                        "facts": [{"property": p, "value": v} for i, ((p, _, _), v) in enumerate(zip(props, vals)) if i != hidden],
                        "statement_only": {"property": props[hidden][0], "value": vals[hidden], "question": props[hidden][2].format(n=name)}})
    return out


def install(store: Store, per_kind: int = 40, seed: int = 20261111, actor: str = "canary") -> dict:
    if store.mpn_by_slug(SLUG):
        raise SystemExit("the diagnostic canary MPN already exists in this store")
    nodes = generate(per_kind, seed)
    with store.edit(actor=actor, kind="import", message=f"diagnostic canary MPN: {len(nodes)} invented nodes (not counted as knowledge)") as tx:
        mid = tx.put_mpn(SLUG, "Diagnostic canaries", "diagnostic",
                         "Invented entities used only to test whether the network supplies knowledge. Not knowledge. Disabled by default.",
                         manifest={"package": "yaob.mpn.diagnostic-canary", "class": "diagnostic", "seed": seed, "generator": "mpnstudio/corpus/canary.py"},
                         installed=True, enabled=False)
        src = tx.put_source("synthetic", "synthetic", "Canary generator (invented facts)", "mpnstudio/corpus/canary.py", "project source",
                            revision=f"seed {seed}", attribution="Generated procedurally. No language model wrote these. Every fact is invented.")
        doms = {k: tx.put_domain(mid, k, DOMAIN_NAMES[k]) for k in KINDS}
        ids = []
        for i, n in enumerate(nodes):
            nid = tx.put_node(uid=f"yaob:canary:{seed}:{i}", type="knowledge", title=n["title"], summary=n["statements"][0], origin="diagnostic",
                              validation="checked",
                              cells={"knowledge": {"statements": [{"text": s, "source": str(src)} for s in n["statements"]],
                                                   "facts": [dict(f, source=str(src)) for f in n["facts"]]},
                                     "thinking": {"entity_class": [f"invented {n['kind']}"], "description": "Diagnostic canary: an invented entity.",
                                                  "answers": [f["property"] for f in n["facts"]]},
                                     "reasoning": {"constraints": [{"kind": "validity", "text": "Invented for diagnostics. Not a fact about the world."}],
                                                   "checks": [{"kind": "fact_consistency", "properties": [f["property"] for f in n["facts"]],
                                                               "text": "An answer that states one of these properties must match the stored fact."}],
                                                   "referrals": []}})
            tx.set_membership(nid, mid, doms[n["kind"]], "primary", {"basis": "generated"})
            tx.link_source(nid, src, f"node {i}")
            ids.append(nid)
        rev = tx.rev
    store.set_setting("canary", {"mpn_id": str(mid), "seed": seed, "nodes": len(nodes), "rev": rev,
                                 "statement_only": {str(i): n["statement_only"] for i, n in zip(ids, nodes)}})
    return {"mpn_id": mid, "nodes": len(nodes), "rev": rev}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--per-kind", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261111)
    args = ap.parse_args()
    print(json.dumps(install(Store(Path(args.db)), args.per_kind, args.seed), indent=1))


if __name__ == "__main__":
    main()
