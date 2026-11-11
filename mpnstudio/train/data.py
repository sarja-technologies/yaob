"""Training and evaluation examples built from the node network itself.

No language model writes these. Every answer is a sourced statement or a structured fact of a node,
so each example is verified by construction and traceable to the node's sources.

Phrasings are split: `TRAIN` phrasings are used for training, `HELD_OUT` phrasings only for evaluation.
That measures paraphrase generalisation, not new knowledge. Knowledge that the network must supply
(rather than the pretrained core) is measured with the diagnostic canary nodes (see canary.py).
"""
from __future__ import annotations

import hashlib
import json
import random

from ..store.db import Store

# property label -> ([training phrasings], [held-out phrasings]); {t} = node title
SPECIFIC = {
    "date of birth": (["When was {t} born?", "What is the date of birth of {t}?"], ["On what date was {t} born?", "{t} was born when?"]),
    "date of death": (["When did {t} die?", "What is the date of death of {t}?"], ["On what date did {t} die?", "{t} died when?"]),
    "place of birth": (["Where was {t} born?", "What is the birthplace of {t}?"], ["In which place was {t} born?"]),
    "place of death": (["Where did {t} die?", "In which place did {t} die?"], ["What is the place of death of {t}?"]),
    "capital": (["What is the capital of {t}?", "Which city is the capital of {t}?"], ["Name the capital of {t}.", "{t} has which capital?"]),
    "capital of": (["{t} is the capital of what?", "What is {t} the capital of?"], ["Of which place is {t} the capital?"]),
    "country": (["In which country is {t}?", "What country does {t} belong to?"], ["{t} is located in which country?"]),
    "continent": (["On which continent is {t}?", "What continent is {t} part of?"], ["{t} lies on which continent?"]),
    "population": (["What is the population of {t}?", "How many people live in {t}?"], ["How large is the population of {t}?"]),
    "area": (["What is the area of {t}?", "How large is {t} in area?"], ["What area does {t} cover?"]),
    "inception": (["When was {t} founded or created?", "What is the inception date of {t}?"], ["When did {t} come into being?"]),
    "dissolved": (["When was {t} dissolved?", "When did {t} cease to exist?"], ["In what year did {t} end?"]),
    "point in time": (["When did {t} take place?", "What is the date of {t}?"], ["At what point in time did {t} occur?"]),
    "start time": (["When did {t} begin?", "What is the start date of {t}?"], ["{t} started when?"]),
    "end time": (["When did {t} end?", "What is the end date of {t}?"], ["{t} ended when?"]),
    "author": (["Who wrote {t}?", "Who is the author of {t}?"], ["{t} was written by whom?"]),
    "creator": (["Who created {t}?", "Who is the creator of {t}?"], ["{t} was created by whom?"]),
    "composer": (["Who composed {t}?", "Who is the composer of {t}?"], ["{t} was composed by whom?"]),
    "director": (["Who directed {t}?", "Who is the director of {t}?"], ["{t} was directed by whom?"]),
    "discoverer or inventor": (["Who discovered or invented {t}?", "Who is credited with discovering or inventing {t}?"], ["{t} was discovered or invented by whom?"]),
    "occupation": (["What was the occupation of {t}?", "What did {t} do for a living?"], ["Which occupations did {t} have?"]),
    "country of citizenship": (["What was the citizenship of {t}?", "Of which country was {t} a citizen?"], ["{t} held citizenship of which country?"]),
    "atomic number": (["What is the atomic number of {t}?", "How many protons does an atom of {t} have?"], ["{t} has which atomic number?"]),
    "element symbol": (["What is the chemical symbol of {t}?", "Which symbol denotes {t}?"], ["{t} is written with which element symbol?"]),
    "chemical formula": (["What is the chemical formula of {t}?", "Which formula describes {t}?"], ["{t} has which chemical formula?"]),
    "official language": (["What is the official language of {t}?", "Which language is official in {t}?"], ["{t} has which official language?"]),
    "currency": (["What is the currency of {t}?", "Which currency is used in {t}?"], ["{t} uses which currency?"]),
    "parent taxon": (["What is the parent taxon of {t}?", "To which higher taxon does {t} belong?"], ["{t} is classified under which parent taxon?"]),
    "instance of": (["What kind of thing is {t}?", "{t} is an instance of what?"], ["What is {t} an example of?"]),
    "part of": (["What is {t} part of?", "{t} belongs to which larger whole?"], ["Of what is {t} a part?"]),
    "named after": (["What is {t} named after?", "After whom or what is {t} named?"], ["{t} takes its name from what?"]),
    "elevation above sea level": (["What is the elevation of {t}?", "How high above sea level is {t}?"], ["{t} stands at what elevation?"]),
    "length": (["What is the length of {t}?", "How long is {t}?"], ["{t} extends for what length?"]),
    "publication date": (["When was {t} published or released?", "What is the publication date of {t}?"], ["{t} first appeared when?"]),
    "genre": (["What genre is {t}?", "To which genre does {t} belong?"], ["{t} is classed in which genre?"]),
    "founded by": (["Who founded {t}?", "Who is the founder of {t}?"], ["{t} was founded by whom?"]),
    "field of work": (["In which field did {t} work?", "What was the field of work of {t}?"], ["{t} is known for work in which field?"]),
    "notable work": (["What is a notable work of {t}?", "Which work is {t} known for?"], ["Name a notable work by {t}."]),
}
GENERIC = (["What is the {p} of {t}?", "{t}: what is the {p}?"], ["Tell me the {p} of {t}.", "Regarding {t}, what is its {p}?"])
SKIP_PROPS = {"coordinates", "different from", "subclass of", "worked example", "taxon name", "has part", "shares border with", "participant",
              "participant in", "follows", "followed by", "influenced by", "symptoms and signs"}
RECITE_TRAIN = ["State what this node says about {t}.", "Summarise the node on {t}."]
RECITE_HELD = ["What does the network record about {t}?"]
DEFINE_TRAIN = ["What is {t}?", "Define {t}."]
DEFINE_HELD = ["Briefly, what is {t}?"]


def split_of(uid: str, seed: int = 0) -> float:
    """Stable pseudo-random number in [0, 1) per node, used for node-level splits."""
    return int(hashlib.sha256(f"{seed}:{uid}".encode()).hexdigest()[:8], 16) / 2**32


def node_examples(node: dict, max_statements: int = 6) -> dict:
    """All examples one node supports: {'train': [...], 'held_out': [...]}; each {q, a, kind, prop?}."""
    t = node["title"]
    k = node["cells"].get("knowledge", {})
    statements = [s["text"] for s in k.get("statements", [])[:max_statements] if s.get("text")]
    out = {"train": [], "held_out": []}
    if not statements:
        return out
    recite = " ".join(statements[:3])
    for q in RECITE_TRAIN:
        out["train"].append({"q": q.format(t=t), "a": recite, "kind": "recite"})
    for q in RECITE_HELD:
        out["held_out"].append({"q": q.format(t=t), "a": recite, "kind": "recite"})
    # one example per statement, so every memory slot has a training signal for the sentence it stands for
    for j, st in enumerate(statements):
        out["train"].append({"q": f"What is point {j + 1} of the node on {t}?", "a": st, "kind": "recite"})
    if node["type"] == "knowledge":
        for q in DEFINE_TRAIN:
            out["train"].append({"q": q.format(t=t), "a": statements[0], "kind": "define"})
        for q in DEFINE_HELD:
            out["held_out"].append({"q": q.format(t=t), "a": statements[0], "kind": "define"})
    by_prop: dict[str, list[str]] = {}
    for f in k.get("facts", []):
        p = f.get("property", "")
        if not p or p in SKIP_PROPS or not f.get("value"):
            continue
        v = str(f["value"])
        if f.get("as_of"):
            v += f" (as of {f['as_of']})"
        by_prop.setdefault(p, []).append(v)
    for p, vals in by_prop.items():
        vals = list(dict.fromkeys(vals))[:4]
        answer = ", ".join(vals)
        tr, ho = SPECIFIC.get(p, GENERIC)
        for q in tr:
            out["train"].append({"q": q.format(t=t, p=p), "a": answer, "kind": "fact", "prop": p, "values": vals})
        for q in ho:
            out["held_out"].append({"q": q.format(t=t, p=p), "a": answer, "kind": "fact", "prop": p, "values": vals})
    return out


def iter_nodes(store: Store, rev: int | None = None, origins: tuple | None = None):
    w, a = store.at(rev)
    sql = f"SELECT node_id, uid, type, title, cells, content_hash, origin FROM node_v WHERE {w} AND status='live'"
    for r in store.q(sql + " ORDER BY node_id", a):
        if origins and r["origin"] not in origins:
            continue
        yield {"id": r["node_id"], "uid": r["uid"], "type": r["type"], "title": r["title"], "cells": json.loads(r["cells"]),
               "content_hash": r["content_hash"], "origin": r["origin"]}


def build_examples(store: Store, rev: int | None = None, seed: int = 0) -> dict:
    """Examples for every live node at `rev`, with dataset hashes for the run record."""
    train, held = [], []
    per_node = {}
    for n in iter_nodes(store, rev):
        ex = node_examples(n)
        for e in ex["train"]:
            train.append(dict(e, node=n["id"]))
        for e in ex["held_out"]:
            held.append(dict(e, node=n["id"]))
        per_node[n["id"]] = {"hash": n["content_hash"], "train": len(ex["train"]), "held_out": len(ex["held_out"])}

    def h(items):
        m = hashlib.sha256()
        for e in items:
            m.update(f"{e['node']}|{e['q']}|{e['a']}\n".encode())
        return m.hexdigest()[:16]

    return {"train": train, "held_out": held, "per_node": per_node,
            "hashes": {"train": h(train), "held_out": h(held), "nodes": len(per_node)},
            "kinds": {k: sum(1 for e in train if e["kind"] == k) for k in ("recite", "define", "fact")}}


def sample_batch(rng: random.Random, train: list[dict], batch: int) -> list[dict]:
    return [train[rng.randrange(len(train))] for _ in range(batch)]
