"""Held-out release evaluation: what do the modules contribute, measured on matched settings?

Arms (same questions, same core, same decoding, same output limit):
  core                 the unmodified shared core, no node access
  retrieval            core + ordinary retrieval (frozen-encoder similarity + BM25) over the same nodes, as text
  yaob                 learned routing + the selected MPNs' specialist adapters + text of the top nodes + procedures
  yaob_closed_book     learned routing + specialist adapters, no node text      (what the learned path alone carries)
  abl_no_adapters      learned routing + node text, adapters off               (routing ablation: is it just retrieval?)
  abl_router_baseline  yaob with ordinary retrieval choosing the nodes          (router replaced)

Task families (all scored deterministically; no model judges another):
  fact_paraphrase      a stored fact of a reference node, asked with a phrasing never used in training
  canary_paraphrase    the same for invented canary nodes: the core cannot know these
  canary_statement     a canary fact that occurs only in a statement; no training question had it as its answer
  two_hop              an interdisciplinary question that needs two linked nodes in different MPNs
  procedure            a numeric or symbolic problem with a computed answer
  unsupported          a question about an invented entity that is in no node: the right answer is to say so
  adversarial          a question carrying distracting vocabulary from an unrelated MPN; routing must not follow it

The suite is built from the store by fixed rules and a seed, saved as a file, and never used for training.
The canary MPN is enabled only at the pinned evaluation revision; the head revision keeps it disabled.

Usage (on the machine running MPN Studio):
  python -m mpnstudio.evals.heldout build --out eval/heldout_v1.json
  python -m mpnstudio.evals.heldout run --suite eval/heldout_v1.json --out runs/eval-001 [--arms core,retrieval,yaob]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path

from ..corpus.canary import SLUG as CANARY_SLUG
from ..runtime import procedures
from ..store.db import Store
from ..train import data as D

# Each arm states its specialist count explicitly, so the arms keep their meaning when the runtime's default changes
# (max_active_adapters became 0 after eval-001). eval-001 ran with the then-default budget: context 4,096, max_active_adapters 1.
ARMS = {
    "core": {"mode": "core"},                                                                                      # the frozen core alone
    "retrieval": {"mode": "retrieval"},                                                                            # ordinary retrieval: node text in the prompt
    "yaob": {"mode": "yaob", "budget": {"max_active_adapters": 1}},                                                # learned routing + one specialist + node text
    "yaob_closed_book": {"mode": "yaob", "options": {"text": False}, "budget": {"max_active_adapters": 1}},         # learned routing + specialist, no node text
    "abl_no_adapters": {"mode": "yaob", "options": {"adapters": False},                                            # learned routing + node text, eval-001's referral handling
                        "budget": {"max_active_adapters": 0, "referral_mode": "plain", "fact_focus": False}},
    "deployed": {"mode": "yaob", "options": {"adapters": False},                                                   # the default from 12 Nov 2026: no specialist,
                 "budget": {"max_active_adapters": 0, "referral_mode": "relation", "fact_focus": True}},           # relation referrals, fact focus (chosen on dev sets)
    "abl_router_baseline": {"mode": "yaob", "options": {"router": "baseline"}, "budget": {"max_active_adapters": 1}},  # ordinary retrieval chooses, specialist applied
}
SPECIFIC_PROPS = ["date of birth", "date of death", "capital", "country", "continent", "inception", "atomic number", "element symbol", "chemical formula",
                  "author", "official language", "currency", "place of birth", "point in time", "named after", "parent taxon", "discoverer or inventor",
                  "composer", "director", "publication date", "founded by", "dissolved", "start time", "end time", "place of death", "country of citizenship"]
HOP = {"capital": "capital", "country": "country", "author": "author", "born_in": "place of birth", "died_in": "place of death", "composer": "composer",
       "director": "director", "discovered_by": "discoverer or inventor", "named_after": "namesake", "located_in": "containing region",
       "founded_by": "founder", "citizen_of": "country of citizenship", "capital_of": "territory it is the capital of", "continent": "continent"}
HOP_FACTS = ["date of birth", "date of death", "inception", "population", "area", "capital", "continent", "official language", "currency", "atomic number",
             "country", "place of birth"]
ABSTAIN = re.compile(r"no node|not (?:have|know|contain|find|supported|available|aware|in the (?:provided|given|nodes?|records?|context|information|text|sources?))|unknown|cannot (?:find|answer|determine|provide)|can't (?:find|answer)|"
                     r"don't (?:know|have)|do not (?:know|have)|no (?:information|record|data|knowledge|such)|unable to|not (?:a )?(?:real|recogni[sz]ed)|"
                     r"doesn't (?:exist|appear)|does not (?:exist|appear)|fictional|couldn't find|could not find|"
                     # added after eval-001, whose replies "None of the provided knowledge nodes mention X … cannot be
                     # determined" were refusals scored as answers; applied to every arm by `rescore`
                     r"cannot be (?:determined|answered|found|confirmed)|can(?:'|no)t be (?:determined|answered)|"
                     r"(?:do|does|did)(?: not|n't) (?:mention|include|contain|provide|say|state|specify)|"
                     r"none of the (?:provided|given|retrieved|listed)|no (?:mention|details?)\b", re.I)
DISTRACT = {
    "biology": "Leaving mitochondria, enzymes, chromosomes and cell membranes entirely aside: ",
    "chemistry": "Ignoring covalent bonds, molar mass, titration and oxidation states: ",
    "astronomy": "Setting aside galaxies, supernovae, red dwarfs and orbital resonance: ",
    "computing-ai": "Disregarding compilers, hash tables, neural networks and TCP sockets: ",
    "medicine-health": "Without reference to antibiotics, hypertension, vaccines or surgery: ",
    "arts-literature-music": "Forgetting sonnets, symphonies, frescoes and film scores for a moment: ",
}
PROC_TASKS = [
    ("math.gcd", "What is the greatest common divisor of 252 and 198?", {"a": 252, "b": 198}),
    ("math.lcm", "What is the least common multiple of 14 and 20?", {"a": 14, "b": 20}),
    ("math.mean", "What is the arithmetic mean of 12, 15, 22, 9 and 17?", {"values": [12, 15, 22, 9, 17]}),
    ("math.median", "What is the median of 41, 7, 19, 88, 23 and 5?", {"values": [41, 7, 19, 88, 23, 5]}),
    ("math.stdev", "What is the population standard deviation of 4, 8, 15, 16, 23 and 42?", {"values": [4, 8, 15, 16, 23, 42]}),
    ("math.combinations", "In how many ways can 4 items be chosen from 11 when order does not matter?", {"n": 11, "r": 4}),
    ("math.permutations", "How many ordered arrangements of 3 items can be made from 9 distinct items?", {"n": 9, "r": 3}),
    ("math.hypotenuse", "A right triangle has legs of 9 and 40. How long is the hypotenuse?", {"a": 9, "b": 40}),
    ("math.triangle_area_heron", "What is the area of a triangle with sides 13, 14 and 15?", {"a": 13, "b": 14, "c": 15}),
    ("math.arithmetic_series", "What is the sum of the first 40 terms of the arithmetic progression starting at 3 with common difference 5?", {"first": 3, "difference": 5, "n": 40}),
    ("math.percent_change", "By what percentage does a value change when it goes from 64 to 80?", {"old": 64, "new": 80}),
    ("math.fibonacci", "What is the 25th Fibonacci number, counting F(0) = 0 and F(1) = 1?", {"n": 25}),
    ("math.binomial_probability", "What is the probability of exactly 3 successes in 8 independent trials with success probability 0.25?", {"n": 8, "k": 3, "p": 0.25}),
    ("phys.kinetic_energy", "What is the kinetic energy in joules of a 1200 kg car moving at 25 metres per second?", {"mass_kg": 1200, "speed": 25}),
    ("phys.resistors", "What is the equivalent resistance of 12 ohm, 6 ohm and 4 ohm resistors connected in parallel?", {"resistances": [12, 6, 4], "arrangement": "parallel"}),
    ("phys.pendulum_period", "What is the period in seconds of a simple pendulum 2.5 metres long on Earth?", {"length_m": 2.5}),
    ("phys.half_life", "A sample starts with 800 grams and has a half-life of 12 years. How many grams remain after 36 years?", {"initial": 800, "half_life": 12, "time": 36}),
    ("phys.heat", "How many joules are needed to warm 3 kg of water by 25 kelvin, taking the specific heat as 4186 J/(kg K)?", {"mass_kg": 3, "specific_heat": 4186, "delta_t": 25}),
    ("chem.molar_mass", "What is the molar mass of Ca(OH)2 in grams per mole?", {"formula": "Ca(OH)2"}),
    ("chem.dilution", "50 mL of a 3 mol/L solution is diluted to 600 mL. What is the new concentration in mol/L?", {"c1": 3, "v1": 50, "v2": 600}),
    ("astro.kepler_period", "A body orbits the Sun with a semi-major axis of 5.2 astronomical units. What is its orbital period in years?", {"semi_major_axis_au": 5.2}),
    ("geo.haversine", "What is the great-circle distance in kilometres between latitude 35.6762, longitude 139.6503 and latitude -33.8688, longitude 151.2093?", {"lat1": 35.6762, "lon1": 139.6503, "lat2": -33.8688, "lon2": 151.2093}),
    ("health.bmi", "What is the body mass index of a person who weighs 82 kg and is 1.80 m tall?", {"weight_kg": 82, "height_m": 1.80}),
    ("cs.base_convert", "What is the decimal number 2026 written in hexadecimal?", {"number": "2026", "from_base": 10, "to_base": 16}),
    ("cs.levenshtein", "What is the Levenshtein distance between the words 'sunday' and 'saturday'?", {"a": "sunday", "b": "saturday"}),
    ("cs.subnet_hosts", "How many usable host addresses does an IPv4 /22 subnet have?", {"prefix_length": 22}),
    ("econ.compound_interest", "What is 4,000 worth after 7 years at 5 percent annual interest compounded quarterly?", {"principal": 4000, "annual_rate_percent": 5, "years": 7, "compounds_per_year": 4}),
    ("econ.loan_payment", "What is the monthly payment on a 150,000 loan over 20 years at 4.5 percent annual interest?", {"principal": 150000, "annual_rate_percent": 4.5, "years": 20, "payments_per_year": 12}),
    ("econ.break_even", "Fixed costs are 18,000, the price per unit is 45 and the variable cost per unit is 27. How many units must be sold to break even?", {"fixed_costs": 18000, "price_per_unit": 45, "variable_cost_per_unit": 27}),
    ("units.temperature", "What is 98.6 degrees Fahrenheit in Celsius?", {"value": 98.6, "from_unit": "f", "to_unit": "c"}),
    ("units.length", "How many kilometres are 26.2 miles?", {"value": 26.2, "from_unit": "mile", "to_unit": "km"}),
    ("date.day_of_week", "What day of the week was 20 July 1969?", {"date": "1969-07-20"}),
    ("date.days_between", "How many days are there from 1 March 2024 to 25 December 2024?", {"start": "2024-03-01", "end": "2024-12-25"}),
    ("lang.to_roman", "How is the number 2026 written in Roman numerals?", {"n": 2026}),
    ("civics.dhondt", "Four parties receive 340,000, 280,000, 160,000 and 60,000 votes. How are 7 seats allocated by the D'Hondt method?", {"votes": [340000, 280000, 160000, 60000], "seats": 7}),
    ("music.note_frequency", "What is the frequency in hertz of the note E4 in equal temperament with A4 at 440 Hz?", {"note": "E4"}),
    ("logic.valid_argument", "Is this argument valid in propositional logic: premises 'p -> q' and '~q', conclusion '~p'?", {"premises": "p -> q; ~q", "conclusion": "~p"}),
    ("bio.gc_content", "What is the GC content, in percent, of the DNA sequence ATGCGCGTATTAGCGC?", {"sequence": "ATGCGCGTATTAGCGC"}),
]

PROC_PICK = {"cs.subnet_hosts": "usable_hosts"}      # the question asks for one field of a multi-field result
QUESTION_WORDS = {"what", "when", "who", "whom", "which", "where", "how", "in", "on", "at", "tell", "regarding", "name", "give", "from", "by"}


# ---------------------------------------------------------------------------- scoring
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower().replace(",", "")          # NFKC: "C₁₃H₁₈O₂" -> "c13h18o2"
    return re.sub(r"\s+", " ", "".join(ch for ch in s if ch.isalnum() or ch in " .-")).strip()


def score_values(reply: str, values: list[str]) -> bool:
    r = norm(re.sub(r"\[\d+\]", "", reply))
    for v in values:
        v = str(v).split(" (as of")[0]
        g = norm(v)
        if not g:
            continue
        if g in r:
            return True
        years = re.findall(r"\b\d{3,4}\b", g)
        if years and re.search(r"[a-z]", g) and all(re.search(rf"\b{y}\b", r) for y in years):      # a date: the year must be right
            return True
        m = re.fullmatch(r"(-?\d+(?:\.\d+)?)( .*)?", g)
        if m and len(m.group(1)) >= 2:                                                           # a quantity: the number must be right
            if re.search(rf"(?<![\d.]){re.escape(m.group(1))}(?![\d])", r):
                return True
    return False


def numbers_in(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))]


def score_result(reply: str, result) -> bool:
    reply2 = re.sub(r"\[\d+\]", "", reply)
    if isinstance(result, bool):
        r = reply2.lower()
        yes = bool(re.search(r"\b(yes|valid|true|is prime|passes)\b", r)) and not re.search(r"\b(not valid|invalid|is not|false)\b", r)
        return yes if result else bool(re.search(r"\b(no|not valid|invalid|false|is not)\b", r))
    if isinstance(result, (int, float)):
        nums = numbers_in(reply2)
        tol = max(abs(result) * 2e-3, 0.006)
        return any(abs(n - result) <= tol for n in nums) or any(abs(n - round(result, d)) < 10 ** -(d + 2) for n in nums for d in (0, 1, 2))
    if isinstance(result, str):
        return norm(result) in norm(reply2)
    if isinstance(result, dict):
        return all(score_result(reply2, v) for v in result.values() if not isinstance(v, (list, dict)))
    if isinstance(result, list):
        if all(isinstance(x, (int, float)) for x in result):
            nums = numbers_in(reply2)
            it = iter(nums)
            return all(any(abs(n - x) <= max(abs(x) * 2e-3, 0.006) for n in it) for x in result)     # in order
        return all(norm(str(x)) in norm(reply2) for x in result)
    return False


def abstains(reply: str) -> bool:
    return bool(ABSTAIN.search(reply))


CLAUSES = re.compile(r"(?<=[.!?;:])\s+|\n+|,?\s+\b(?:but|however|although|though|yet|nevertheless)\b", re.I)
VALUE = re.compile(r"\b\d[\d,.]*\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b(?! be\b)"
                   r"|\bby\s+(?:the\s+)?[A-Z][a-z]+")


def asserts_unsupported(reply: str, name: str) -> bool:
    """Whether a reply states a value for the invented entity in a clause that does not itself decline, e.g. "No node
    mentions it, but the Garlocht Regatta is held in April." A value about some other, real entity (a year of a
    different observatory) does not count: the clause must name the entity or refer to it as "it"."""
    r = re.sub(r"\[\d+\]", " ", reply or "")
    for clause in CLAUSES.split(r):
        if not clause or abstains(clause):
            continue
        about = (name and name.lower() in clause.lower()) or re.search(r"\b(?:it|its)\b", clause, re.I)
        if about and VALUE.search(clause):
            return True
    return False


def score_task(task: dict, reply: str) -> tuple[bool, bool]:
    """(correct, raw containment hit). A reply that says it does not know is never correct for a question that has an
    answer, even when it repeats the value (pilot 2's composition scorer accepted "I have no information about X" as
    naming X). The raw hit is kept so the report can show how many replies this rule overruled."""
    if task["scoring"] == "abstain":
        # declining is correct only if the reply does not also assert a value for the invented entity
        a = abstains(reply) and not asserts_unsupported(reply, task.get("invented", ""))
        return a, a
    if task["scoring"] == "values":
        raw = score_values(reply, task["gold"])
    elif task["scoring"] == "result":
        raw = score_result(reply, task["gold"])
    else:
        raise ValueError(task["scoring"])
    return raw and not abstains(reply), raw


def gold_in_question(question: str, gold) -> bool:
    """A value that the question already contains cannot be scored by looking for it in the reply."""
    q = norm(question)
    vals = gold if isinstance(gold, list) else [gold]
    return any(len(norm(str(v).split(" (as of")[0])) >= 3 and norm(str(v).split(" (as of")[0]) in q for v in vals if v is not None)


# ---------------------------------------------------------------------------- suite construction
def build_suite(store: Store, seed: int = 20261111, sizes: dict | None = None) -> dict:
    sz = {"fact_paraphrase": 64, "canary_paraphrase": 40, "canary_statement": 20, "two_hop": 30, "procedure": len(PROC_TASKS), "unsupported": 20, "adversarial": 24}
    sz.update(sizes or {})
    rng = random.Random(seed)
    rev = store.head()
    mpns = {int(m["id"]): m for m in store.mpns(rev)}
    canary = store.mpn_by_slug(CANARY_SLUG, rev)
    canary_id = int(canary["id"]) if canary else None
    nodes = {n["id"]: n for n in D.iter_nodes(store, rev)}
    prim, mems = {}, defaultdict(list)
    for r in store.q("SELECT node_id, mpn_id, role FROM member_v WHERE rev_to IS NULL"):
        mems[r["node_id"]].append(r["mpn_id"])
        if r["role"] == "primary":
            prim[r["node_id"]] = r["mpn_id"]
    tasks: list[dict] = []

    skipped = defaultdict(int)

    def add(family, q, gold, scoring="values", nodes_=(), mpns_=(), forbid=(), **kw):
        if scoring == "values" and gold_in_question(q, gold):
            skipped[family] += 1
            return
        tasks.append({"id": f"{family}-{sum(1 for t in tasks if t['family'] == family) + 1:03d}", "family": family, "question": q, "gold": gold,
                      "scoring": scoring, "expect_nodes": [nodes[n]["uid"] for n in nodes_], "expect_mpns": [mpns[m]["slug"] for m in mpns_],
                      "forbid_mpns": list(forbid), **kw})

    def held_facts(n, specific_only=True):
        ex = [e for e in D.node_examples(n)["held_out"] if e["kind"] == "fact"]
        if specific_only:
            ex = [e for e in ex if e["prop"] in SPECIFIC_PROPS]
        # the training data keeps at most 4 values per property; a full list may be longer, and then a correct answer
        # outside the stored ones would be scored wrong. Only properties with fewer values than the cap are asked.
        return [e for e in ex if len(e["values"]) < 4]

    # fact_paraphrase: stratified over the knowledge MPNs
    by_mpn = defaultdict(list)
    for nid, n in nodes.items():
        if n["origin"] != "diagnostic" and n["type"] == "knowledge" and held_facts(n):
            by_mpn[prim.get(nid)].append(nid)
    pool_mpns = sorted(m for m in by_mpn if m is not None)
    used_nodes: set[int] = set()
    per = max(1, sz["fact_paraphrase"] // max(1, len(pool_mpns)))
    for m in pool_mpns:
        for nid in rng.sample(sorted(by_mpn[m]), min(per + 1, len(by_mpn[m]))):
            if sum(1 for t in tasks if t["family"] == "fact_paraphrase") >= sz["fact_paraphrase"]:
                break
            if sum(1 for t in tasks if t["family"] == "fact_paraphrase" and t["expect_mpns"][:1] == [mpns[m]["slug"]]) >= per + 1:
                break
            e = rng.choice(held_facts(nodes[nid]))
            add("fact_paraphrase", e["q"], e["values"], nodes_=[nid], mpns_=sorted(set(mems[nid])), prop=e["prop"])
            used_nodes.add(nid)

    # canaries
    if canary_id is not None:
        cn = sorted(n for n in nodes if prim.get(n) == canary_id)
        for nid in rng.sample(cn, min(sz["canary_paraphrase"], len(cn))):
            e = rng.choice([x for x in D.node_examples(nodes[nid])["held_out"] if x["kind"] == "fact"])
            add("canary_paraphrase", e["q"], e["values"], nodes_=[nid], mpns_=[canary_id], prop=e["prop"])
        so = (store.get_setting("canary") or {}).get("statement_only", {})
        ids = sorted(int(k) for k in so if int(k) in nodes)
        for nid in rng.sample(ids, min(sz["canary_statement"], len(ids))):
            add("canary_statement", so[str(nid)]["question"], [so[str(nid)]["value"]], nodes_=[nid], mpns_=[canary_id], prop=so[str(nid)]["property"])

    # two_hop: a typed link into a node of a different MPN, then one of that node's facts
    hops = []
    types = ",".join("?" * len(HOP))
    for r in store.q(f"SELECT src, dst, type FROM edge_v WHERE rev_to IS NULL AND type IN ({types})", tuple(HOP)):
        a, b = r["src"], r["dst"]
        if a in nodes and b in nodes and prim.get(a) != prim.get(b) and a not in used_nodes and nodes[a]["origin"] != "diagnostic":
            facts = defaultdict(list)
            for f in nodes[b]["cells"].get("knowledge", {}).get("facts", []):
                if f.get("property") in HOP_FACTS and HOP[r["type"]] != f["property"]:
                    facts[f["property"]].append(str(f["value"]))
            # the link itself must be unambiguous from the source node's facts
            src_vals = [f for f in nodes[a]["cells"].get("knowledge", {}).get("facts", []) if f.get("node") == nodes[b]["uid"]]
            same_prop = [f for f in nodes[a]["cells"].get("knowledge", {}).get("facts", []) if src_vals and f.get("property") == src_vals[0]["property"]]
            facts = {k: v for k, v in facts.items() if len(v) == 1}        # single-valued only: the gold is then complete
            if facts and len(same_prop) == 1:
                hops.append((a, b, r["type"], dict(facts)))
    rng.shuffle(hops)
    seen_pairs = set()
    for a, b, typ, facts in hops:
        if sum(1 for t in tasks if t["family"] == "two_hop") >= sz["two_hop"]:
            break
        if a in seen_pairs or b in seen_pairs:
            continue
        seen_pairs.update((a, b))
        p = rng.choice(sorted(facts))
        add("two_hop", f"What is the {p} of the {HOP[typ]} of {nodes[a]['title']}?", facts[p][:4], nodes_=[a, b],
            mpns_=sorted({prim[a], prim[b]}), via=typ, prop=p, bridge=nodes[b]["title"])

    # procedures
    proc_nodes = {n["cells"].get("thinking", {}).get("impl"): nid for nid, n in nodes.items() if n["type"] == "procedure"}
    for key, q, args in PROC_TASKS[: sz["procedure"]]:
        r = procedures.run(key, args)
        if r["ok"] and key in proc_nodes:
            nid = proc_nodes[key]
            add("procedure", q, r["result"][PROC_PICK[key]] if key in PROC_PICK else r["result"], "result", nodes_=[nid], mpns_=[prim[nid]], impl=key, args=args)

    # unsupported: invented entities that exist in no node (a different seed and name pattern from the canaries)
    r2 = random.Random(seed + 99)
    syl = ["qua", "zor", "min", "tul", "bex", "hav", "rin", "dol", "pesk", "yun", "garl", "fim", "ocht", "sarn", "wex", "juv"]
    forms = [("What is the founding year of the {w} Conservatory?", "Conservatory"), ("How long in kilometres is the {w} Canal?", "Canal"),
             ("Who first described the mineral {w}olite?", "mineral"), ("In which month is the {w} Regatta held?", "Regatta"),
             ("What is the population of the town of {w}?", "town"), ("Who founded the {w} Observatory?", "Observatory")]
    titles = {n["title"].lower() for n in nodes.values()}
    while sum(1 for t in tasks if t["family"] == "unsupported") < sz["unsupported"]:
        w = (r2.choice(syl) + r2.choice(syl) + r2.choice(["", r2.choice(syl)])).capitalize()
        q, _ = r2.choice(forms)
        q = q.format(w=w)
        if not any(w.lower() in t for t in titles):
            add("unsupported", q, None, "abstain", invented=w)

    # adversarial: a real fact question behind distracting vocabulary from an unrelated MPN
    slugs = {m["slug"]: mid for mid, m in mpns.items()}
    cands = [t for t in tasks if t["family"] == "fact_paraphrase"]
    rng.shuffle(cands)
    for t in cands:
        if sum(1 for x in tasks if x["family"] == "adversarial") >= sz["adversarial"]:
            break
        options = [d for d in DISTRACT if d in slugs and d not in t["expect_mpns"]]
        if not options:
            continue
        d = rng.choice(options)
        nid = store.node_by_uid(t["expect_nodes"][0])
        # a different node from the same MPN than the plain task used, so the two families do not share questions
        alt = [x for x in by_mpn[prim[nid]] if x not in used_nodes]
        if not alt:
            continue
        nid2 = rng.choice(sorted(alt))
        used_nodes.add(nid2)
        e = rng.choice(held_facts(nodes[nid2]))
        first = e["q"].split(" ", 1)[0]
        q2 = (first.lower() + e["q"][len(first):]) if first.lower() in QUESTION_WORDS else e["q"]      # keep names capitalised
        add("adversarial", DISTRACT[d] + q2, e["values"], nodes_=[nid2], mpns_=sorted(set(mems[nid2])), forbid=[d], prop=e["prop"])

    return {"version": 1, "seed": seed, "built_at_rev": rev, "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sizes": {f: sum(1 for t in tasks if t["family"] == f) for f in sz}, "tasks": tasks,
            "skipped_gold_in_question": dict(skipped),
            "note": "Deterministic scoring. fact/canary paraphrase: stored facts with held-out phrasings. canary_statement: facts never used as a training answer. "
                    "two_hop, procedure, unsupported, adversarial: not represented in training in any form."}


# ---------------------------------------------------------------------------- running
def post_stream(url: str, body: dict, timeout: int = 600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    ev, data = None, []
    for raw in resp:
        line = raw.decode("utf-8", "replace").rstrip("\n")
        if line.startswith("event:"):
            ev = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
        elif line == "" and ev:
            yield ev, json.loads("\n".join(data) or "{}")
            ev, data = None, []


def wait_ready(runtime: str, need_checkpoint: bool, timeout_s: float = 600) -> None:
    """Wait until the runtime has loaded the core (and a checkpoint, for yaob arms), so a restart does not turn into a
    run of error rows. Raises after timeout_s."""
    t0 = time.perf_counter()
    while True:
        try:
            st = json.loads(urllib.request.urlopen(runtime + "/status", timeout=10).read())
            if (st.get("core") or {}).get("loaded") and not st.get("loading") and (st.get("checkpoint") or not need_checkpoint):
                return
        except Exception:
            pass
        if time.perf_counter() - t0 > timeout_s:
            raise SystemExit(f"runtime at {runtime} is not ready (core loaded and, for yaob arms, a checkpoint active)")
        time.sleep(5)


def ask(runtime: str, question: str, arm: dict, rev: int, budget: dict) -> dict:
    b = dict(budget)
    b.update(arm.get("budget", {}))
    out = {"reply": "", "route": None, "error": None}
    t0 = time.perf_counter()
    for ev, d in post_stream(runtime + "/run", {"messages": [{"role": "user", "content": question}], "mode": arm["mode"], "rev": rev,
                                                "budget": b, "options": arm.get("options", {})}):
        if ev == "route":
            out["route"] = d
        elif ev == "done":
            out.update(reply=d["text"], usage=d["usage"], timing=d["timing"], memory=d["memory"])
        elif ev == "error":
            out["error"] = d["error"]
    out["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


def pinned_eval_revision(store: Store) -> tuple[int, bool]:
    """A revision at which the canary MPN is enabled. Head is left with it disabled."""
    c = store.mpn_by_slug(CANARY_SLUG)
    if c is None:
        return store.head(), False
    if c["enabled"]:
        return store.head(), True
    with store.edit(actor="evaluation", kind="system", message="enable diagnostic canaries for evaluation (pinned revision)") as tx:
        tx.set_mpn_state(int(c["id"]), enabled=True)
        rev = tx.rev
    with store.edit(actor="evaluation", kind="system", message="disable diagnostic canaries again") as tx:
        tx.set_mpn_state(int(c["id"]), enabled=False)
    return rev, True


def run_suite(store: Store, suite: dict, out_dir: Path, runtime: str, arms: list[str], budget: dict, limit: int = 0, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rev, canaries = pinned_eval_revision(store)
    slug_of = {m["id"]: m["slug"] for m in store.mpns(rev)}
    tasks = suite["tasks"][:limit] if limit else suite["tasks"]
    uid_to_id = {t_uid: str(store.node_by_uid(t_uid)) for t in tasks for t_uid in t["expect_nodes"]}
    status = json.loads(urllib.request.urlopen(runtime + "/status", timeout=10).read())
    rows_path = out_dir / "rows.jsonl"
    done = set()
    if rows_path.exists():
        for line in rows_path.open():
            r = json.loads(line)
            done.add((r["arm"], r["id"]))
    wait_ready(runtime, need_checkpoint=any(ARMS[a]["mode"] == "yaob" for a in arms))
    f = rows_path.open("a")
    t_start = time.perf_counter()
    for arm in arms:
        # cold measurement: evict every shard before the arm's first request
        try:
            urllib.request.urlopen(urllib.request.Request(runtime + "/evict", data=b"{}", headers={"Content-Type": "application/json"}), timeout=10).read()
        except Exception:
            pass
        for i, t in enumerate(tasks):
            if (arm, t["id"]) in done:
                continue
            r = ask(runtime, t["question"], ARMS[arm], rev, budget)
            ok, raw = score_task(t, r["reply"]) if r["reply"] and not r["error"] else (False, False)
            route = r["route"] or {}
            sel = [n["id"] for n in route.get("nodes", [])]
            active = [slug_of.get(m["id"], "") for m in route.get("mpns", []) if m.get("active")]
            exp_nodes = [uid_to_id[u] for u in t["expect_nodes"]]
            row = {"arm": arm, "id": t["id"], "family": t["family"], "ok": ok, "raw_hit": raw, "reply": r["reply"][:500], "error": r["error"],
                   "selected": len(sel), "node_hit": all(n in sel for n in exp_nodes) if exp_nodes else None,
                   "active_mpns": active, "mpn_recall": (len(set(active) & set(t["expect_mpns"])) > 0) if t["expect_mpns"] and arm not in ("core",) else None,
                   "mpn_precision": (len(set(active) & set(t["expect_mpns"])) / len(active)) if active and t["expect_mpns"] else None,
                   "forbidden_active": bool(set(active) & set(t["forbid_mpns"])) if t["forbid_mpns"] else None,
                   "shards_loaded_now": sum(1 for s in route.get("shards", []) if s.get("state") == "loaded now"),
                   "shard_bytes": sum(s.get("bytes", 0) for s in route.get("shards", [])),
                   "procedures": [{"impl": p["impl"], "ok": p["ok"]} for p in route.get("procedures", [])],
                   "usage": r.get("usage"), "timing": r.get("timing"), "memory": r.get("memory"), "first_in_arm": i == 0}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0:
                log(f"{arm}: {i + 1}/{len(tasks)}  elapsed {time.perf_counter() - t_start:.0f} s")
    f.close()
    rows = [json.loads(l) for l in rows_path.open()]
    report = summarise(rows, suite, arms)
    report.update(rev=rev, canaries_enabled_at_rev=canaries, runtime={k: status.get(k) for k in ("core", "encoder", "checkpoint", "learned")},
                  budget=budget, wall_s=round(time.perf_counter() - t_start, 1), finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (out_dir / "report.md").write_text(render_markdown(report, suite))
    return report


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)


def summarise(rows: list[dict], suite: dict, arms: list[str]) -> dict:
    fams = list(suite["sizes"])
    out: dict = {"arms": {}, "families": fams}
    for arm in arms:
        ar = [r for r in rows if r["arm"] == arm]
        if not ar:
            continue
        per = {}
        for fam in fams:
            fr = [r for r in ar if r["family"] == fam]
            k = sum(r["ok"] for r in fr)
            per[fam] = {"n": len(fr), "correct": k, "accuracy": round(k / len(fr), 4) if fr else None, "ci95": wilson(k, len(fr))}

        def mean(key, sub=None, cond=lambda r: True):
            v = [(r[key][sub] if sub else r[key]) for r in ar if r.get(key) is not None and cond(r) and (not sub or r[key].get(sub) is not None)]
            return round(statistics.fmean(v), 4) if v else None

        def med(key, sub):
            v = [r[key][sub] for r in ar if r.get(key) and r[key].get(sub) is not None]
            return round(statistics.median(v), 1) if v else None
        warm = [r for r in ar if not r["first_in_arm"]]
        k = sum(r["ok"] for r in ar)
        out["arms"][arm] = {
            "overall": {"n": len(ar), "correct": k, "accuracy": round(k / len(ar), 4), "ci95": wilson(k, len(ar))},
            "by_family": per, "errors": sum(1 for r in ar if r["error"]),
            "overruled_abstentions": sum(1 for r in ar if r.get("raw_hit") and not r["ok"]),
            "routing": {"node_recall": mean("node_hit"), "mpn_recall": mean("mpn_recall"), "mpn_precision": mean("mpn_precision"),
                        "forbidden_mpn_activated": mean("forbidden_active"), "active_mpns_mean": round(statistics.fmean(len(r["active_mpns"]) for r in ar), 2),
                        "selected_nodes_mean": round(statistics.fmean(r["selected"] for r in ar), 2)},
            "tokens": {"prompt_median": med("usage", "prompt_tokens"), "adapters_median": med("usage", "adapters_applied"),
                       "text_evidence_median": med("usage", "text_evidence_tokens"), "completion_median": med("usage", "completion_tokens")},
            "latency_ms": {"cold_first_request_total": next((r["timing"]["total_ms"] for r in ar if r["first_in_arm"] and r.get("timing")), None),
                           "warm_ttft_median": round(statistics.median([r["timing"]["ttft_ms"] for r in warm if r.get("timing")]), 1) if warm else None,
                           "warm_total_median": round(statistics.median([r["timing"]["total_ms"] for r in warm if r.get("timing")]), 1) if warm else None,
                           "warm_route_median": round(statistics.median([r["timing"]["route_ms"] for r in warm if r.get("timing")]), 2) if warm else None,
                           "tokens_per_s_median": med("timing", "tokens_per_s")},
            "memory": {"torch_peak_mib_max": max((r["memory"]["torch_peak_mib"] for r in ar if r.get("memory")), default=None),
                       "process_rss_mib_max": max((r["memory"]["process_rss_mib"] for r in ar if r.get("memory")), default=None),
                       "resident_cache_mib_max": round(max((r["memory"]["resident_cache_bytes"] for r in ar if r.get("memory")), default=0) / 2**20, 2),
                       "shards_loaded_total": sum(r["shards_loaded_now"] for r in ar)},
        }
    return out


def render_markdown(rep: dict, suite: dict) -> str:
    arms = list(rep["arms"])
    L = ["# Held-out evaluation", "", f"Suite version {suite['version']}, seed {suite['seed']}, {len(suite['tasks'])} tasks. Evaluated at revision {rep.get('rev')}.",
         "", "Accuracy by task family (correct / n). Scoring is deterministic.", "", "| Family | n | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
    for fam in rep["families"]:
        n = next((rep["arms"][a]["by_family"][fam]["n"] for a in arms), 0)
        L.append(f"| {fam} | {n} | " + " | ".join(f"{rep['arms'][a]['by_family'][fam]['correct']} ({rep['arms'][a]['by_family'][fam]['accuracy']:.0%})"
                                                   if rep['arms'][a]['by_family'][fam]['n'] else "–" for a in arms) + " |")
    L.append("| **all** | " + str(next(rep["arms"][a]["overall"]["n"] for a in arms)) + " | " + " | ".join(
        f"**{rep['arms'][a]['overall']['correct']} ({rep['arms'][a]['overall']['accuracy']:.0%})** [{rep['arms'][a]['overall']['ci95'][0]:.0%}–{rep['arms'][a]['overall']['ci95'][1]:.0%}]" for a in arms) + " |")
    L += ["", "Routing, cost and speed.", "", "| | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]

    def row(label, fn):
        L.append(f"| {label} | " + " | ".join(str(fn(rep["arms"][a])) if fn(rep["arms"][a]) is not None else "–" for a in arms) + " |")
    row("Replies that named the value but said they did not know (scored wrong)", lambda a: a["overruled_abstentions"])
    row("Expected node among those read", lambda a: a["routing"]["node_recall"])
    row("Expected MPN consulted", lambda a: a["routing"]["mpn_recall"])
    row("Share of consulted MPNs that were expected", lambda a: a["routing"]["mpn_precision"])
    row("Distractor MPN consulted (adversarial tasks)", lambda a: a["routing"]["forbidden_mpn_activated"])
    row("MPNs consulted per request", lambda a: a["routing"]["active_mpns_mean"])
    row("Nodes read per request", lambda a: a["routing"]["selected_nodes_mean"])
    row("Prompt tokens, median", lambda a: a["tokens"]["prompt_median"])
    row("Specialist adapters applied, median", lambda a: a["tokens"]["adapters_median"])
    row("of which text evidence", lambda a: a["tokens"]["text_evidence_median"])
    row("Warm time to first token, median ms", lambda a: a["latency_ms"]["warm_ttft_median"])
    row("Warm total, median ms", lambda a: a["latency_ms"]["warm_total_median"])
    row("Routing time, median ms", lambda a: a["latency_ms"]["warm_route_median"])
    row("First request after evicting all shards, total ms", lambda a: a["latency_ms"]["cold_first_request_total"])
    row("Tokens per second, median", lambda a: a["latency_ms"]["tokens_per_s_median"])
    row("Peak torch memory, MiB", lambda a: a["memory"]["torch_peak_mib_max"])
    row("Peak process RSS, MiB", lambda a: a["memory"]["process_rss_mib_max"])
    row("Peak resident MPN state, MiB", lambda a: a["memory"]["resident_cache_mib_max"])
    return "\n".join(L) + "\n"


def rescore(suite: dict, out_dir: Path) -> dict:
    """Score the stored replies of a finished run again with the current scorer, without asking the runtime anything.
    The run's own report.json/report.md are kept; the new ones are written as report-rescored.*, with the number of
    rows whose score changed per arm."""
    tasks = {t["id"]: t for t in suite["tasks"]}
    rows = [json.loads(l) for l in (out_dir / "rows.jsonl").open()]
    changed: dict[str, int] = {}
    for r in rows:
        ok, raw = score_task(tasks[r["id"]], r["reply"]) if r["reply"] and not r["error"] else (False, False)
        if ok != r["ok"]:
            changed[r["arm"]] = changed.get(r["arm"], 0) + 1
        r["ok"], r["raw_hit"] = ok, raw
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    orig = json.loads((out_dir / "report.json").read_text())
    report = summarise(rows, suite, arms)
    report.update({k: orig.get(k) for k in ("rev", "canaries_enabled_at_rev", "runtime", "budget", "wall_s", "finished")})
    report.update(rescored=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), rescore_changed=changed)
    (out_dir / "rows-rescored.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (out_dir / "report-rescored.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    note = ("\n\nRe-scored from the stored replies with the current scorer (no new requests). Rows whose score changed: "
            + (", ".join(f"{a} {n}" for a, n in changed.items()) or "none") + ".\n")
    (out_dir / "report-rescored.md").write_text(render_markdown(report, suite) + note)
    return report


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--db")
    b.add_argument("--out", required=True)
    b.add_argument("--seed", type=int, default=20261111)
    r = sub.add_parser("run")
    r.add_argument("--db")
    r.add_argument("--suite", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--runtime", default="http://127.0.0.1:8841")
    r.add_argument("--arms", default=",".join(ARMS))
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--output-tokens", type=int, default=128)
    rs = sub.add_parser("rescore")
    rs.add_argument("--suite", required=True)
    rs.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "rescore":
        suite = json.loads(Path(args.suite).read_text())
        rescore(suite, Path(args.out))
        print((Path(args.out) / "report-rescored.md").read_text())
        return
    from ..train.jobs import db_path
    store = Store(args.db or db_path())
    if args.cmd == "build":
        s = build_suite(store, args.seed)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(s, indent=1, ensure_ascii=False))
        print(json.dumps({"sizes": s["sizes"], "total": len(s["tasks"]), "rev": s["built_at_rev"]}))
    else:
        from ..api.app import get_budget
        budget = dict(get_budget(store), output_tokens=args.output_tokens)
        rep = run_suite(store, json.loads(Path(args.suite).read_text()), Path(args.out), args.runtime, args.arms.split(","), budget, args.limit)
        print(render_markdown(rep, json.loads(Path(args.suite).read_text())))


if __name__ == "__main__":
    main()
