"""Stage 5: turn fetched material into canonical node, edge and membership records.

Output (in --out): nodes.jsonl, edges.jsonl, mpns.json, sources.json, build_report.json.
Nothing here calls a model. Every statement is a sentence of a Wikipedia lead at a recorded
revision; every fact is a Wikidata claim; every procedure is code in this repository.

Initial cell mapping for a knowledge node (3CN schema, version 1):
  knowledge   statements (lead sentences) and facts (structured claims), each with its source
  thinking    how a query reaches this node: entity class, aliases and cue terms; the learned routing key lives here
  reasoning   what limits an answer drawn from this node: validity dates, things it must not be confused with,
              facts that carry an "as of" date, and typed referrals to related nodes

Usage: python -m mpnstudio.corpus.build --out data/corpus
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..runtime.procedures import REGISTRY
from .families import FAMILIES, FAMILY_SLUGS, TARGETS, segments
from .fetch import PROPS

MAX_STATEMENTS = 6
ABBREV = {"mr", "mrs", "ms", "dr", "prof", "st", "sr", "jr", "vs", "etc", "e.g", "i.e", "c", "ca", "cf", "fl", "no", "vol", "ed", "lit",
          "approx", "est", "inc", "ltd", "co", "mt", "ft", "gen", "col", "lt", "capt", "sgt", "rev", "hon", "u.s", "u.k", "a.d", "b.c", "b.c.e",
          "c.e", "p", "pp", "op", "al", "ph.d", "m.d", "b.a", "d.c", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
WIKIPEDIA_LICENSE = ("CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0/")
WIKIDATA_LICENSE = ("CC0 1.0", "https://creativecommons.org/publicdomain/zero/1.0/")


def clean_extract(text: str) -> str:
    text = text.replace(" ", " ").replace("​", "")
    text = re.sub("[\u2060\u2061\u2062\u2063\ufe0e\ufe0f]", "", text)          # word joiners, invisible operators, variation selectors
    text = re.sub("[\u2009\u200a\u202f]", " ", text)                                   # thin spaces
    for _ in range(3):
        text = re.sub(r"\(\s*[;,:]?\s*\)", "", text)                    # parentheses emptied by the extract
        text = re.sub(r"\(\s*[;,]\s*", "(", text)
        text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


MATH_CHARS = set("=+−×⋅∑∏∫∂∇√∞≤≥≠≈≡∈∉⊂⊆∀∃⇒→↦⋯…⏟⏞^")


def garbled(sentence: str) -> str | None:
    """Why a lead sentence is unusable as a statement, or None. The plaintext extract flattens markup: a display
    formula becomes a run of symbols, and a superscript joins its base ("2×10^30 kg" arrives as "2×1030 kg"), which
    would store a wrong number as a sourced statement."""
    if re.search(r"×\s?10[−-]?\d", sentence) or re.search(r"\d\s?[x×]\s?10\d{2}", sentence):
        return "flattened exponent"
    if sum(ch in MATH_CHARS for ch in sentence) >= 4 or "⏟" in sentence:
        return "flattened formula"
    if "=" in sentence and len(re.findall(r"[A-Za-z]{3,}", sentence)) < 5:
        return "flattened formula"
    if sentence.rstrip().endswith(":"):
        return "introduces a removed formula or list"
    if re.match(r"(where|with|and|or|then|i\.e\.)\s", sentence):
        return "fragment after a removed formula"
    return None


def split_sentences(text: str) -> list[str]:
    out, start, depth, i, n = [], 0, 0, 0, len(text)
    while i < n:
        ch = text[i]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif ch == "\n":
            if text[start:i].strip():
                out.append(text[start:i].strip())
            start, depth = i + 1, 0
        elif ch in ".!?" and depth == 0:
            j = i + 1
            while j < n and text[j] in "\"')”’":
                j += 1
            if j >= n or (text[j] == " " and j + 1 < n and (text[j + 1].isupper() or text[j + 1].isdigit() or text[j + 1] in "\"“'(")):
                prev = re.findall(r"[\w.]+$", text[start:i])
                word = prev[0].lower().rstrip(".") if prev else ""
                is_initial = len(word) == 1 and word.isalpha() and prev[0][0].isupper()
                if ch != "." or (word not in ABBREV and not is_initial):
                    out.append(text[start:j].strip())
                    start = j
        i += 1
    if text[start:].strip():
        out.append(text[start:].strip())
    return out


def fmt_time(v: dict) -> tuple[str, int | None]:
    t, prec = v.get("time", ""), v.get("precision", 9)
    m = re.match(r"([+-])(\d+)-(\d\d)-(\d\d)", t)
    if not m:
        return "", None
    sign, year, month, day = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    era = " BCE" if sign == "-" else ""
    y = -year if sign == "-" else year
    if prec >= 11 and month and day:
        return f"{day} {MONTHS[month - 1]} {year}{era}", y
    if prec == 10 and month:
        return f"{MONTHS[month - 1]} {year}{era}", y
    if prec == 9:
        return f"{year}{era}", y
    if prec == 8:
        return f"{year // 10 * 10}s{era}", y
    if prec == 7:
        c = (year - 1) // 100 + 1
        suf = "th" if 10 <= c % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(c % 10, "th")
        return f"{c}{suf} century{era}", y
    if prec == 6:
        k = (year - 1) // 1000 + 1
        suf = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
        return f"{k}{suf} millennium{era}", y
    if prec <= 5:                                    # deep time: Wikidata stores e.g. -3500000000 at precision 1
        return (f"about {year:,} years ago" if sign == "-" else f"about {year:,}"), y
    return f"{year}{era}", y


def fmt_number(s: str) -> str:
    s = s.lstrip("+")
    try:
        f = float(s)
    except ValueError:
        return s
    if f == int(f) and abs(f) < 1e15 and "." not in s:
        return f"{int(f):,}"
    return f"{f:,.6g}" if abs(f) >= 1e6 else f"{f:.6g}"


def build_facts(entity: dict, labels: dict, qid_to_uid: dict) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """-> (facts for the Knowledge cell, relations [(type, target_uid, pid)])."""
    facts, rels = [], []
    seen: set[tuple] = set()
    for pid, vals in entity.get("claims", {}).items():
        label, rel = PROPS[pid]
        kept = 0
        for v in vals:
            f = {"pid": pid, "property": label}
            if v["type"] == "wikibase-entityid":
                q = v["value"]["id"]
                name = labels.get(q, "")
                if not name:
                    continue
                f.update(value=name, qid=q)
                if q in qid_to_uid:
                    f["node"] = qid_to_uid[q]
                    if rel:
                        rels.append((rel, qid_to_uid[q], pid))
            elif v["type"] == "time":
                text, _ = fmt_time(v["value"])
                if not text:
                    continue
                f["value"] = text
            elif v["type"] == "quantity":
                unit = v["value"].get("unit", "1")
                ulabel = "" if unit == "1" else labels.get(unit.rsplit("/", 1)[-1], "")
                if unit != "1" and not ulabel:
                    continue
                f["value"] = fmt_number(v["value"]["amount"]) + (f" {ulabel}" if ulabel else "")
            elif v["type"] == "globecoordinate":
                f["value"] = f"{v['value']['latitude']:.4f}, {v['value']['longitude']:.4f}"
            elif v["type"] == "string":
                f["value"] = v["value"]
            elif v["type"] == "monolingualtext":
                f["value"] = v["value"].get("text", "")
            else:
                continue
            q = v.get("qualifiers", {})
            if "P585" in q:
                t, _ = fmt_time(q["P585"])
                if t:
                    f["as_of"] = t
            for qp, key in (("P580", "from"), ("P582", "until")):
                if qp in q:
                    t, _ = fmt_time(q[qp])
                    if t:
                        f[key] = t
            key = (pid, f["value"], f.get("as_of"), f.get("from"), f.get("until"))
            if key in seen:                          # the same claim stated twice (e.g. with different references)
                continue
            seen.add(key)
            facts.append(f)
            kept += 1
            if kept >= (6 if v["type"] == "wikibase-entityid" else 2):
                break
    return facts, rels


def domain_path(topic: dict) -> list[str]:
    """Up to two levels of sub-domain inside the family, from the list's own headings (original capitalisation)."""
    page = topic["list_page"]
    tail = page.split("/Level ", 1)[-1]
    tail = tail.split("/", 1)[1] if "/" in tail else ""
    raw = [x for x in tail.split("/") if x] + list(topic["path"])
    if not raw:
        return ["General"]
    if raw[0].lower() == "people":
        last = raw[-1] if len(raw) > 2 else (raw[1] if len(raw) > 1 else "People")
        return ["People", re.sub(r":\s*General$", "", last)]
    rest = [re.sub(r":\s*General$", "", x).strip() for x in raw[1:]] or [raw[0]]
    return [x for x in rest[:2] if x] or ["General"]


def uid_for(qid: str) -> str:
    return f"yaob:wd:{qid}"


def shingle_key(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:40]
    return hashlib.sha1(" ".join(words).encode()).hexdigest()


def build(out: Path) -> dict:
    sel = json.loads((out / "selection.json").read_text())
    leads = json.loads((out / "leads.json").read_text())
    wd = json.loads((out / "wikidata.json").read_text())
    labels = json.loads((out / "labels.json").read_text())
    lists = {lvl: json.loads((out / f"vital_level{lvl}.json").read_text()) for lvl in (4, 5)}
    list_meta = {p["page"]: p for lvl in lists.values() for p in lvl["pages"]}
    proc_count = Counter(p.family for p in REGISTRY.values())
    report: dict = {"rejected": defaultdict(list), "families": {}, "dropped_sentences": Counter()}

    # ---------------------------------------------------------------- choose valid topics, filling from the reserve
    by_fam_sel, by_fam_res = defaultdict(list), defaultdict(list)
    for t in sel["selected"]:
        by_fam_sel[t["family"]].append(t)
    for t in sel["reserve"]:
        by_fam_res[t["family"]].append(t)
    seen_text: dict[str, str] = {}
    chosen: list[dict] = []

    def usable(t: dict) -> str | None:
        lead = leads.get(t["title"])
        if not lead or not lead.get("extract"):
            return "no lead text"
        sents = []
        for s in split_sentences(clean_extract(lead["extract"])):
            why = garbled(s) if len(s) >= 25 else "too short"
            if why:
                report["dropped_sentences"][why] += 1
            else:
                sents.append(s)
        if not sents:
            return "lead too short"
        if t["qid"] not in wd:
            return "no structured item"
        key = shingle_key(sents[0])
        if key in seen_text:
            return f"same opening text as {seen_text[key]}"
        seen_text[key] = t["title"]
        t["_sents"] = sents
        return None

    for fam in FAMILY_SLUGS:
        k = TARGETS[fam] - proc_count.get(fam, 0)
        got = []
        for t in by_fam_sel[fam] + by_fam_res[fam]:
            if len(got) >= k:
                break
            why = usable(t)
            if why:
                report["rejected"][why].append(t["title"])
                continue
            got.append(t)
        report["families"][fam] = {"target": TARGETS[fam], "knowledge": len(got), "procedures": proc_count.get(fam, 0),
                                   "short": k - len(got), "from_level5": sum(1 for t in got if t["level"] == 5),
                                   "from_reserve": sum(1 for t in got if "rank" not in t)}
        chosen += got

    qid_to_uid = {t["qid"]: uid_for(t["qid"]) for t in chosen}
    title_to_uid = {}
    for t in chosen:
        title_to_uid[t["title"]] = qid_to_uid[t["qid"]]
        title_to_uid.setdefault(t["listed_as"], qid_to_uid[t["qid"]])
    fam_of_uid = {qid_to_uid[t["qid"]]: t["family"] for t in chosen}
    title_of_uid = {qid_to_uid[t["qid"]]: t["title"] for t in chosen}

    # ---------------------------------------------------------------- sources
    sources: dict[str, dict] = {}

    def src(key: str, **kw) -> str:
        sources.setdefault(key, kw)
        return key

    nodes, edges = [], []
    edge_seen: set[tuple[str, str, str]] = set()

    def add_edge(a: str, b: str, typ: str, weight: float, evidence: dict):
        if a == b or (a, b, typ) in edge_seen:
            return
        edge_seen.add((a, b, typ))
        edges.append({"src": a, "dst": b, "type": typ, "weight": weight, "origin": "reference", "evidence": evidence})

    # ---------------------------------------------------------------- knowledge nodes
    for t in chosen:
        uid = qid_to_uid[t["qid"]]
        lead, ent = leads[t["title"]], wd[t["qid"]]
        url = "https://en.wikipedia.org/wiki/" + t["title"].replace(" ", "_")
        s_wp = src(f"wp:{t['pageid']}:{lead['revid']}", kind="wikipedia", origin="reference", title=t["title"],
                   url=f"https://en.wikipedia.org/w/index.php?title={t['title'].replace(' ', '_')}&oldid={lead['revid']}",
                   license=WIKIPEDIA_LICENSE[0], license_url=WIKIPEDIA_LICENSE[1], revision=str(lead["revid"]),
                   retrieved_at=lead.get("timestamp") or "", attribution=f"Wikipedia contributors, “{t['title']}”, {url}")
        s_wd = src(f"wd:{t['qid']}", kind="wikidata", origin="reference", title=f"{ent['label'] or t['title']} ({t['qid']})",
                   url=f"https://www.wikidata.org/wiki/{t['qid']}", license=WIKIDATA_LICENSE[0], license_url=WIKIDATA_LICENSE[1],
                   revision="", retrieved_at="", attribution="Wikidata")
        lm = list_meta[t["list_page"]]
        s_list = src(f"list:{lm['page']}:{lm['revid']}", kind="wikipedia-list", origin="reference", title=lm["page"],
                     url=f"https://en.wikipedia.org/w/index.php?title={lm['page'].replace(' ', '_')}&oldid={lm['revid']}",
                     license=WIKIPEDIA_LICENSE[0], license_url=WIKIPEDIA_LICENSE[1], revision=str(lm["revid"]),
                     retrieved_at=lm.get("retrieved_at") or "", attribution="Wikipedia contributors, Vital articles")
        sents = t["_sents"]
        statements = [{"text": s[:480], "source": s_wp} for s in sents[:MAX_STATEMENTS]]
        facts, rels = build_facts(ent, labels, qid_to_uid)
        for f in facts:
            f["source"] = s_wd
        classes = [f["value"] for f in facts if f["pid"] in ("P31", "P279")][:4]
        aliases = [a for a in ent.get("aliases", []) if a.lower() != t["title"].lower()][:8]
        if t["listed_as"] != t["title"] and t["listed_as"] not in aliases:
            aliases.append(t["listed_as"])
        # reasoning cell
        constraints = []
        for f in facts:
            if f["pid"] == "P576":
                constraints.append({"kind": "validity", "text": f"Ceased to exist: {f['value']}. Answer present-tense questions accordingly."})
            if f["pid"] == "P570":
                constraints.append({"kind": "validity", "text": f"Died {f['value']}; statements describe a historical person."})
            if f["pid"] == "P1889":
                constraints.append({"kind": "distinct_from", "text": f"Not the same as {f['value']}.", **({"node": f["node"]} if "node" in f else {})})
            if "as_of" in f:
                constraints.append({"kind": "dated_value", "text": f"{f['property'].capitalize()} {f['value']} is as of {f['as_of']}; it may have changed."})
        for h in lead.get("hatnotes", []):
            if h in title_to_uid and title_to_uid[h] != uid:
                constraints.append({"kind": "distinct_from", "text": f"Not to be confused with {h}.", "node": title_to_uid[h]})
        checks = []
        numeric = sorted({f["property"] for f in facts if f["pid"] not in ("P31", "P279") and "qid" not in f})
        if numeric:
            checks.append({"kind": "fact_consistency", "properties": numeric,
                           "text": "An answer that states one of these properties must match the stored fact."})
        checks.append({"kind": "source_required", "text": "Claims drawn from this node cite its source revision."})
        for rel, dst, pid in rels:
            add_edge(uid, dst, rel, 1.0, {"source": "wikidata", "property": pid})
        mention = 0
        for link in lead.get("lead_links", []):
            dst = title_to_uid.get(link)
            if dst and dst != uid and mention < 30:
                add_edge(uid, dst, "mentions", 0.5, {"source": "wikipedia-lead", "revision": lead["revid"]})
                mention += 1
        nodes.append({
            "uid": uid, "type": "knowledge", "title": t["title"], "summary": sents[0][:300], "aliases": aliases,
            "family": t["family"], "domain_path": domain_path(t), "origin": "reference", "validation": "source_backed",
            "sources": [(s_wp, "lead section"), (s_wd, "claims"), (s_list, " > ".join(t["path"]))],
            "cells": {
                "knowledge": {"statements": statements, "facts": facts},
                "thinking": {"entity_class": classes, "description": ent.get("description", ""), "cues": aliases[:6],
                             "answers": sorted({f["property"] for f in facts})[:24]},
                "reasoning": {"constraints": constraints[:10], "checks": checks, "referrals": []},
            },
            "_meta": {"qid": t["qid"], "pageid": t["pageid"], "level": t["level"], "langlinks": t["langlinks"],
                      "wikipedia_revision": lead["revid"]},
        })

    # ---------------------------------------------------------------- procedure nodes
    s_code = src("authored:procedures", kind="authored", origin="reference", title="MPN Studio procedure library",
                 url="mpnstudio/runtime/procedures.py", license="project source", license_url="", revision="0.1.0", retrieved_at="",
                 attribution="Fylume, MPN Studio")
    for key, p in REGISTRY.items():
        uid = f"yaob:proc:{key}"
        concept_uid = title_to_uid.get(p.concept)
        ex_args, ex_out = p.examples[0]
        nodes.append({
            "uid": uid, "type": "procedure", "title": p.title, "summary": p.summary, "aliases": list(p.aliases),
            "family": p.family, "domain_path": ["Procedures"], "origin": "reference", "validation": "checked",
            "shared_families": list(p.shared), "sources": [(s_code, key)],
            "cells": {
                "knowledge": {"statements": [{"text": p.summary, "source": s_code}, {"text": f"Formula: {p.formula}", "source": s_code}],
                              "facts": [{"property": "worked example", "value": f"{json.dumps(ex_args)} gives {json.dumps(ex_out)}", "source": s_code}]},
                "thinking": {"impl": key, "params": p.params, "formula": p.formula, "unit": p.unit, "cues": list(p.aliases),
                             "entity_class": ["procedure"], "description": p.summary, "answers": ["computed result"]},
                "reasoning": {"constraints": [{"kind": "precondition", "text": c} for c in p.constraints],
                              "checks": [{"kind": "worked_example", "text": "The stored example is re-run by the test suite.", "args": ex_args, "expect": ex_out}],
                              "referrals": []},
            },
            "_meta": {"impl": key, "concept": p.concept},
        })
        fam_of_uid[uid] = p.family
        title_of_uid[uid] = p.title
        if concept_uid:
            add_edge(uid, concept_uid, "applies", 1.0, {"source": "authored", "procedure": key})

    # ---------------------------------------------------------------- referrals and derived shared memberships
    out_edges = defaultdict(list)
    for e in edges:
        out_edges[e["src"]].append(e)
    in_deg = Counter(e["dst"] for e in edges)
    shared_stats = Counter()
    for n in nodes:
        es = out_edges.get(n["uid"], [])
        typed = [e for e in es if e["type"] != "mentions"]
        best = sorted(typed, key=lambda e: -in_deg[e["dst"]])[:8] or sorted(es, key=lambda e: -in_deg[e["dst"]])[:5]
        n["cells"]["reasoning"]["referrals"] = [{"relation": e["type"], "node": e["dst"], "title": title_of_uid[e["dst"]]} for e in best]
        memberships = [{"family": n["family"], "role": "primary", "evidence": {"basis": "list section", "path": n["domain_path"]}}]
        extra = []
        for f in n.pop("shared_families", []):
            if f != n["family"]:
                extra.append((1.0, f, {"basis": "declared by the procedure"}))
        if n["type"] == "knowledge" and len(es) >= 6:
            fam_counts = Counter(fam_of_uid[e["dst"]] for e in es)
            total = sum(fam_counts.values())
            for f, c in fam_counts.items():
                if f != n["family"] and c >= 4 and c / total >= 0.3:
                    ex = [title_of_uid[e["dst"]] for e in es if fam_of_uid[e["dst"]] == f][:3]
                    extra.append((c / total, f, {"basis": "links into that MPN", "links": c, "of": total, "examples": ex}))
        for _, f, ev in sorted(extra, key=lambda x: -x[0])[:2]:
            memberships.append({"family": f, "role": "shared", "evidence": ev})
            shared_stats[(n["family"], f)] += 1
        n["memberships"] = memberships

    mpns = [{"slug": s, "name": name, "description": desc, "kind": "domain"} for s, name, desc in FAMILIES]
    report["rejected"] = {k: {"count": len(v), "examples": v[:8]} for k, v in report["rejected"].items()}
    report["totals"] = {"nodes": len(nodes), "knowledge": sum(1 for n in nodes if n["type"] == "knowledge"),
                        "procedures": sum(1 for n in nodes if n["type"] == "procedure"), "edges": len(edges),
                        "edge_types": dict(Counter(e["type"] for e in edges).most_common()),
                        "multi_membership_nodes": sum(1 for n in nodes if len(n["memberships"]) > 1),
                        "sources": len(sources), "statements": sum(len(n["cells"]["knowledge"]["statements"]) for n in nodes),
                        "facts": sum(len(n["cells"]["knowledge"]["facts"]) for n in nodes),
                        "nodes_without_edges": sum(1 for n in nodes if n["uid"] not in out_edges and in_deg[n["uid"]] == 0)}
    report["shared_membership_pairs"] = {f"{a} -> {b}": c for (a, b), c in shared_stats.most_common(40)}
    report["exact_10000"] = len(nodes) == 10000

    with (out / "nodes.jsonl").open("w") as f:
        for n in nodes:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
    with (out / "edges.jsonl").open("w") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    (out / "mpns.json").write_text(json.dumps(mpns, ensure_ascii=False, indent=1))
    (out / "sources.json").write_text(json.dumps(sources, ensure_ascii=False))
    (out / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus")
    args = ap.parse_args()
    r = build(Path(args.out))
    print(json.dumps({k: r[k] for k in ("totals", "exact_10000", "families")}, indent=1))
    print("rejected:", {k: v["count"] for k, v in r["rejected"].items()})


if __name__ == "__main__":
    main()
