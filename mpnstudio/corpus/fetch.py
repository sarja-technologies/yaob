"""Stage 4: fetch article leads and structured facts for the selected topics.

Wikipedia: plain-text lead extract, lead wikitext (for links between topics), revision id and timestamp.
Wikidata: labels, descriptions, aliases and claims of each topic's item, then English labels for the
values those claims point to.

Usage: python -m mpnstudio.corpus.fetch --out data/corpus [--stage leads|wikidata|labels|all]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .wiki import WikiClient

WP, WD = "en.wikipedia.org", "www.wikidata.org"

# Properties kept as facts or typed relations: pid -> (label, relation type or None for a plain fact).
PROPS = {
    "P31": ("instance of", "instance_of"), "P279": ("subclass of", "subclass_of"), "P361": ("part of", "part_of"),
    "P527": ("has part", "has_part"), "P17": ("country", "country"), "P131": ("located in", "located_in"),
    "P36": ("capital", "capital"), "P1376": ("capital of", "capital_of"), "P30": ("continent", "continent"),
    "P47": ("shares border with", "borders"), "P276": ("location", "location"), "P138": ("named after", "named_after"),
    "P61": ("discoverer or inventor", "discovered_by"), "P170": ("creator", "created_by"), "P50": ("author", "author"),
    "P86": ("composer", "composer"), "P57": ("director", "director"), "P800": ("notable work", "notable_work"),
    "P737": ("influenced by", "influenced_by"), "P1344": ("participant in", "participant_in"),
    "P710": ("participant", "participant"), "P155": ("follows", "follows"), "P156": ("followed by", "followed_by"),
    "P2579": ("studied in", "studied_by"), "P1269": ("facet of", "facet_of"), "P101": ("field of work", "field_of_work"),
    "P106": ("occupation", None), "P27": ("country of citizenship", "citizen_of"), "P19": ("place of birth", "born_in"),
    "P20": ("place of death", "died_in"), "P1542": ("has effect", "has_effect"), "P828": ("has cause", "has_cause"),
    "P1889": ("different from", "different_from"), "P37": ("official language", "official_language"),
    "P38": ("currency", "currency"), "P35": ("head of state", None), "P6": ("head of government", None),
    "P122": ("basic form of government", "form_of_government"), "P140": ("religion or worldview", "religion"),
    "P171": ("parent taxon", "parent_taxon"), "P105": ("taxon rank", None), "P225": ("taxon name", None),
    "P136": ("genre", "genre"), "P135": ("movement", "movement"), "P921": ("main subject", "main_subject"),
    "P1366": ("replaced by", "replaced_by"), "P1365": ("replaces", "replaces"), "P206": ("located next to body of water", "next_to_water"),
    "P403": ("mouth of the watercourse", "flows_into"), "P706": ("located on terrain feature", "on_terrain"),
    "P2175": ("medical condition treated", "treats"), "P2176": ("drug or therapy used for treatment", "treated_by"),
    "P780": ("symptoms and signs", "symptom"), "P1995": ("health specialty", "health_specialty"),
    "P277": ("programming language", "written_in"), "P178": ("developer", "developed_by"), "P112": ("founded by", "founded_by"),
    "P159": ("headquarters location", "headquartered_in"), "P452": ("industry", "industry"), "P463": ("member of", "member_of"),
    "P279x": ("", None),
    # literal-valued facts
    "P569": ("date of birth", None), "P570": ("date of death", None), "P571": ("inception", None), "P576": ("dissolved", None),
    "P580": ("start time", None), "P582": ("end time", None), "P585": ("point in time", None), "P577": ("publication date", None),
    "P575": ("time of discovery or invention", None), "P1082": ("population", None), "P2046": ("area", None), "P2044": ("elevation above sea level", None),
    "P2043": ("length", None), "P2048": ("height", None), "P2067": ("mass", None), "P2386": ("diameter", None), "P1086": ("atomic number", None),
    "P246": ("element symbol", None), "P274": ("chemical formula", None), "P2101": ("melting point", None), "P2102": ("boiling point", None),
    "P2054": ("density", None), "P2146": ("orbital period", None), "P2583": ("distance from Earth", None), "P625": ("coordinates", None),
    "P1566": ("", None), "P2131": ("nominal GDP", None), "P1081": ("Human Development Index", None), "P2250": ("life expectancy", None),
    "P4511": ("vertical depth", None), "P2120": ("radius", None), "P1120": ("number of deaths", None),
}
PROPS = {k: v for k, v in PROPS.items() if v[0]}


def fetch_leads(client: WikiClient, titles: list[str], log_every: int = 25) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(titles), 20):
        chunk = titles[i:i + 20]
        d = client.get(WP, action="query", prop="extracts|revisions", exintro=1, explaintext=1, exlimit=20,
                       rvprop="ids|timestamp|content", rvslots="main", rvsection=0, redirects=1, titles="|".join(chunk))
        for p in d.get("query", {}).get("pages", []):
            if p.get("missing"):
                continue
            rev = (p.get("revisions") or [{}])[0]
            wikitext = rev.get("slots", {}).get("main", {}).get("content", "")
            out[p["title"]] = {"pageid": p["pageid"], "extract": p.get("extract", ""), "revid": rev.get("revid"),
                               "timestamp": rev.get("timestamp"), "lead_links": lead_links(wikitext),
                               "hatnotes": hatnotes(wikitext)}
        if (i // 20) % log_every == 0:
            print(f"  leads {min(i + 20, len(titles))}/{len(titles)}", flush=True)
    return out


LINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")


def strip_templates(text: str) -> str:
    """Remove {{...}} templates (nested), <ref> footnotes, comments and tables."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    out, depth, i = [], 0, 0
    while i < len(text):
        two = text[i:i + 2]
        if two in ("{{", "{|"):
            depth += 1
            i += 2
        elif two in ("}}", "|}"):
            depth = max(0, depth - 1)
            i += 2
        else:
            if depth == 0:
                out.append(text[i])
            i += 1
    return "".join(out)


def lead_links(wikitext: str) -> list[str]:
    body = strip_templates(wikitext)
    seen, out = set(), []
    for m in LINK.finditer(body):
        t = m.group(1).strip().replace("_", " ")
        if not t or ":" in t.split(" ")[0] and t.split(":")[0] in ("File", "Image", "Category", "wikt", "Wikipedia", "Help", "Template"):
            continue
        t = t[0].upper() + t[1:]
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:120]


def hatnotes(wikitext: str) -> list[str]:
    """Targets of 'not to be confused with' style hatnotes at the top of the article."""
    out = []
    for m in re.finditer(r"\{\{\s*(?:Distinguish|Not to be confused with|For|About|Other uses)\s*\|([^{}]*)\}\}", wikitext[:3000], flags=re.I):
        for part in m.group(1).split("|"):
            part = part.strip()
            if part and "=" not in part and len(part) < 80 and part[0].isupper():
                out.append(part)
    return out[:6]


def truncated(d: dict) -> bool:
    return "truncated" in json.dumps(d.get("warnings", {}))


def get_entities(client: WikiClient, chunk: list[str], props: str) -> dict[str, dict]:
    """One wbgetentities read. The API cuts a response at 12 MB and says so only in a warning (fifty country items
    with all their claims exceed it); a truncated batch is split in half and asked again."""
    d = client.get(WD, action="wbgetentities", ids="|".join(chunk), props=props, languages="en", formatversion=2)
    if truncated(d) and len(chunk) > 1:
        h = len(chunk) // 2
        return {**get_entities(client, chunk[:h], props), **get_entities(client, chunk[h:], props)}
    if truncated(d):
        raise RuntimeError(f"Wikidata truncated the single item {chunk[0]}")
    return d.get("entities", {})


def fetch_entities(client: WikiClient, qids: list[str], props: str, what: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(qids), 50):
        chunk = qids[i:i + 50]
        for q, e in get_entities(client, chunk, props).items():
            if "missing" not in e:
                out[q] = e
        if (i // 50) % 20 == 0:
            print(f"  {what} {min(i + 50, len(qids))}/{len(qids)}", flush=True)
    return out


def slim_entity(e: dict) -> dict:
    claims = {}
    for pid, statements in e.get("claims", {}).items():
        if pid not in PROPS:
            continue
        vals = []
        best = [s for s in statements if s.get("rank") == "preferred"] or [s for s in statements if s.get("rank") != "deprecated"]
        for s in best[:12]:
            snak = s.get("mainsnak", {})
            if snak.get("snaktype") != "value":
                continue
            dv = snak["datavalue"]
            q = {}
            for qp in ("P585", "P580", "P582"):
                for qs in s.get("qualifiers", {}).get(qp, [])[:1]:
                    if qs.get("snaktype") == "value":
                        q[qp] = qs["datavalue"]["value"]
            vals.append({"type": dv["type"], "value": dv["value"], "qualifiers": q, "rank": s.get("rank")})
        if vals:
            claims[pid] = vals
    return {"label": e.get("labels", {}).get("en", {}).get("value", ""),
            "description": e.get("descriptions", {}).get("en", {}).get("value", ""),
            "aliases": [a["value"] for a in e.get("aliases", {}).get("en", [])][:12], "claims": claims}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus")
    ap.add_argument("--stage", default="all")
    args = ap.parse_args()
    out = Path(args.out)
    client = WikiClient(out / "cache")
    sel = json.loads((out / "selection.json").read_text())
    topics = sel["selected"] + sel["reserve"]

    if args.stage in ("all", "leads"):
        leads = fetch_leads(client, [t["title"] for t in topics])
        (out / "leads.json").write_text(json.dumps(leads, ensure_ascii=False))
        print(f"leads: {len(leads)} of {len(topics)}")

    if args.stage in ("all", "wikidata"):
        ents = fetch_entities(client, [t["qid"] for t in topics], "labels|descriptions|aliases|claims", "entities")
        slim = {q: slim_entity(e) for q, e in ents.items()}
        (out / "wikidata.json").write_text(json.dumps(slim, ensure_ascii=False))
        print(f"wikidata: {len(slim)} of {len(topics)}")

    if args.stage in ("all", "labels"):
        slim = json.loads((out / "wikidata.json").read_text())
        have = set(slim)
        need: dict[str, int] = {}
        for e in slim.values():
            for vals in e["claims"].values():
                for v in vals:
                    if v["type"] == "wikibase-entityid":
                        q = v["value"]["id"]
                        if q not in have:
                            need[q] = need.get(q, 0) + 1
                    elif v["type"] == "quantity":
                        unit = v["value"].get("unit", "1")
                        if unit != "1":
                            q = unit.rsplit("/", 1)[-1]
                            need[q] = need.get(q, 0) + 5
        # Value labels are looked up for the values that recur (and all units). Rare one-off values are dropped
        # rather than fetched: at the unidentified-client rate each 50 labels cost six seconds.
        ordered = [q for q in sorted(need, key=lambda q: -need[q]) if need[q] >= 2][:6000]
        # labels fetched by an earlier run are kept; only values not looked up before are fetched (batches are
        # formed from the frequency order, so a changed item set would otherwise miss the cache for every batch)
        prev = json.loads((out / "labels.json").read_text()) if (out / "labels.json").exists() else {}
        todo = [q for q in ordered if q not in prev]
        labels = fetch_entities(client, todo, "labels", "labels")
        lab = {q: prev[q] for q in ordered if q in prev}
        lab.update({q: e.get("labels", {}).get("en", {}).get("value", "") for q, e in labels.items()})
        lab.update({q: e["label"] for q, e in slim.items()})
        (out / "labels.json").write_text(json.dumps(lab, ensure_ascii=False))
        print(f"labels: {len(lab)} ({len(ordered)} values, {len(todo)} fetched now)")
    print(f"network calls {client.network_calls}, cache hits {client.cache_hits}")


if __name__ == "__main__":
    main()
