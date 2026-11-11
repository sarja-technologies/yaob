"""Stage 2: page facts for every listed topic (resolved title, page id, Wikidata item, size, reach).

`langlinks` (how many language editions have the article) is the reach signal used
when a family has more candidates than its target: the most widely covered topics
are kept. It is a proxy for general importance, not a quality measure.

Usage: python -m mpnstudio.corpus.pages --out data/corpus
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .wiki import WikiClient

WP = "en.wikipedia.org"


def fetch_pageinfo(client: WikiClient, titles: list[str]) -> dict[str, dict]:
    info: dict[str, dict] = {}
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        d = client.get(WP, action="query", prop="pageprops|info|langlinkscount", ppprop="wikibase_item|disambiguation",
                       redirects=1, titles="|".join(chunk))
        q = d.get("query", {})
        norm = {n["from"]: n["to"] for n in q.get("normalized", [])}
        redir = {r["from"]: r["to"] for r in q.get("redirects", [])}
        pages = {p["title"]: p for p in q.get("pages", [])}
        for t in chunk:
            r = norm.get(t, t)
            r = redir.get(r, r)
            p = pages.get(r)
            if not p or p.get("missing") or p.get("invalid"):
                info[t] = {"missing": True}
                continue
            pp = p.get("pageprops", {})
            info[t] = {"title": p["title"], "pageid": p["pageid"], "qid": pp.get("wikibase_item"),
                       "disambiguation": "disambiguation" in pp, "length": p.get("length", 0),
                       "lastrevid": p.get("lastrevid"), "touched": p.get("touched"),
                       "langlinks": p.get("langlinkscount", 0), "redirected": r != t}
        if (i // 50) % 20 == 0:
            print(f"  pageinfo {i + len(chunk)}/{len(titles)}", flush=True)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus")
    args = ap.parse_args()
    out = Path(args.out)
    client = WikiClient(out / "cache")
    # Level 4 in full, plus level 5 only for the families whose target exceeds what level 4 offers.
    from collections import Counter
    from .families import TARGETS, family_of
    l4 = json.loads((out / "vital_level4.json").read_text())
    l5 = json.loads((out / "vital_level5.json").read_text())
    have = Counter(family_of(p["page"], e["path"]) for p in l4["pages"] for e in p["entries"])
    thin = {f for f, t in TARGETS.items() if have[f] < t * 1.08}
    titles: list[str] = []
    seen = set()
    for p in l4["pages"]:
        for e in p["entries"]:
            if e["title"] not in seen:
                seen.add(e["title"])
                titles.append(e["title"])
    for p in l5["pages"]:
        for e in p["entries"]:
            if e["title"] not in seen and family_of(p["page"], e["path"]) in thin:
                seen.add(e["title"])
                titles.append(e["title"])
    print("families topped up from level 5:", sorted(thin))
    print(f"{len(titles)} unique listed titles")
    info = fetch_pageinfo(client, titles)
    (out / "pageinfo.json").write_text(json.dumps(info, ensure_ascii=False))
    ok = sum(1 for v in info.values() if not v.get("missing"))
    print(f"resolved {ok}, missing {len(info) - ok}, network calls {client.network_calls}, cache hits {client.cache_hits}")


if __name__ == "__main__":
    main()
