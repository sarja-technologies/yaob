"""Stage 6: load built corpus files into a model store as one import revision.

Usage: python -m mpnstudio.corpus.load --corpus data/corpus --db data/yaob-n10k.sqlite
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ..store.db import Store
from ..store.schemas import seed_schemas


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "general"


def load(corpus: Path, db: Path, actor: str = "corpus-import") -> dict:
    store = Store(db)
    seed_schemas(store)
    mpns = json.loads((corpus / "mpns.json").read_text())
    sources = json.loads((corpus / "sources.json").read_text())
    nodes = [json.loads(l) for l in (corpus / "nodes.jsonl").open()]
    edges = [json.loads(l) for l in (corpus / "edges.jsonl").open()]
    with store.edit(actor=actor, kind="import", message=f"Yaob N10K starter network: {len(nodes)} nodes") as tx:
        mpn_id = {}
        for m in mpns:
            mpn_id[m["slug"]] = tx.put_mpn(m["slug"], m["name"], m.get("kind", "domain"), m.get("description", ""),
                                           manifest={"package": f"yaob.mpn.{m['slug']}", "class": "knowledge-only"})  # a package export reports learned state once trained
        src_id = {}
        for key, s in sources.items():
            src_id[key] = tx.put_source(s["kind"], s["origin"], s["title"], s.get("url", ""), s.get("license", ""), s.get("license_url", ""),
                                        s.get("revision", ""), s.get("retrieved_at", ""), s.get("attribution", ""))
        dom_cache: dict[tuple, int] = {}

        def domain(mid: int, path: list[str]) -> int | None:
            parent = None
            for depth, name in enumerate(path[:2], start=1):
                key = (mid, parent, slug(name))
                if key not in dom_cache:
                    dslug = slug(name) if parent is None else f"{slug(path[0])}--{slug(name)}"
                    dom_cache[key] = tx.put_domain(mid, dslug, name, parent, depth)
                parent = dom_cache[key]
            return parent

        uid_to_id = {}
        for n in nodes:
            cells = n["cells"]
            # sources inside cells are stored by numeric id (as strings) so the UI can resolve them
            for st in cells["knowledge"].get("statements", []):
                st["source"] = str(src_id[st["source"]])
            for f in cells["knowledge"].get("facts", []):
                f["source"] = str(src_id[f["source"]])
            nid = tx.put_node(uid=n["uid"], type=n["type"], title=n["title"], summary=n["summary"], aliases=n["aliases"],
                              schema_id="3cn", schema_version=1, validation=n["validation"], origin=n["origin"], cells=cells)
            uid_to_id[n["uid"]] = nid
            for key, locator in n["sources"]:
                tx.link_source(nid, src_id[key], locator)
        for n in nodes:
            nid = uid_to_id[n["uid"]]
            for m in n["memberships"]:
                mid = mpn_id[m["family"]]
                dom = domain(mid, n["domain_path"]) if m["role"] == "primary" else None
                tx.set_membership(nid, mid, dom, m["role"], m["evidence"])
        # resolve node references held inside cells (uids -> ids are resolved at read time through node_by_uid)
        for e in edges:
            tx.put_edge(uid_to_id[e["src"]], uid_to_id[e["dst"]], e["type"], e["weight"], e["origin"], e["evidence"])
        rev = tx.rev
    c = store.counts()
    store.set_setting("corpus", {"import_rev": rev, "nodes": len(nodes), "edges": len(edges), "sources": len(sources)})
    return {"rev": rev, "counts": c}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--db", required=True)
    args = ap.parse_args()
    if Path(args.db).exists():
        raise SystemExit(f"{args.db} already exists; refusing to import into an existing store")
    print(json.dumps(load(Path(args.corpus), Path(args.db)), indent=1))


if __name__ == "__main__":
    main()
