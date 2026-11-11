"""Portable interchange: MPN packages and whole-model bundles.

Formats are plain directories (optionally zipped) of JSON Lines and safetensors files with a manifest
and checksums. Importing never executes anything from a package: procedure nodes can only name
implementations that already ship with MPN Studio, and that is checked on import.

MPN package  (<slug>-<version>/)
  manifest.json      name, slug, version, class, counts, dependencies, compatibility, source licences,
                     file checksums, resource estimates, evaluation results
  nodes.jsonl        canonical nodes whose primary membership is this MPN (portable `uid` identity)
  members.jsonl      memberships this MPN holds on nodes that live elsewhere: {uid, role, evidence}
  edges.jsonl        typed relations from this MPN's nodes: {src_uid, dst_uid, type, weight, origin, evidence}
  sources.jsonl      every source referenced, with licence and revision
  domains.jsonl      the MPN's sub-domain tree
  weights/           optional: this MPN's routing shard and specialist adapter for a named checkpoint and core

Model bundle  (yaob-<name>/)
  manifest.json      bundle format, model name, revision, exact counts, core reference (repository, revision,
                     licence, file checksums; base weights are referenced, not included), encoder reference,
                     checkpoint record with learned-parameter counts, runtime requirement, file checksums
  mpns/<slug>/...    one MPN package each
  schemas.json       cell schemas (3CN, 5CN, custom)
  layout.jsonl       board geometry: {uid, x, y, pinned}; regions.jsonl; views.json
  checkpoint/        router.safetensors, node_emb.f16.npy, nodes.json (uid-keyed), mpn_index.json (slug-keyed),
                     adapters/<slug>/ (LoRA specialists), manifest.json. Routing shards are rebuilt on import
                     against the importing store's ids.
  MODEL_CARD.md

A bundle needs MPN Studio's runtime: the modular network is not contained in the core's weights, and a GGUF
of the core alone is only the core.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

from ..runtime import procedures
from . import layout, schemas
from .db import Store, sid

FORMAT = "yaob-bundle/1"
PKG_FORMAT = "yaob-mpn-package/1"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows) -> int:
    n = 0
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def checksums(root: Path) -> dict:
    return {str(p.relative_to(root)): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json"}


def verify(root: Path, manifest: dict) -> list[str]:
    bad = []
    for rel, info in manifest.get("files", {}).items():
        p = root / rel
        if not p.is_file():
            bad.append(f"missing file {rel}")
        elif sha256_file(p) != info["sha256"]:
            bad.append(f"checksum mismatch in {rel}")
    return bad


# ---------------------------------------------------------------------------- MPN packages
def portable_adapter(adapter_dir: Path, core: dict | None) -> None:
    """An exported adapter names its base model by repository, not by the path it was trained from (which would also
    disclose the training machine's directory layout)."""
    cfg = adapter_dir / "adapter_config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        c["base_model_name_or_path"] = (core or {}).get("repo") or Path(str(c.get("base_model_name_or_path") or "")).name
        cfg.write_text(json.dumps(c, indent=2))


def export_mpn(store: Store, mpn_id: int, out: Path, rev: int | None = None, checkpoint_dir: Path | None = None,
               include_adapter: bool = True) -> dict:
    m = store.mpn(mpn_id, rev)
    if m is None:
        raise ValueError("no such MPN")
    out.mkdir(parents=True, exist_ok=True)
    w, a = store.at(rev, "m")
    wn, an = store.at(rev, "n")
    rows = store.q(f"SELECT n.*, m.role, m.domain_id, m.evidence AS mev FROM member_v m JOIN node_v n ON n.node_id=m.node_id AND {wn} "
                   f"WHERE m.mpn_id=? AND {w} AND n.status<>'tombstone' ORDER BY n.node_id", (*an, mpn_id, *a))
    doms = {int(d["id"]): d for d in store.domains(mpn_id, rev)}
    uid_of = {}
    src_ids: set[int] = set()
    nodes, members, licences = [], [], {}
    ids_primary = []
    for r in rows:
        uid_of[r["node_id"]] = r["uid"]
        if r["role"] != "primary":
            members.append({"uid": r["uid"], "role": r["role"], "evidence": json.loads(r["mev"]), "title": r["title"]})
            continue
        ids_primary.append(r["node_id"])
        srcs = store.node_sources(r["node_id"], rev)
        for s in srcs:
            src_ids.add(int(s["id"]))
        d = doms.get(r["domain_id"]) if r["domain_id"] else None
        nodes.append({"uid": r["uid"], "type": r["type"], "title": r["title"], "summary": r["summary"], "aliases": json.loads(r["aliases"]),
                      "schema": {"id": r["schema_id"], "version": r["schema_version"]}, "status": r["status"], "validation": r["validation"],
                      "origin": r["origin"], "cells": json.loads(r["cells"]), "content_hash": r["content_hash"],
                      "domain": d["slug"] if d else None, "evidence": json.loads(r["mev"]),
                      "sources": [{"source": s["id"], "locator": s["locator"]} for s in srcs]})
    # cells cite sources by id: collect those too
    for n in nodes:
        for st in n["cells"].get("knowledge", {}).get("statements", []) + n["cells"].get("knowledge", {}).get("facts", []):
            if str(st.get("source", "")).isdigit():
                src_ids.add(int(st["source"]))
    sources = []
    for sid_ in sorted(src_ids):
        s = store.q1("SELECT * FROM sources WHERE id=?", (sid_,))
        if s:
            sources.append({k: s[k] for k in ("id", "kind", "origin", "title", "url", "license", "license_url", "revision", "retrieved_at", "attribution")})
            licences[s["license"] or "unspecified"] = licences.get(s["license"] or "unspecified", 0) + 1
    edges, deps = [], {}
    if ids_primary:
        we, ae = store.at(rev, "e")
        for i in range(0, len(ids_primary), 500):
            chunk = ids_primary[i:i + 500]
            ph = ",".join("?" * len(chunk))
            for e in store.q(f"SELECT e.*, n.uid AS dst_uid, (SELECT p.slug FROM member_v mm JOIN mpn_v p ON p.mpn_id=mm.mpn_id AND p.rev_to IS NULL "
                             f"WHERE mm.node_id=e.dst AND mm.rev_to IS NULL AND mm.role='primary') AS dst_mpn "
                             f"FROM edge_v e JOIN node_v n ON n.node_id=e.dst AND {wn} WHERE e.src IN ({ph}) AND {we}", (*an, *chunk, *ae)):
                edges.append({"src_uid": uid_of[e["src"]], "dst_uid": e["dst_uid"], "type": e["type"], "weight": e["weight"], "origin": e["origin"],
                              "evidence": json.loads(e["evidence"])})
                if e["dst_mpn"] and e["dst_mpn"] != m["slug"]:
                    deps[e["dst_mpn"]] = deps.get(e["dst_mpn"], 0) + 1
    n_nodes = write_jsonl(out / "nodes.jsonl", nodes)
    write_jsonl(out / "members.jsonl", members)
    n_edges = write_jsonl(out / "edges.jsonl", edges)
    write_jsonl(out / "sources.jsonl", sources)
    write_jsonl(out / "domains.jsonl", [{"slug": d["slug"], "name": d["name"], "depth": d["depth"],
                                         "parent": doms[int(d["parent_id"])]["slug"] if d["parent_id"] else None} for d in doms.values()])
    klass = "knowledge-only"
    weights = None
    weight_bytes = 0
    if checkpoint_dir is not None and (checkpoint_dir / "shards" / f"mpn_{mpn_id}.safetensors").exists():
        (out / "weights").mkdir(exist_ok=True)
        shutil.copy2(checkpoint_dir / "shards" / f"mpn_{mpn_id}.safetensors", out / "weights" / "shard.safetensors")
        adir = checkpoint_dir / "adapters" / str(mpn_id)
        has_adapter = include_adapter and (adir / "adapter_config.json").exists()
        cm = json.loads((checkpoint_dir / "manifest.json").read_text())
        if has_adapter:
            shutil.copytree(adir, out / "weights" / "adapter", ignore=shutil.ignore_patterns("opt.pt", "state.json"))
            portable_adapter(out / "weights" / "adapter", cm.get("core"))
        weight_bytes = sum(f.stat().st_size for f in (out / "weights").rglob("*") if f.is_file())
        ainfo = json.loads((adir / "adapter.json").read_text()) if has_adapter and (adir / "adapter.json").exists() else {}
        weights = {"checkpoint": cm.get("id"), "core": (cm.get("core") or {}).get("name"), "core_revision": (cm.get("core") or {}).get("revision"),
                   "routing_shard": True, "specialist_adapter": has_adapter,
                   "adapter": {k: ainfo.get(k) for k in ("rank", "trainable_params", "validated", "closed_book_check", "trained_at_rev")} if has_adapter else None,
                   "note": "The routing shard is only meaningful together with the shared router of the same checkpoint, and the adapter only on "
                           "the same core revision. A package imported into another model contributes its knowledge at once and its learned "
                           "state after retraining there."}
        klass = "knowledge+routing" + ("+specialist" if has_adapter else "")
    if any(n["type"] == "procedure" for n in nodes):
        klass += "+deterministic-tools"
    record_bytes = (out / "nodes.jsonl").stat().st_size
    manifest = {"format": PKG_FORMAT, "slug": m["slug"], "name": m["name"], "kind": m["kind"], "description": m["description"], "version": m["version"],
                "class": klass, "exported_at_rev": rev if rev is not None else store.head(), "exported": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "counts": {"nodes": n_nodes, "shared_members": len(members), "edges": n_edges, "sources": len(sources), "domains": len(doms),
                           "procedures": sum(1 for n in nodes if n["type"] == "procedure")},
                "dependencies": [{"mpn": k, "edges": v, "kind": "referenced by edges; optional"} for k, v in sorted(deps.items(), key=lambda kv: -kv[1])],
                "compatibility": {"schemas": sorted({f"{n['schema']['id']}@{n['schema']['version']}" for n in nodes}),
                                  "procedure_implementations": sorted({n["cells"].get("thinking", {}).get("impl") for n in nodes if n["type"] == "procedure"}),
                                  "weights": weights},
                "licences": licences,
                "resources": {"record_bytes": record_bytes, "weight_bytes": weight_bytes, "estimate_resident_bytes": weight_bytes},
                "evaluation": m["manifest"].get("evaluation", {}), "installs_code": False}
    manifest["files"] = checksums(out)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    return manifest


def import_mpn(store: Store, pkg: Path, actor: str = "import") -> dict:
    """Install an MPN package. Nodes are matched by `uid`: a node that already exists gains a membership and is not duplicated."""
    manifest = json.loads((pkg / "manifest.json").read_text())
    if manifest.get("format") != PKG_FORMAT:
        raise ValueError(f"not an MPN package (format {manifest.get('format')!r})")
    bad = verify(pkg, manifest)
    if bad:
        raise ValueError("package failed verification: " + "; ".join(bad[:5]))
    nodes = list(read_jsonl(pkg / "nodes.jsonl"))
    problems = []
    for n in nodes:
        spec = schemas.get_schema(store, n["schema"]["id"], n["schema"]["version"])
        if spec is None:
            problems.append(f"{n['title']}: schema {n['schema']['id']} v{n['schema']['version']} is not installed")
            continue
        errs = schemas.validate_cells(spec, n["cells"])
        if errs:
            problems.append(f"{n['title']}: {errs[0]}")
        impl = n["cells"].get("thinking", {}).get("impl")
        if impl and impl not in procedures.REGISTRY:
            problems.append(f"{n['title']}: procedure '{impl}' is not built in; packages cannot bring code")
    if problems:
        raise ValueError(f"{len(problems)} nodes cannot be imported: " + "; ".join(problems[:5]))
    created = merged = 0
    with store.edit(actor=actor, kind="mpn", message=f"install MPN package {manifest['name']} {manifest['version']}") as tx:
        existing = tx.c.execute("SELECT mpn_id FROM mpn_v WHERE slug=? AND rev_to IS NULL", (manifest["slug"],)).fetchone()
        mid = tx.put_mpn(manifest["slug"], manifest["name"], manifest.get("kind", "domain"), manifest.get("description", ""), manifest.get("version", "0.1.0"),
                         {"package": f"yaob.mpn.{manifest['slug']}", "class": manifest.get("class"), "licences": manifest.get("licences"),
                          "evaluation": manifest.get("evaluation", {}), "imported": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                         installed=True, enabled=True, mpn_id=existing[0] if existing else None)
        dom = {}
        for d in sorted(read_jsonl(pkg / "domains.jsonl"), key=lambda d: d["depth"]):
            dom[d["slug"]] = tx.put_domain(mid, d["slug"], d["name"], dom.get(d["parent"]) if d["parent"] else None, d["depth"])
        src_map = {}
        for s in read_jsonl(pkg / "sources.jsonl"):
            src_map[str(s["id"])] = tx.put_source(s["kind"], s["origin"], s["title"], s["url"], s["license"], s["license_url"], s["revision"], s["retrieved_at"], s["attribution"])
        uid_to_id = {}
        for n in nodes:
            cells = n["cells"]
            for st in cells.get("knowledge", {}).get("statements", []) + cells.get("knowledge", {}).get("facts", []):
                if str(st.get("source", "")) in src_map:
                    st["source"] = str(src_map[str(st["source"])])
            cur = tx.c.execute("SELECT node_id, status FROM node_v WHERE uid=? AND rev_to IS NULL", (n["uid"],)).fetchone()
            if cur is not None:
                nid = cur[0]
                if cur[1] == "tombstone":
                    tx.update_node(nid, status="live")
                has_primary = tx.c.execute("SELECT 1 FROM member_v WHERE node_id=? AND rev_to IS NULL AND role='primary'", (nid,)).fetchone()
                tx.set_membership(nid, mid, dom.get(n.get("domain")), "shared" if has_primary else "primary", n.get("evidence") or {"basis": "package"})
                merged += 1
            else:
                nid = tx.put_node(uid=n["uid"], type=n["type"], title=n["title"], summary=n["summary"], aliases=n["aliases"], schema_id=n["schema"]["id"],
                                  schema_version=n["schema"]["version"], status=n["status"], validation=n["validation"], origin=n["origin"], cells=cells)
                tx.set_membership(nid, mid, dom.get(n.get("domain")), "primary", n.get("evidence") or {"basis": "package"})
                created += 1
            for s in n.get("sources", []):
                if str(s["source"]) in src_map:
                    tx.link_source(nid, src_map[str(s["source"])], s.get("locator", ""))
            uid_to_id[n["uid"]] = nid
        shared_added = 0
        for mrow in read_jsonl(pkg / "members.jsonl"):
            cur = tx.c.execute("SELECT node_id FROM node_v WHERE uid=? AND rev_to IS NULL AND status<>'tombstone'", (mrow["uid"],)).fetchone()
            if cur is not None:
                tx.set_membership(cur[0], mid, None, "shared", mrow.get("evidence") or {"basis": "package"})
                shared_added += 1
        edges = skipped = 0
        for e in read_jsonl(pkg / "edges.jsonl"):
            a = uid_to_id.get(e["src_uid"])
            b = uid_to_id.get(e["dst_uid"])
            if b is None:
                r = tx.c.execute("SELECT node_id FROM node_v WHERE uid=? AND rev_to IS NULL AND status<>'tombstone'", (e["dst_uid"],)).fetchone()
                b = r[0] if r else None
            if a is None or b is None:
                skipped += 1        # the other end lives in an MPN that is not installed here
                continue
            tx.put_edge(a, b, e["type"], e["weight"], e["origin"], e["evidence"])
            edges += 1
        rev = tx.rev
    for nid in uid_to_id.values():
        layout.place_node(store, 1, nid)
    return {"rev": rev, "mpn_id": sid(mid), "nodes_created": created, "nodes_already_present": merged, "shared_memberships": shared_added,
            "edges": edges, "edges_waiting_for_other_mpns": skipped, "class": manifest.get("class"),
            "note": "Knowledge is live now. Learned state for these nodes needs indexing, then training in this model."}


# ---------------------------------------------------------------------------- whole-model bundles
def export_bundle(store: Store, out: Path, name: str, checkpoint_dir: Path | None = None, core: dict | None = None, model_card: str | None = None,
                  include_diagnostic: bool = True, include_adapters: bool = True) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    rev = store.head()
    mpns = store.mpns(rev)
    (out / "mpns").mkdir()
    packages = []
    for m in mpns:
        if m["kind"] == "diagnostic" and not include_diagnostic:
            continue
        pm = export_mpn(store, int(m["id"]), out / "mpns" / m["slug"], rev, checkpoint_dir, include_adapter=include_adapters)
        packages.append({"slug": m["slug"], "name": m["name"], "kind": m["kind"], "installed": m["installed"], "enabled": m["enabled"],
                         "nodes": pm["counts"]["nodes"], "class": pm["class"]})
    (out / "schemas.json").write_text(json.dumps([s["spec"] for s in schemas.list_schemas(store)], indent=1))
    uid = {r[0]: r[1] for r in store.q("SELECT node_id, uid FROM node_v WHERE rev_to IS NULL")}
    write_jsonl(out / "layout.jsonl", ({"uid": uid[r["node_id"]], "x": r["x"], "y": r["y"], "pinned": r["pinned"]}
                                       for r in store.q("SELECT node_id, x, y, pinned FROM layout WHERE board_id=1") if r["node_id"] in uid))
    slug_of = {int(m["id"]): m["slug"] for m in mpns}
    dslug = {int(d["id"]): (slug_of.get(int(d["mpn_id"])), d["slug"]) for d in store.domains(rev=rev)}
    write_jsonl(out / "regions.jsonl", ({"kind": r["kind"], "mpn": slug_of.get(r["ref_id"]) if r["kind"] == "mpn" else dslug.get(r["ref_id"], (None, None))[0],
                                         "domain": None if r["kind"] == "mpn" else dslug.get(r["ref_id"], (None, None))[1], "x": r["cx"], "y": r["cy"], "r": r["r"]}
                                        for r in store.q("SELECT * FROM regions WHERE board_id=1")))
    (out / "views.json").write_text(json.dumps([dict(r) for r in store.q("SELECT name, x, y, zoom FROM saved_views WHERE board_id=1")]))
    ck = None
    if checkpoint_dir is not None:
        cko = out / "checkpoint"
        cko.mkdir()
        shutil.copy2(checkpoint_dir / "router.safetensors", cko / "router.safetensors")
        shutil.copy2(checkpoint_dir / "node_emb.f16.npy", cko / "node_emb.f16.npy")
        cm = json.loads((checkpoint_dir / "manifest.json").read_text())
        nodes = json.loads((checkpoint_dir / "nodes.json").read_text())
        (cko / "nodes.json").write_text(json.dumps({uid[int(k)]: v for k, v in nodes.items() if int(k) in uid}))
        (cko / "manifest.json").write_text(json.dumps({k: v for k, v in cm.items() if k not in ("shards", "shared")}, indent=1))
        (cko / "mpn_index.json").write_text(json.dumps({slug_of[int(k)]: v for k, v in _mpn_index(checkpoint_dir).items() if int(k) in slug_of}))
        for d in sorted((checkpoint_dir / "adapters").iterdir()) if include_adapters and (checkpoint_dir / "adapters").exists() else []:
            if (d / "adapter_config.json").exists() and int(d.name) in slug_of:
                dst = cko / "adapters" / slug_of[int(d.name)]
                shutil.copytree(d, dst, ignore=shutil.ignore_patterns("opt.pt", "state.json"))
                portable_adapter(dst, cm.get("core"))
                a = json.loads((dst / "adapter.json").read_text())
                a["nodes"] = {uid[int(k)]: v for k, v in a.get("nodes", {}).items() if int(k) in uid}
                (dst / "adapter.json").write_text(json.dumps(a, indent=1))
        ck = {"id": cm.get("id"), "rev": cm.get("rev"), "learned_params": cm.get("learned_params"), "router": (cm.get("router") or {}).get("spec"),
              "adapters": sorted(p.name for p in (cko / "adapters").iterdir()) if (cko / "adapters").exists() else [], "core": cm.get("core")}
    if model_card:
        (out / "MODEL_CARD.md").write_text(model_card)
    counts = store.counts(rev)
    manifest = {"format": FORMAT, "name": name, "exported": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "revision": rev, "counts": counts,
                "packages": packages, "core": core or {}, "checkpoint": ck,
                "encoder": {"repo": "Qwen/Qwen3-Embedding-0.6B", "license": "apache-2.0", "note": "frozen; referenced, not included"},
                "requires": {"runtime": "MPN Studio >= 0.1.0", "note": "The modular network runs in MPN Studio's runtime. The core's weights alone do not contain it."},
                "base_weights": "Not included. Retrieve the core at the repository and revision above; file checksums are listed under core.files."}
    manifest["files"] = checksums(out)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    return manifest


def _mpn_index(checkpoint_dir: Path) -> dict:
    from safetensors import safe_open
    with safe_open(str(checkpoint_dir / "shards" / "shared.safetensors"), "pt") as f:
        return json.loads(f.metadata()["mpn_index"])


def import_bundle(bundle: Path, db_path: Path, actor: str = "bundle-import", ck_root: Path | None = None) -> dict:
    """Create a new store from a bundle. Refuses to write into an existing store."""
    if Path(db_path).exists():
        raise ValueError(f"{db_path} already exists")
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError(f"not a Yaob bundle (format {manifest.get('format')!r})")
    bad = verify(bundle, manifest)
    if bad:
        raise ValueError("bundle failed verification: " + "; ".join(bad[:5]))
    store = Store(db_path)
    for spec in json.loads((bundle / "schemas.json").read_text()):
        if schemas.get_schema(store, spec["id"], spec["version"]) is None:
            schemas.save_schema(store, spec)
    results = []
    # two passes so that edges into MPNs installed later are not lost: nodes first, then relations
    order = sorted(manifest["packages"], key=lambda p: p["slug"])
    for p in order:
        results.append(import_mpn(store, bundle / "mpns" / p["slug"], actor))
    waiting = sum(r["edges_waiting_for_other_mpns"] for r in results)
    if waiting:
        with store.edit(actor=actor, kind="import", message="link relations across MPNs") as tx:
            for p in order:
                for e in read_jsonl(bundle / "mpns" / p["slug"] / "edges.jsonl"):
                    a = tx.c.execute("SELECT node_id FROM node_v WHERE uid=? AND rev_to IS NULL", (e["src_uid"],)).fetchone()
                    b = tx.c.execute("SELECT node_id FROM node_v WHERE uid=? AND rev_to IS NULL", (e["dst_uid"],)).fetchone()
                    if a and b:
                        tx.put_edge(a[0], b[0], e["type"], e["weight"], e["origin"], e["evidence"])
        for p in order:        # shared memberships whose node arrived with a later package
            mid = store.mpn_by_slug(p["slug"])
            pend = [m for m in read_jsonl(bundle / "mpns" / p["slug"] / "members.jsonl")]
            if pend and mid:
                with store.edit(actor=actor, kind="import", message=f"shared memberships of {p['slug']}") as tx:
                    for mrow in pend:
                        r = tx.c.execute("SELECT node_id FROM node_v WHERE uid=? AND rev_to IS NULL", (mrow["uid"],)).fetchone()
                        if r and not tx.c.execute("SELECT 1 FROM member_v WHERE node_id=? AND mpn_id=? AND rev_to IS NULL", (r[0], int(mid["id"]))).fetchone():
                            tx.set_membership(r[0], int(mid["id"]), None, "shared", mrow.get("evidence") or {"basis": "package"})
    with store.edit(actor=actor, kind="mpn", message="restore installed/enabled state") as tx:
        for p in manifest["packages"]:
            m = store.mpn_by_slug(p["slug"])
            if m and (m["enabled"] != p["enabled"] or m["installed"] != p["installed"]):
                tx.set_mpn_state(int(m["id"]), enabled=p["enabled"], installed=p["installed"])
    # geometry
    uid_to_id = {r[1]: r[0] for r in store.q("SELECT node_id, uid FROM node_v WHERE rev_to IS NULL")}
    c = store.conn()
    with store._wlock:
        c.execute("BEGIN IMMEDIATE")
        c.execute("INSERT OR IGNORE INTO boards(id, name, created) VALUES(1, 'Yaob network', datetime('now'))")
        c.execute("DELETE FROM layout WHERE board_id=1")
        c.execute("DELETE FROM layout_rtree")
        for r in read_jsonl(bundle / "layout.jsonl"):
            nid = uid_to_id.get(r["uid"])
            if nid is not None:
                c.execute("INSERT OR REPLACE INTO layout VALUES(1,?,?,?,?)", (nid, r["x"], r["y"], r["pinned"]))
                c.execute("INSERT INTO layout_rtree VALUES(?,?,?,?,?)", (nid, r["x"], r["x"], r["y"], r["y"]))
        c.execute("DELETE FROM regions WHERE board_id=1")
        mp = {m["slug"]: int(m["id"]) for m in store.mpns()}
        dm = {(d["mpn_id"], d["slug"]): int(d["id"]) for d in store.domains()}
        for r in read_jsonl(bundle / "regions.jsonl"):
            ref = mp.get(r["mpn"]) if r["kind"] == "mpn" else dm.get((str(mp.get(r["mpn"])), r["domain"]))
            if ref is not None:
                c.execute("INSERT OR REPLACE INTO regions VALUES(1,?,?,?,?,?)", (r["kind"], ref, r["x"], r["y"], r["r"]))
        for v in json.loads((bundle / "views.json").read_text()):
            c.execute("INSERT INTO saved_views(board_id, name, x, y, zoom, created) VALUES(1,?,?,?,?,datetime('now'))", (v["name"], v["x"], v["y"], v["zoom"]))
        layout.rebuild_tiles(c, 1)
        c.execute("COMMIT")
    store.set_setting("imported_bundle", {"name": manifest["name"], "revision": manifest["revision"], "exported": manifest["exported"], "core": manifest.get("core")})
    ck = None
    if (bundle / "checkpoint" / "router.safetensors").exists():
        ck = restore_checkpoint(store, bundle / "checkpoint", Path(ck_root) if ck_root else Path(db_path).parent / "banks")
    counts = store.counts()
    want = manifest["counts"]
    return {"db": str(db_path), "counts": counts, "matches_bundle": all(counts[k] == want[k] for k in ("canonical_live_nodes", "edges", "memberships", "mpns", "multi_membership_nodes")),
            "bundle_counts": {k: want[k] for k in ("canonical_live_nodes", "edges", "memberships", "mpns", "multi_membership_nodes")},
            "checkpoint": ck}


def restore_checkpoint(store: Store, src: Path, ck_root: Path) -> dict:
    """Rebuild a bundle's checkpoint against this store's ids: router rows are kept, routing shards are
    re-packaged with the new node and MPN ids, adapters are copied under the new MPN ids. Nodes whose
    content differs from what was trained are recorded as not trained. Needs torch."""
    import numpy as np
    import torch
    from ..model.bank import load_tensors, package_router, save_tensors, sha256_file as sha
    from ..model.router import Router, RouterSpec
    cm = json.loads((src / "manifest.json").read_text())
    ck_id = cm["id"]
    out = ck_root / "main" / "checkpoints" / ck_id
    if out.exists():
        raise ValueError(f"{out} already exists")
    out.mkdir(parents=True)
    shutil.copy2(src / "router.safetensors", out / "router.safetensors")
    shutil.copy2(src / "node_emb.f16.npy", out / "node_emb.f16.npy")
    tensors, meta = load_tensors(out / "router.safetensors")
    spec = RouterSpec(**(meta.get("spec") or (cm.get("router") or {}).get("spec")))
    router = Router(spec)
    router.load_state_dict(tensors)
    by_slug = {m["slug"]: int(m["id"]) for m in store.mpns()}
    mpn_index = {by_slug[s]: i for s, i in json.loads((src / "mpn_index.json").read_text()).items() if s in by_slug}
    cur = {r["uid"]: (r["node_id"], r["content_hash"]) for r in store.q("SELECT node_id, uid, content_hash FROM node_v WHERE rev_to IS NULL")}
    exported = json.loads((src / "nodes.json").read_text())
    node_rows, node_hash, nodes_json = {}, {}, {}
    for u, info in exported.items():
        if u not in cur:
            continue
        nid, h = cur[u]
        node_rows[nid], node_hash[nid] = info["row"], h
        same = h == info["hash"]
        nodes_json[str(nid)] = {"row": info["row"], "hash": h, "key_trained": bool(info.get("key_trained")) and same,
                                "adapter_trained": bool(info.get("adapter_trained")) and same}
    primary, shared = {}, {}
    for r in store.q("SELECT node_id, mpn_id, role FROM member_v WHERE rev_to IS NULL"):
        (primary.__setitem__(r["node_id"], r["mpn_id"]) if r["role"] == "primary" else shared.setdefault(r["node_id"], []).append(r["mpn_id"]))
    emb = torch.from_numpy(np.load(out / "node_emb.f16.npy").astype("float32"))
    pk = package_router(out, router, node_rows, node_hash, primary, shared, mpn_index, emb)
    (out / "nodes.json").write_text(json.dumps(nodes_json))
    adapters = []
    for d in sorted((src / "adapters").iterdir()) if (src / "adapters").exists() else []:
        if d.name in by_slug:
            dst = out / "adapters" / str(by_slug[d.name])
            shutil.copytree(d, dst)
            a = json.loads((dst / "adapter.json").read_text())
            a["nodes"] = {str(cur[u][0]): h for u, h in a.get("nodes", {}).items() if u in cur}
            a["mpn_id"] = by_slug[d.name]
            (dst / "adapter.json").write_text(json.dumps(a, indent=1))
            adapters.append(d.name)
    cm.update(pk, restored_from_bundle=True)
    (out / "manifest.json").write_text(json.dumps(cm, indent=1))
    c = store.conn()
    with store._wlock:
        c.execute("BEGIN IMMEDIATE")
        c.execute("INSERT OR IGNORE INTO banks(id, kind, spec, core, created) VALUES('main','router+adapters',?,?,datetime('now'))",
                  (json.dumps({"d_key": spec.d_key, "d_enc": spec.d_enc}), (cm.get("core") or {}).get("name", "")))
        for nid, row in node_rows.items():
            c.execute("INSERT OR REPLACE INTO bank_rows(bank_id, node_id, row, state) VALUES('main',?,?,'allocated')", (nid, row))
        c.execute("INSERT INTO checkpoints(id, bank_id, kind, path, parent, step, rev, metrics, sha256, status, created, job_id) "
                  "VALUES(?,?,?,?,?,?,?,?,?,?,datetime('now'),?)",
                  (ck_id, "main", "router+adapters", str(out), None, 0, store.head(), "{}", sha(out / "router.safetensors"), "saved", None))
        c.execute("COMMIT")
    store.set_setting("bank_spec", spec.to_json())
    store.set_setting("active_checkpoint", {"id": ck_id, "activated": None, "previous": None, "note": "restored from a bundle; activated when the runtime starts"})
    return {"id": ck_id, "path": str(out), "nodes": len(nodes_json), "trained_and_unchanged": sum(1 for v in nodes_json.values() if v["key_trained"]),
            "adapters": adapters}


def active_checkpoint_dir(store: Store) -> Path | None:
    ck = (store.get_setting("active_checkpoint") or {}).get("id")
    if not ck:
        return None
    r = store.q1("SELECT path FROM checkpoints WHERE id=?", (ck,))
    return Path(r["path"]) if r and Path(r["path"]).exists() else None


def main():
    """Offline tool. Usage:
      python -m mpnstudio.store.bundle export <db> <out_dir> --name yaob-n10k [--core-json CORE.json] [--card MODEL_CARD.md]
      python -m mpnstudio.store.bundle import <bundle_dir> <new_db>
      python -m mpnstudio.store.bundle export-mpn <db> <slug> <out_dir>
      python -m mpnstudio.store.bundle import-mpn <db> <package_dir>
      python -m mpnstudio.store.bundle verify <dir>"""
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("db"); e.add_argument("out"); e.add_argument("--name", default="yaob-n10k"); e.add_argument("--core-json"); e.add_argument("--card")
    e.add_argument("--no-adapters", action="store_true", help="leave the per-MPN specialist adapters out (the router and node data are kept)")
    i = sub.add_parser("import"); i.add_argument("bundle"); i.add_argument("db")
    em = sub.add_parser("export-mpn"); em.add_argument("db"); em.add_argument("slug"); em.add_argument("out")
    im = sub.add_parser("import-mpn"); im.add_argument("db"); im.add_argument("pkg")
    v = sub.add_parser("verify"); v.add_argument("dir")
    a = ap.parse_args()
    if a.cmd == "export":
        st = Store(a.db)
        core = json.loads(Path(a.core_json).read_text()) if a.core_json else (st.get_setting("core_reference") or {})
        m = export_bundle(st, Path(a.out), a.name, active_checkpoint_dir(st), core, Path(a.card).read_text() if a.card else None,
                          include_adapters=not a.no_adapters)
        print(json.dumps({"out": a.out, "revision": m["revision"], "counts": m["counts"], "files": len(m["files"]), "checkpoint": (m["checkpoint"] or {}).get("id")}, indent=1))
    elif a.cmd == "import":
        print(json.dumps(import_bundle(Path(a.bundle), Path(a.db)), indent=1))
    elif a.cmd == "export-mpn":
        st = Store(a.db)
        m = st.mpn_by_slug(a.slug)
        if m is None:
            raise SystemExit(f"no MPN {a.slug!r}")
        print(json.dumps({k: v for k, v in export_mpn(st, int(m["id"]), Path(a.out), None, active_checkpoint_dir(st)).items() if k != "files"}, indent=1))
    elif a.cmd == "import-mpn":
        print(json.dumps(import_mpn(Store(a.db), Path(a.pkg)), indent=1))
    else:
        bad = verify(Path(a.dir), json.loads((Path(a.dir) / "manifest.json").read_text()))
        print("verified" if not bad else "\n".join(bad))
        raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
