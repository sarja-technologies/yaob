"""SYNTHETIC scale fixture: a store with N placeholder nodes, to measure storage and viewport behaviour.

This is not a model. Its nodes are generated placeholders (origin=synthetic). It is never trained on,
never evaluated as knowledge and never counted in a Yaob node count. It exists to measure how the store,
indexes and viewport queries behave at a size the real network has not reached.

Usage: python scripts/synth_scale_fixture.py build data/synthetic-1m.sqlite --nodes 1000000
       python scripts/synth_scale_fixture.py measure data/synthetic-1m.sqlite --out docs/measurements/scale-1m.json
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpnstudio.store import layout, schemas  # noqa: E402
from mpnstudio.store.db import Store, content_hash, now  # noqa: E402

WORDS = ("river mountain theorem protein treaty engine galaxy vowel market statute painting recipe glacier enzyme harbour empire alloy sonata "
         "census orbit fungus canal dialect ledger verdict fresco tide lattice kernel motif reef delta quorum anvil prism ember").split()


def build(path: Path, n: int, mpns: int, edges_per_node: float, seed: int):
    if path.exists():
        raise SystemExit(f"{path} exists")
    t0 = time.perf_counter()
    rng = random.Random(seed)
    s = Store(path)
    schemas.seed_schemas(s)
    c = s.conn()
    c.execute("PRAGMA synchronous=OFF")
    c.execute("BEGIN")
    c.execute("INSERT INTO revisions(rev, parent, created, actor, kind, message) VALUES(1, 0, ?, 'synthetic-fixture', 'import', ?)",
              (now(), f"SYNTHETIC scale fixture: {n} placeholder nodes"))
    c.execute("INSERT INTO sources(kind, origin, title, license, attribution) VALUES('synthetic','synthetic','Scale fixture (synthetic placeholders)','none','scripts/synth_scale_fixture.py')")
    dom_ids, did = [], 0
    for m in range(1, mpns + 1):
        c.execute("INSERT INTO mpn_v(mpn_id, slug, name, kind, description, rev_from) VALUES(?,?,?,?,?,1)",
                  (m, f"synthetic-{m}", f"Synthetic MPN {m}", "domain", "Synthetic placeholder module"))
        for j in range(8):
            did += 1
            c.execute("INSERT INTO domain_v(domain_id, mpn_id, parent_id, slug, name, depth, rev_from) VALUES(?,?,?,?,?,1,1)", (did, m, None, f"d{j}", f"Domain {j + 1}"))
            parent = did
            for k in range(6):
                did += 1
                c.execute("INSERT INTO domain_v(domain_id, mpn_id, parent_id, slug, name, depth, rev_from) VALUES(?,?,?,?,?,2,1)", (did, m, parent, f"d{j}--t{k}", f"Topic {j + 1}.{k + 1}"))
                dom_ids.append((m, did))
    c.execute("INSERT INTO ids VALUES('mpn', ?), ('domain', ?), ('node', ?)", (mpns + 1, did + 1, n + 1))

    def nodes():
        for i in range(1, n + 1):
            a, b, d = rng.choice(WORDS), rng.choice(WORDS), rng.choice(WORDS)
            title = f"{a.capitalize()} {b} {i}"
            summary = f"Synthetic placeholder about {a}, {b} and {d}."
            cells = {"knowledge": {"statements": [{"text": f"Placeholder statement {i} about {a} and {b}.", "source": "1"},
                                                  {"text": f"Placeholder statement {i} about {d}.", "source": "1"}], "facts": [{"property": "index", "value": str(i), "source": "1"}]},
                     "thinking": {"entity_class": ["synthetic record"], "cues": [a, b]}, "reasoning": {"constraints": [], "checks": [], "referrals": []}}
            yield (i, f"yaob:synth:{i}", "knowledge", title, summary, "[]", "3cn", 1, "live", "unverified", "synthetic",
                   json.dumps(cells), content_hash(title, summary, [], cells, "knowledge"), 1)

    batch, fts, mem = [], [], []
    for row in nodes():
        batch.append(row)
        fts.append((row[0], row[3], "", row[4], f"Placeholder statement {row[0]}", "synthetic Scale fixture"))
        m, d = dom_ids[rng.randrange(len(dom_ids))]
        mem.append((row[0], m, d, "primary", "{}", 1))
        if rng.random() < 0.04:
            m2 = rng.randrange(1, mpns + 1)
            if m2 != m:
                mem.append((row[0], m2, None, "shared", "{}", 1))
        if len(batch) >= 20000:
            c.executemany("INSERT INTO node_v(node_id, uid, type, title, summary, aliases, schema_id, schema_version, status, validation, origin, cells, content_hash, rev_from) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
            c.executemany("INSERT INTO node_fts(rowid, title, aliases, summary, body, sources) VALUES(?,?,?,?,?,?)", fts)
            c.executemany("INSERT INTO member_v(node_id, mpn_id, domain_id, role, evidence, rev_from) VALUES(?,?,?,?,?,?)", mem)
            batch, fts, mem = [], [], []
            if row[0] % 200000 == 0:
                print(f"  nodes {row[0]:,}  {time.perf_counter() - t0:.0f} s", flush=True)
    c.executemany("INSERT INTO node_v(node_id, uid, type, title, summary, aliases, schema_id, schema_version, status, validation, origin, cells, content_hash, rev_from) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
    c.executemany("INSERT INTO node_fts(rowid, title, aliases, summary, body, sources) VALUES(?,?,?,?,?,?)", fts)
    c.executemany("INSERT INTO member_v(node_id, mpn_id, domain_id, role, evidence, rev_from) VALUES(?,?,?,?,?,?)", mem)
    n_edges = int(n * edges_per_node)
    eb = []
    for e in range(1, n_edges + 1):
        a = rng.randrange(1, n + 1)
        b = min(n, max(1, a + rng.randrange(-2000, 2000))) if rng.random() < 0.8 else rng.randrange(1, n + 1)
        if a != b:
            eb.append((e, a, b, rng.choice(["related_to", "part_of", "mentions"]), 1.0, "derived", "{}", 1))
        if len(eb) >= 50000:
            c.executemany("INSERT INTO edge_v(edge_id, src, dst, type, weight, origin, evidence, rev_from) VALUES(?,?,?,?,?,?,?,?)", eb)
            eb = []
    c.executemany("INSERT INTO edge_v(edge_id, src, dst, type, weight, origin, evidence, rev_from) VALUES(?,?,?,?,?,?,?,?)", eb)
    c.execute("INSERT INTO ids VALUES('edge', ?)", (n_edges + 1,))
    c.execute("INSERT INTO node_source_v(node_id, source_id, locator, rev_from) SELECT node_id, 1, 'fixture', 1 FROM node_v")
    c.execute("COMMIT")
    t_insert = time.perf_counter() - t0
    s.rebuild_summaries()
    print(f"inserted in {t_insert:.0f} s; computing layout", flush=True)
    t1 = time.perf_counter()
    info = layout.compute_layout(s)
    t_layout = time.perf_counter() - t1
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    s.set_setting("synthetic_fixture", {"nodes": n, "mpns": mpns, "edges": n_edges, "seed": seed, "insert_s": round(t_insert, 1), "layout_s": round(t_layout, 1),
                                        "warning": "SYNTHETIC. Not a model. Not counted. Not trained."})
    print(json.dumps({"nodes": n, "edges": n_edges, "insert_s": round(t_insert, 1), "layout_s": round(t_layout, 1), "layout": info["nodes"],
                      "file_bytes": path.stat().st_size}))


def table_bytes(path: Path) -> dict:
    con = sqlite3.connect(path)
    try:
        rows = con.execute("SELECT name, SUM(pgsize) FROM dbstat GROUP BY name").fetchall()
    except sqlite3.OperationalError:
        return {}
    return {r[0]: r[1] for r in rows}


def pct(v, p):
    v = sorted(v)
    return round(v[min(len(v) - 1, int(len(v) * p))], 3)


def measure(path: Path, out: Path | None, cap: int):
    s = Store(path)
    info = s.get_setting("synthetic_fixture") or {}
    n = s.q1("SELECT COUNT(*) FROM node_v WHERE rev_to IS NULL")[0]
    e = s.q1("SELECT COUNT(*) FROM edge_v WHERE rev_to IS NULL")[0]
    tb = table_bytes(path)
    groups = {"node records (node_v + its indexes)": ["node_v", "node_head", "node_hist", "node_uid_head", "node_closed", "node_opened"],
              "text index (FTS5)": [k for k in tb if k.startswith("node_fts")],
              "memberships": [k for k in tb if k.startswith("member_")],
              "edges (edge_v + its indexes)": [k for k in tb if k.startswith("edge_")],
              "layout + spatial index": [k for k in tb if k.startswith("layout")],
              "node sources": [k for k in tb if k.startswith("nsrc") or k == "node_source_v"]}
    sizes = {g: sum(tb.get(k, 0) for k in ks) for g, ks in groups.items()}
    b = layout.bounds(s)
    rng = random.Random(3)
    res = {}
    W, H = 1920, 1000
    for label, frac in (("overview (whole board)", 1.0), ("one MPN wide", 0.2), ("one domain wide", 0.04), ("node level", 0.004), ("cell level", 0.0008)):
        times, payload, items, levels = [], [], [], set()
        for _ in range(60):
            w = (b[2] - b[0]) * frac
            h = w * H / W
            cx = rng.uniform(b[0] + w / 2, b[2] - w / 2) if frac < 1 else (b[0] + b[2]) / 2
            cy = rng.uniform(b[1] + h / 2, b[3] - h / 2) if frac < 1 else (b[1] + b[3]) / 2
            t0 = time.perf_counter()
            v = layout.viewport(s, 1, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], W / w, cap)
            times.append((time.perf_counter() - t0) * 1000)
            payload.append(len(json.dumps(v)))
            items.append(len(v.get("nodes", v.get("cells", []))))
            levels.add(v["level"])
        res[label] = {"p50_ms": pct(times, 0.5), "p95_ms": pct(times, 0.95), "payload_kb_p95": round(pct(payload, 0.95) / 1024, 1),
                      "items_max": max(items), "level": sorted(levels)}
    st = []
    for _ in range(100):
        q = f"{rng.choice(WORDS)} {rng.choice(WORDS)}"
        t0 = time.perf_counter()
        s.search_lexical(q, limit=20)
        st.append((time.perf_counter() - t0) * 1000)
    one = []
    for _ in range(200):
        t0 = time.perf_counter()
        s.get_node(rng.randrange(1, n + 1))
        one.append((time.perf_counter() - t0) * 1000)
    # incremental edit cost: one node edit updates its record, FTS row and nothing else
    t0 = time.perf_counter()
    with s.edit(actor="measure", message="measure edit") as tx:
        tx.update_node(rng.randrange(1, n + 1), summary="Edited during measurement.")
    edit_ms = (time.perf_counter() - t0) * 1000
    s.undo()
    file_bytes = path.stat().st_size
    per_node = {g: round(v / n, 1) for g, v in sizes.items()}
    per_node["edges (edge_v + its indexes)"] = round(sizes["edges (edge_v + its indexes)"] / max(e, 1), 1)
    total_per_node = file_bytes / n
    report = {"synthetic": True, "warning": "SYNTHETIC fixture. Not a model; not trained; not counted as Yaob nodes.", "nodes": n, "edges": e,
              "file_bytes": file_bytes, "bytes_per_node_all_in": round(total_per_node, 1), "bytes_by_group": sizes,
              "bytes_per_node_by_group (edges: per edge)": per_node, "build": info, "viewport": res, "viewport_cap": cap,
              "lexical_search_ms": {"p50": pct(st, 0.5), "p95": pct(st, 0.95)}, "node_fetch_ms": {"p50": pct(one, 0.5), "p95": pct(one, 0.95)},
              "single_node_edit_ms": round(edit_ms, 2),
              "extrapolation_1e9_nodes": {
                  "assumption": "Same record shape (two short statements, one fact), same edge density, linear growth of rows and indexes. Real nodes are larger: the N10K store is measured separately.",
                  "store_bytes": int(total_per_node * 1e9), "store_tib": round(total_per_node * 1e9 / 2**40, 2),
                  "learned_state_bytes_f16": int(1e9 * (6 * 384 + 256 + 1) * 2), "learned_state_tib_f16": round(1e9 * (6 * 384 + 256 + 1) * 2 / 2**40, 2),
                  "statement_embeddings_bytes_f16": int(1e9 * 6 * 1024 * 2), "statement_embeddings_tib_f16": round(1e9 * 6 * 1024 * 2 / 2**40, 2),
                  "uncertainty": "A single SQLite file is not the right container at 1e9 rows. The numbers size the data, not that design: see docs/scale.md for the partitioned layout."}}
    print(json.dumps(report, indent=1))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("path")
    b.add_argument("--nodes", type=int, default=1_000_000)
    b.add_argument("--mpns", type=int, default=40)
    b.add_argument("--edges-per-node", type=float, default=3.0)
    b.add_argument("--seed", type=int, default=1)
    m = sub.add_parser("measure")
    m.add_argument("path")
    m.add_argument("--out")
    m.add_argument("--cap", type=int, default=2500)
    a = ap.parse_args()
    if a.cmd == "build":
        build(Path(a.path), a.nodes, a.mpns, a.edges_per_node, a.seed)
    else:
        measure(Path(a.path), Path(a.out) if a.out else None, a.cap)


if __name__ == "__main__":
    main()
