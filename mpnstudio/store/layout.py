"""Board geometry: stable hierarchical layout and bounded viewport queries.

Layout is hierarchical and deterministic, never a global force simulation over nodes:
  1. MPN regions are circles sized by node count, arranged by a small relaxation over the
     MPNs only (tens of objects), pulled together where MPNs share nodes or edges.
  2. Domains are circles packed inside their MPN; sub-domains inside their domain.
  3. Nodes fill their leaf region on a sunflower spiral.
  4. A node with shared memberships is placed in the gap between the regions it belongs to.

Geometry lives in `layout`, `layout_rtree` and `regions`. It is not versioned with the model:
moving a node changes no model semantics.

Viewport queries are bounded by construction: the server returns individual nodes only when
the viewport holds fewer than a cap; otherwise it returns grid aggregates. The browser never
receives the whole graph.
"""
from __future__ import annotations

import math
from collections import defaultdict

from .db import Store, now, sid

SPACING = 26.0            # world units between neighbouring nodes
GOLDEN = math.pi * (3 - math.sqrt(5))
FILL = 0.62               # fraction of a region's area that its children may occupy
MPN_GAP = 150.0           # corridor between MPN regions where shared nodes sit
TILE_BASE = SPACING * 4   # side of a level-0 aggregate cell, in world units
TILE_LEVELS = 16          # level l has cells of side TILE_BASE * 2^l
FLOOR = "(CAST({v} / {s} AS INTEGER) - ({v} / {s} < CAST({v} / {s} AS INTEGER)))"


def _tile_delta(c, x: float, y: float, sign: int) -> None:
    for level in range(TILE_LEVELS):
        size = TILE_BASE * (1 << level)
        c.execute("INSERT INTO tiles(level, gx, gy, n, sx, sy) VALUES(?,?,?,?,?,?) "
                  "ON CONFLICT(level, gx, gy) DO UPDATE SET n=n+excluded.n, sx=sx+excluded.sx, sy=sy+excluded.sy",
                  (level, math.floor(x / size), math.floor(y / size), sign, sign * x, sign * y))


def rebuild_tiles(c, board_id: int) -> None:
    c.execute("DELETE FROM tiles")
    for level in range(TILE_LEVELS):
        size = TILE_BASE * (1 << level)
        gx, gy = FLOOR.format(v="x", s=size), FLOOR.format(v="y", s=size)
        c.execute(f"INSERT INTO tiles(level, gx, gy, n, sx, sy) SELECT ?, {gx}, {gy}, COUNT(*), SUM(x), SUM(y) FROM layout WHERE board_id=? GROUP BY 2, 3",
                  (level, board_id))


def leaf_radius(n: int) -> float:
    return SPACING * math.sqrt(max(n, 1) / math.pi) * 1.08 + SPACING * 0.6


def relax_circles(items: list[dict], iters: int = 400, gap: float = 0.0, attract: dict | None = None, contain: float | None = None) -> None:
    """Separate overlapping circles in place. `items` have x, y, r. Optional pairwise attraction and a containing radius."""
    n = len(items)
    if n == 0:
        return
    for it in range(iters):
        cool = 1.0 - it / iters
        moved = 0.0
        for i in range(n):
            a = items[i]
            fx = fy = 0.0
            for j in range(n):
                if i == j:
                    continue
                b = items[j]
                dx, dy = a["x"] - b["x"], a["y"] - b["y"]
                d = math.hypot(dx, dy) or 1e-6
                need = a["r"] + b["r"] + gap
                if d < need:
                    push = (need - d) / d * 0.5
                    fx += dx * push
                    fy += dy * push
                elif attract:
                    w = attract.get((min(a["key"], b["key"]), max(a["key"], b["key"])), 0.0)
                    if w:
                        pull = min(0.06 * w * cool, 0.2) * (d - need) / d
                        fx -= dx * pull
                        fy -= dy * pull
            if contain is None:
                fx -= a["x"] * 0.01 * cool           # weak gravity keeps the arrangement compact
                fy -= a["y"] * 0.01 * cool
            a["x"] += fx
            a["y"] += fy
            if contain is not None:
                d = math.hypot(a["x"], a["y"])
                lim = max(contain - a["r"], 0.0)
                if d > lim:
                    s = lim / (d or 1e-6)
                    a["x"] *= s
                    a["y"] *= s
            moved += abs(fx) + abs(fy)
        if moved < 1e-3 * n:
            break


def ring_init(items: list[dict], radius: float) -> None:
    n = len(items)
    for i, it in enumerate(items):
        ang = 2 * math.pi * i / max(n, 1)
        it["x"], it["y"] = (radius * math.cos(ang), radius * math.sin(ang)) if n > 1 else (0.0, 0.0)


def sunflower(n: int, radius: float) -> list[tuple[float, float]]:
    if n == 1:
        return [(0.0, 0.0)]
    c = radius / math.sqrt(n)
    return [(c * math.sqrt(i + 0.5) * math.cos(i * GOLDEN), c * math.sqrt(i + 0.5) * math.sin(i * GOLDEN)) for i in range(n)]


def compute_layout(store: Store, board_id: int = 1, keep_pinned: bool = True) -> dict:
    """(Re)build the hierarchical layout for a board from the head revision."""
    c = store.conn()
    mpns = store.mpns()
    doms = store.domains()
    rows = store.q(
        "SELECT m.node_id, m.mpn_id, m.domain_id, m.role, n.title FROM member_v m JOIN node_v n ON n.node_id=m.node_id AND n.rev_to IS NULL "
        "WHERE m.rev_to IS NULL AND n.status<>'tombstone' ORDER BY n.title, m.node_id")
    primary: dict[int, tuple[int, int | None]] = {}
    shared: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        if r["role"] == "primary":
            primary[r["node_id"]] = (r["mpn_id"], r["domain_id"])
        else:
            shared[r["node_id"]].append(r["mpn_id"])
    order = [r["node_id"] for r in rows if r["role"] == "primary"]

    # ---- tree: mpn -> level-1 domain -> level-2 domain -> nodes (shared nodes are placed separately)
    dom_by_id = {int(d["id"]): d for d in doms}
    inside: dict[tuple, list[int]] = defaultdict(list)       # (mpn, d1, d2) -> node ids
    boundary: dict[tuple, list[int]] = defaultdict(list)     # (mpn, sorted shared mpns) -> node ids
    for nid in order:
        mid, did = primary[nid]
        if shared.get(nid):
            boundary[(mid, tuple(sorted(shared[nid])))].append(nid)
            continue
        d1 = d2 = None
        if did is not None and did in dom_by_id:
            d = dom_by_id[did]
            if d["parent_id"] is None:
                d1 = did
            else:
                d1, d2 = int(d["parent_id"]), did
        inside[(mid, d1, d2)].append(nid)

    tree: dict[int, dict] = {}
    for (mid, d1, d2), ids in inside.items():
        tree.setdefault(mid, {}).setdefault(d1, {})[d2] = ids

    regions: list[tuple] = []           # (kind, ref_id, cx, cy, r)
    pos: dict[int, tuple[float, float]] = {}
    mpn_items = []
    local: dict[int, dict] = {}         # mpn -> {"r": radius, "domains": [...], "nodes": {nid: (x, y)}}
    for m in mpns:
        mid = int(m["id"])
        d1_items = []
        for d1, subs in sorted(tree.get(mid, {}).items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
            leaves = []
            for d2, ids in sorted(subs.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
                leaves.append({"key": d2 or 0, "d2": d2, "ids": ids, "r": leaf_radius(len(ids)), "x": 0.0, "y": 0.0})
            if len(leaves) == 1:
                r1 = leaves[0]["r"]
            else:
                r1 = math.sqrt(sum(l["r"] ** 2 for l in leaves) / FILL) + SPACING
                ring_init(leaves, r1 * 0.5)
                relax_circles(leaves, 300, gap=SPACING * 0.5, contain=r1)
                r1 = max(r1, max(math.hypot(l["x"], l["y"]) + l["r"] for l in leaves))
            d1_items.append({"key": d1 or 0, "d1": d1, "leaves": leaves, "r": r1 + (SPACING * 0.4 if len(leaves) > 1 else 0), "x": 0.0, "y": 0.0})
        if not d1_items:
            R = leaf_radius(1) * 2
        elif len(d1_items) == 1:
            R = d1_items[0]["r"] + SPACING
        else:
            R = math.sqrt(sum(d["r"] ** 2 for d in d1_items) / FILL) + SPACING
            d1_items.sort(key=lambda d: -d["r"])
            ring_init(d1_items, R * 0.55)
            d1_items[0]["x"] = d1_items[0]["y"] = 0.0
            relax_circles(d1_items, 400, gap=SPACING * 0.8, contain=R)
            R = max(R, max(math.hypot(d["x"], d["y"]) + d["r"] for d in d1_items) + SPACING * 0.5)
        local[mid] = {"r": R, "d1": d1_items}
        mpn_items.append({"key": mid, "r": R, "x": 0.0, "y": 0.0, "n": m["node_count"]})

    # ---- arrange MPN circles: attraction where MPNs share nodes or edges
    aff: dict[tuple, float] = defaultdict(float)
    for (mid, others), ids in boundary.items():
        for o in others:
            aff[(min(mid, o), max(mid, o))] += len(ids)
    for r in store.q(
            "SELECT a.mpn_id AS ma, b.mpn_id AS mb, COUNT(*) AS n FROM edge_v e "
            "JOIN member_v a ON a.node_id=e.src AND a.rev_to IS NULL AND a.role='primary' "
            "JOIN member_v b ON b.node_id=e.dst AND b.rev_to IS NULL AND b.role='primary' "
            "WHERE e.rev_to IS NULL AND a.mpn_id<>b.mpn_id GROUP BY 1, 2"):
        aff[(min(r["ma"], r["mb"]), max(r["ma"], r["mb"]))] += r["n"] * 0.15
    top = max(aff.values(), default=1.0)
    attract = {k: v / top for k, v in aff.items()}
    mpn_items.sort(key=lambda m: m["key"])
    total_r = sum(m["r"] for m in mpn_items)
    ring_init(mpn_items, total_r / math.pi * 0.9)
    relax_circles(mpn_items, 900, gap=MPN_GAP, attract=attract)
    mpn_pos = {m["key"]: m for m in mpn_items}

    # ---- emit regions and node positions
    for m in mpn_items:
        mid = m["key"]
        regions.append(("mpn", mid, m["x"], m["y"], m["r"]))
        for d in local[mid]["d1"]:
            dx, dy = m["x"] + d["x"], m["y"] + d["y"]
            if d["d1"] is not None:
                regions.append(("domain", d["d1"], dx, dy, d["r"]))
            for l in d["leaves"]:
                lx, ly = dx + l["x"], dy + l["y"]
                if l["d2"] is not None:
                    regions.append(("domain", l["d2"], lx, ly, l["r"]))
                for nid, (px, py) in zip(l["ids"], sunflower(len(l["ids"]), l["r"] - SPACING * 0.6)):
                    pos[nid] = (lx + px, ly + py)

    # ---- shared nodes: in the corridor between their regions
    for (mid, others), ids in sorted(boundary.items()):
        a = mpn_pos[mid]
        tx = sum(mpn_pos[o]["x"] for o in others if o in mpn_pos) / len(others)
        ty = sum(mpn_pos[o]["y"] for o in others if o in mpn_pos) / len(others)
        dx, dy = tx - a["x"], ty - a["y"]
        d = math.hypot(dx, dy) or 1.0
        ux, uy = dx / d, dy / d
        o_r = sum(mpn_pos[o]["r"] for o in others if o in mpn_pos) / len(others)
        gap_mid = a["r"] + max((d - a["r"] - o_r) * 0.35, SPACING)       # nearer the primary region
        cx, cy = a["x"] + ux * gap_mid, a["y"] + uy * gap_mid
        rad = leaf_radius(len(ids)) * 0.9
        for nid, (px, py) in zip(ids, sunflower(len(ids), rad)):
            # squash along the axis between the regions so the cluster reads as a boundary band
            along, across = px * ux + py * uy, -px * uy + py * ux
            along *= 0.55
            across *= 1.5
            pos[nid] = (cx + along * ux - across * uy, cy + along * uy + across * ux)

    # ---- write
    with store._wlock:
        c.execute("BEGIN IMMEDIATE")
        try:
            c.execute("INSERT OR IGNORE INTO boards(id, name, created) VALUES(?,?,?)", (board_id, "Yaob network", now()))
            pinned = {}
            if keep_pinned:
                pinned = {r[0]: (r[1], r[2]) for r in c.execute("SELECT node_id, x, y FROM layout WHERE board_id=? AND pinned=1", (board_id,))}
            c.execute("DELETE FROM layout WHERE board_id=?", (board_id,))
            c.execute("DELETE FROM layout_rtree")       # one spatial index: the store holds a single board
            c.execute("DELETE FROM regions WHERE board_id=?", (board_id,))
            c.executemany("INSERT INTO regions VALUES(?,?,?,?,?,?)", [(board_id, k, i, x, y, r) for k, i, x, y, r in regions])
            rows_l, rows_r = [], []
            for nid, (x, y) in pos.items():
                pin = 0
                if nid in pinned:
                    x, y = pinned[nid]
                    pin = 1
                rows_l.append((board_id, nid, x, y, pin))
                rows_r.append((nid, x, x, y, y))
            c.executemany("INSERT INTO layout VALUES(?,?,?,?,?)", rows_l)
            c.executemany("INSERT INTO layout_rtree VALUES(?,?,?,?,?)", rows_r)
            rebuild_tiles(c, board_id)
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise
    xs = [p[0] for p in pos.values()] or [0.0]
    ys = [p[1] for p in pos.values()] or [0.0]
    b = bounds(store, board_id)
    return {"nodes": len(pos), "regions": len(regions), "shared_nodes": sum(len(v) for v in boundary.values()),
            "bounds": b, "node_extent": [min(xs), min(ys), max(xs), max(ys)]}


def ensure_tiles(store: Store, board_id: int = 1) -> None:
    """Build the aggregate pyramid if a store has positions but no pyramid (stores laid out by an older version)."""
    if store.q1("SELECT 1 FROM tiles LIMIT 1") is None and store.q1("SELECT 1 FROM layout LIMIT 1") is not None:
        c = store.conn()
        with store._wlock:
            c.execute("BEGIN IMMEDIATE")
            try:
                rebuild_tiles(c, board_id)
                c.execute("COMMIT")
            except BaseException:
                c.execute("ROLLBACK")
                raise


def bounds(store: Store, board_id: int = 1) -> list[float]:
    r = store.q1("SELECT MIN(cx-r), MIN(cy-r), MAX(cx+r), MAX(cy+r) FROM regions WHERE board_id=? AND kind='mpn'", (board_id,))
    if r is None or r[0] is None:
        return [-1000.0, -1000.0, 1000.0, 1000.0]
    return [r[0], r[1], r[2], r[3]]


def region_list(store: Store, board_id: int, bbox: list[float] | None, kinds: tuple = ("mpn", "domain"), min_r: float = 0.0, rev: int | None = None) -> list[dict]:
    sql = "SELECT kind, ref_id, cx, cy, r FROM regions WHERE board_id=? AND r>=?"
    args: list = [board_id, min_r]
    if bbox:
        sql += " AND cx+r>=? AND cx-r<=? AND cy+r>=? AND cy-r<=?"
        args += [bbox[0], bbox[2], bbox[1], bbox[3]]
    rows = store.q(sql + " ORDER BY r DESC LIMIT 600", args)
    mp = {int(m["id"]): m for m in store.mpns(rev)}
    dm = {int(d["id"]): d for d in store.domains(rev=rev)} if any(r["kind"] == "domain" for r in rows) else {}
    out = []
    for r in rows:
        if r["kind"] not in kinds:
            continue
        if r["kind"] == "mpn":
            m = mp.get(r["ref_id"])
            if not m:
                continue
            out.append({"kind": "mpn", "id": sid(r["ref_id"]), "x": r["cx"], "y": r["cy"], "r": r["r"], "name": m["name"], "slug": m["slug"],
                        "count": m["node_count"], "shared": m["shared_count"], "enabled": m["enabled"], "installed": m["installed"]})
        else:
            d = dm.get(r["ref_id"])
            if not d:
                continue
            out.append({"kind": "domain", "id": sid(r["ref_id"]), "mpn_id": d["mpn_id"], "x": r["cx"], "y": r["cy"], "r": r["r"],
                        "name": d["name"], "count": d["node_count"], "depth": d["depth"]})
    return out


def viewport(store: Store, board_id: int, bbox: list[float], scale: float, cap: int = 2500, label_cap: int = 160) -> dict:
    """Bounded contents of a viewport.

    bbox = [minx, miny, maxx, maxy] in world units; scale = screen pixels per world unit.
    Returns either `nodes` (when few enough) or `cells` (grid aggregates), never more than `cap` items.
    """
    cap = max(100, min(cap, 6000))
    minx, miny, maxx, maxy = bbox

    def tiles_at(level: int, limit: int):
        size = TILE_BASE * (1 << level)
        return size, store.q("SELECT n, sx, sy FROM tiles WHERE level=? AND gx BETWEEN ? AND ? AND gy BETWEEN ? AND ? AND n>0 LIMIT ?",
                             (level, math.floor(minx / size), math.floor(maxx / size), math.floor(miny / size), math.floor(maxy / size), limit))

    # estimate how many nodes the viewport holds from the pyramid (a few hundred rows at most), never by scanning nodes
    span = max(maxx - minx, maxy - miny, 1.0)
    est_level = max(0, min(TILE_LEVELS - 1, math.ceil(math.log2(max(span / 24.0, TILE_BASE) / TILE_BASE))))
    _, est_rows = tiles_at(est_level, 2000)
    est = sum(r["n"] for r in est_rows)
    out: dict = {"total_in_view": est, "scale": scale}
    node_px = SPACING * scale
    rows = None
    if est <= cap * 1.5 and node_px >= 3.0:
        rows = store.q(
            "SELECT r.id, l.x, l.y, l.pinned, n.title, n.type, n.status, n.validation, n.origin, "
            "(SELECT mpn_id FROM member_v m WHERE m.node_id=r.id AND m.rev_to IS NULL AND m.role='primary') AS mpn, "
            "(SELECT COUNT(*) FROM member_v m WHERE m.node_id=r.id AND m.rev_to IS NULL) AS nm, "
            "(SELECT COUNT(*) FROM edge_v e WHERE e.dst=r.id AND e.rev_to IS NULL) AS indeg, "
            "d.embed_hash IS NOT NULL AND d.embed_hash = n.content_hash AS indexed, "
            "d.learned_hash IS NOT NULL AS has_learned, d.learned_hash IS NOT NULL AND d.learned_hash = n.content_hash AS learned_fresh "
            # CROSS JOIN fixes the join order: the spatial index first, then lookups by id. Left to itself the planner
            # may walk every node and probe the spatial index a million times.
            "FROM layout_rtree r CROSS JOIN layout l ON l.node_id=r.id AND l.board_id=? "
            "CROSS JOIN node_v n ON n.node_id=r.id AND n.rev_to IS NULL LEFT JOIN derived_state d ON d.node_id=r.id "
            "WHERE r.maxx>=? AND r.minx<=? AND r.maxy>=? AND r.miny<=? AND n.status<>'tombstone' LIMIT ?",
            (board_id, minx, maxx, miny, maxy, cap + 1))
        if len(rows) > cap:
            rows = None                        # denser than estimated: fall back to aggregates
    if rows is not None:
        nodes = [{"id": sid(r["id"]), "x": round(r["x"], 2), "y": round(r["y"], 2), "t": r["type"], "s": r["status"], "v": r["validation"],
                  "o": r["origin"], "m": sid(r["mpn"]), "nm": r["nm"], "deg": r["indeg"], "pin": r["pinned"],
                  "learned": "trained" if r["learned_fresh"] else ("stale" if r["has_learned"] else "none"),
                  "indexed": bool(r["indexed"])} for r in rows]
        by_deg = sorted(range(len(rows)), key=lambda i: -rows[i]["indeg"])
        show_all = node_px >= 60
        for i in (range(len(rows)) if show_all else by_deg[:label_cap]):
            nodes[i]["title"] = rows[i]["title"]
        out["level"] = "nodes"
        out["total_in_view"] = len(nodes)
        out["nodes"] = nodes
    else:
        # aggregates from the pyramid: cells about 28 screen pixels wide, coarsened until they fit the cap
        want = 28.0 / max(scale, 1e-9)
        level = max(0, min(TILE_LEVELS - 1, math.ceil(math.log2(max(want, TILE_BASE) / TILE_BASE))))
        while True:
            size, trows = tiles_at(level, cap + 1)
            if len(trows) <= cap or level >= TILE_LEVELS - 1:
                break
            level += 1
        out["level"] = "cells"
        out["cell_size"] = size
        out["cells"] = [{"x": round(r["sx"] / r["n"], 1), "y": round(r["sy"] / r["n"], 1), "n": r["n"], "m": None} for r in trows[:cap]]
    min_r = 8.0 / max(scale, 1e-9)       # regions smaller than 8 px on screen are not sent
    out["regions"] = region_list(store, board_id, bbox, min_r=min_r)
    return out


def viewport_edges(store: Store, node_ids: list[int], cap: int = 1500) -> list[dict]:
    """Edges among the given (visible) nodes, strongest first, bounded."""
    if not node_ids:
        return []
    ids = node_ids[:3000]
    ph = ",".join("?" * len(ids))
    rows = store.q(f"SELECT edge_id, src, dst, type, weight FROM edge_v WHERE rev_to IS NULL AND src IN ({ph}) AND dst IN ({ph}) "
                   f"ORDER BY weight DESC, edge_id LIMIT ?", (*ids, *ids, cap))
    return [{"id": sid(r["edge_id"]), "s": sid(r["src"]), "d": sid(r["dst"]), "t": r["type"], "w": r["weight"]} for r in rows]


def mpn_bundles(store: Store, rev: int | None = None) -> list[dict]:
    """Aggregated relations between MPNs: edge counts and shared-node counts. This is what the overview draws."""
    w, a = store.at(rev, "e")
    wa, aa = store.at(rev, "a")
    wb, ab = store.at(rev, "b")
    edges = store.q(
        f"SELECT a.mpn_id AS ma, b.mpn_id AS mb, COUNT(*) AS n FROM edge_v e "
        f"JOIN member_v a ON a.node_id=e.src AND {wa} AND a.role='primary' JOIN member_v b ON b.node_id=e.dst AND {wb} AND b.role='primary' "
        f"WHERE {w} AND a.mpn_id<>b.mpn_id GROUP BY 1, 2", (*aa, *ab, *a))
    agg: dict[tuple, dict] = {}
    for r in edges:
        k = (min(r["ma"], r["mb"]), max(r["ma"], r["mb"]))
        agg.setdefault(k, {"edges": 0, "shared": 0})["edges"] += r["n"]
    wm, am = store.at(rev, "p")
    ws, as_ = store.at(rev, "s")
    for r in store.q(
            f"SELECT p.mpn_id AS ma, s.mpn_id AS mb, COUNT(*) AS n FROM member_v p JOIN member_v s ON s.node_id=p.node_id AND {ws} AND s.role='shared' "
            f"WHERE {wm} AND p.role='primary' GROUP BY 1, 2", (*as_, *am)):
        k = (min(r["ma"], r["mb"]), max(r["ma"], r["mb"]))
        agg.setdefault(k, {"edges": 0, "shared": 0})["shared"] += r["n"]
    return [{"a": sid(k[0]), "b": sid(k[1]), **v} for k, v in sorted(agg.items(), key=lambda kv: -(kv[1]["edges"] + 20 * kv[1]["shared"]))]


def node_position(store: Store, board_id: int, node_id: int) -> tuple[float, float] | None:
    r = store.q1("SELECT x, y FROM layout WHERE board_id=? AND node_id=?", (board_id, node_id))
    return (r[0], r[1]) if r else None


def move_nodes(store: Store, board_id: int, moves: list[tuple[int, float, float]]) -> int:
    """Geometry-only change. Does not create a model revision."""
    c = store.conn()
    with store._wlock:
        c.execute("BEGIN IMMEDIATE")
        try:
            for nid, x, y in moves:
                old = c.execute("SELECT x, y FROM layout WHERE board_id=? AND node_id=?", (board_id, nid)).fetchone()
                if old is not None:
                    _tile_delta(c, old[0], old[1], -1)
                _tile_delta(c, x, y, 1)
                c.execute("INSERT INTO layout(board_id, node_id, x, y, pinned) VALUES(?,?,?,?,1) "
                          "ON CONFLICT(board_id, node_id) DO UPDATE SET x=excluded.x, y=excluded.y, pinned=1", (board_id, nid, x, y))
                c.execute("DELETE FROM layout_rtree WHERE id=?", (nid,))
                c.execute("INSERT INTO layout_rtree VALUES(?,?,?,?,?)", (nid, x, x, y, y))
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise
    return len(moves)


def place_node(store: Store, board_id: int, node_id: int, near: tuple[float, float] | None = None) -> tuple[float, float]:
    """Give a new node a free position: at `near` if given, else beside its primary domain or MPN region."""
    existing = node_position(store, board_id, node_id)
    if existing:
        return existing
    if near is None:
        m = store.q1("SELECT mpn_id, domain_id FROM member_v WHERE node_id=? AND rev_to IS NULL AND role='primary'", (node_id,))
        reg = None
        if m is not None:
            if m["domain_id"] is not None:
                reg = store.q1("SELECT cx, cy, r FROM regions WHERE board_id=? AND kind='domain' AND ref_id=?", (board_id, m["domain_id"]))
            if reg is None:
                reg = store.q1("SELECT cx, cy, r FROM regions WHERE board_id=? AND kind='mpn' AND ref_id=?", (board_id, m["mpn_id"]))
        near = (reg["cx"], reg["cy"]) if reg else (0.0, 0.0)
    x, y = near
    for i in range(1, 4000):        # walk a spiral outward until no node is within one spacing
        hit = store.q1("SELECT 1 FROM layout_rtree WHERE maxx>=? AND minx<=? AND maxy>=? AND miny<=? LIMIT 1",
                       (x - SPACING * 0.8, x + SPACING * 0.8, y - SPACING * 0.8, y + SPACING * 0.8))
        if hit is None:
            break
        rad = SPACING * 0.75 * math.sqrt(i)
        x, y = near[0] + rad * math.cos(i * GOLDEN), near[1] + rad * math.sin(i * GOLDEN)
    c = store.conn()
    with store._wlock:
        c.execute("INSERT OR REPLACE INTO layout(board_id, node_id, x, y, pinned) VALUES(?,?,?,?,0)", (board_id, node_id, x, y))
        c.execute("DELETE FROM layout_rtree WHERE id=?", (node_id,))
        c.execute("INSERT INTO layout_rtree VALUES(?,?,?,?,?)", (node_id, x, x, y, y))
        _tile_delta(c, x, y, 1)
    return x, y


def remove_node(store: Store, board_id: int, node_id: int) -> None:
    c = store.conn()
    with store._wlock:
        old = c.execute("SELECT x, y FROM layout WHERE board_id=? AND node_id=?", (board_id, node_id)).fetchone()
        if old is not None:
            _tile_delta(c, old[0], old[1], -1)
        c.execute("DELETE FROM layout WHERE board_id=? AND node_id=?", (board_id, node_id))
        c.execute("DELETE FROM layout_rtree WHERE id=?", (node_id,))
