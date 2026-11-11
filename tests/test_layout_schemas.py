import random

import pytest

from mpnstudio.store import layout, schemas
from mpnstudio.store.db import Store


@pytest.fixture(scope="module")
def big(tmp_path_factory):
    s = Store(tmp_path_factory.mktemp("s") / "m.sqlite")
    schemas.seed_schemas(s)
    rng = random.Random(1)
    with s.edit(kind="import", message="synthetic") as tx:
        mp = [tx.put_mpn(f"m{i}", f"MPN {i}") for i in range(6)]
        doms = {m: [tx.put_domain(m, f"d{j}", f"Domain {j}") for j in range(4)] for m in mp}
        ids = []
        for i in range(1500):
            m = rng.choice(mp)
            n = tx.put_node(title=f"Node {i}", cells={"knowledge": {"statements": [{"text": f"Statement {i}."}]}})
            tx.set_membership(n, m, rng.choice(doms[m]))
            if i % 12 == 0:
                tx.set_membership(n, rng.choice([x for x in mp if x != m]))
            ids.append(n)
        for _ in range(3000):
            a, b = rng.sample(ids, 2)
            tx.put_edge(a, b, "related_to")
    s.ids = ids
    s.layout_info = layout.compute_layout(s)
    return s


def test_every_live_node_has_a_position(big):
    assert big.layout_info["nodes"] == 1500
    assert big.q1("SELECT COUNT(*) FROM layout_rtree")[0] == 1500


def test_nodes_do_not_pile_up(big):
    pts = [(r[0], r[1]) for r in big.q("SELECT x, y FROM layout")]
    cells = {}
    for x, y in pts:
        cells.setdefault((round(x / 6), round(y / 6)), 0)
        cells[(round(x / 6), round(y / 6))] += 1
    assert max(cells.values()) <= 3


def test_layout_is_deterministic(big):
    before = {r[0]: (r[1], r[2]) for r in big.q("SELECT node_id, x, y FROM layout")}
    layout.compute_layout(big)
    after = {r[0]: (r[1], r[2]) for r in big.q("SELECT node_id, x, y FROM layout")}
    assert before == after


def test_viewport_is_bounded_and_switches_level(big):
    b = layout.bounds(big)
    far = layout.viewport(big, 1, b, scale=0.02, cap=400)
    assert far["level"] == "cells" and len(far["cells"]) <= 400 and sum(c["n"] for c in far["cells"]) <= 1500
    assert any(r["kind"] == "mpn" for r in far["regions"])
    x, y = big.q1("SELECT x, y FROM layout LIMIT 1")
    near = layout.viewport(big, 1, [x - 200, y - 200, x + 200, y + 200], scale=2.0, cap=400)
    assert near["level"] == "nodes" and 0 < len(near["nodes"]) <= 400
    assert all(isinstance(n["id"], str) for n in near["nodes"])


def test_move_is_geometry_only(big):
    head = big.head()
    nid = big.ids[0]
    layout.move_nodes(big, 1, [(nid, 12345.0, -777.0)])
    assert layout.node_position(big, 1, nid) == (12345.0, -777.0)
    assert big.head() == head                       # no model revision
    layout.compute_layout(big)                      # pinned position survives a re-layout
    assert layout.node_position(big, 1, nid) == (12345.0, -777.0)


def test_new_node_gets_a_free_spot(big):
    with big.edit() as tx:
        n = tx.put_node(title="Fresh", cells={"knowledge": {"statements": [{"text": "New."}]}})
        tx.set_membership(n, 1)
    x, y = layout.place_node(big, 1, n)
    near = big.q1("SELECT COUNT(*) FROM layout_rtree WHERE maxx>=? AND minx<=? AND maxy>=? AND miny<=? AND id<>?", (x - 5, x + 5, y - 5, y + 5, n))[0]
    assert near == 0


def test_bundles_report_shared_nodes(big):
    b = layout.mpn_bundles(big)
    assert b and sum(x["shared"] for x in b) == big.counts()["multi_membership_nodes"]


def test_schema_relabel_is_compatible_but_role_change_is_not():
    import copy
    a = schemas.S3CN
    b = copy.deepcopy(a)
    b["roles"][1]["label"] = "Strategy"
    d = schemas.diff_schemas(a, b)
    assert not d["breaking"] and d["changes"][0]["kind"] == "relabelled"
    c = copy.deepcopy(a)
    c["roles"][1]["runtime_role"] = "reasoning"
    assert schemas.validate_schema(c)               # two reasoning cells is invalid
    c["roles"][2]["runtime_role"] = "none"
    d = schemas.diff_schemas(a, c)
    assert d["breaking"] and d["retrain_required"]


def test_schema_validation_and_migration(big):
    s3 = schemas.get_schema(big, "3cn")
    s5 = schemas.get_schema(big, "5cn")
    assert schemas.validate_schema(s3) == [] and schemas.validate_schema(s5) == []
    cells = {"knowledge": {"statements": [{"text": "A."}]}, "thinking": {}, "reasoning": {}}
    assert schemas.validate_cells(s3, cells) == []
    assert schemas.validate_cells(s3, {"knowledge": {}}) != []
    assert schemas.validate_cells(s3, {**cells, "interface": {}}) != []
    moved = schemas.migrate_cells(s3, s5, cells)
    assert set(moved) == {"knowledge", "thinking", "reasoning", "provenance", "interface"} and schemas.validate_cells(s5, moved) == []
    d = schemas.diff_schemas(s3, s5)
    assert not d["breaking"] and {c["kind"] for c in d["changes"]} == {"cell_added"}


def test_used_schema_version_cannot_change_structure(big):
    import copy
    s3 = copy.deepcopy(schemas.get_schema(big, "3cn"))
    s3["roles"][0]["fields"] = s3["roles"][0]["fields"][:1]
    with pytest.raises(ValueError):
        schemas.save_schema(big, s3)
    s3 = copy.deepcopy(schemas.get_schema(big, "3cn"))
    s3["roles"][0]["label"] = "Facts"
    assert schemas.save_schema(big, s3)["saved"]


def test_incremental_counts_match_a_full_recount(big):
    import random as _r
    rng = _r.Random(5)
    ids = big.ids
    for step in range(30):
        with big.edit() as tx:
            n = rng.choice(ids)
            op = rng.randrange(5)
            cur = tx.c.execute("SELECT status FROM node_v WHERE node_id=? AND rev_to IS NULL", (n,)).fetchone()[0]
            if op == 0 and cur == "live":
                tx.delete_node(n)
            elif op == 1 and cur == "tombstone":
                tx.update_node(n, status="live")
            elif op == 2:
                try:
                    tx.set_membership(n, rng.randrange(1, 7))
                except Exception:
                    pass
            elif op == 3:
                ms = tx.c.execute("SELECT mpn_id FROM member_v WHERE node_id=? AND rev_to IS NULL", (n,)).fetchall()
                if len(ms) > 1:
                    tx.remove_membership(n, ms[-1][0])
            else:
                tx.update_node(n, status="disabled" if cur == "live" else "live")
        if step % 7 == 0:
            big.undo()
    inc = {k: big.summary_counts(k) for k in ("mpn", "domain")}
    big.rebuild_summaries()
    full = {k: big.summary_counts(k) for k in ("mpn", "domain")}
    strip = lambda d: {k: v for k, v in d.items() if v != (0, 0)}
    assert strip(inc["mpn"]) == strip(full["mpn"]) and strip(inc["domain"]) == strip(full["domain"])


def test_tile_pyramid_tracks_moves(big):
    total = big.q1("SELECT COUNT(*) FROM layout")[0]
    for level in (0, 3, 9):
        assert big.q1("SELECT SUM(n) FROM tiles WHERE level=?", (level,))[0] == total
    nid = big.ids[5]
    layout.move_nodes(big, 1, [(nid, -99999.0, 88888.0)])
    for level in (0, 3, 9):
        assert big.q1("SELECT SUM(n) FROM tiles WHERE level=?", (level,))[0] == total
    size = layout.TILE_BASE
    import math
    assert big.q1("SELECT n FROM tiles WHERE level=0 AND gx=? AND gy=?", (math.floor(-99999.0 / size), math.floor(88888.0 / size)))[0] == 1
    far = layout.viewport(big, 1, [-100500, 88000, -99500, 89500], scale=0.01)
    assert far["level"] == "cells" and sum(c["n"] for c in far["cells"]) == 1
