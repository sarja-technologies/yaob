import json
import shutil
from pathlib import Path

import pytest

from mpnstudio.corpus import canary
from mpnstudio.store import bundle
from mpnstudio.store.db import Store

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "dev-fixture.sqlite"
pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="run scripts/dev_fixture.py first")


@pytest.fixture
def src(tmp_path):
    db = tmp_path / "src.sqlite"
    shutil.copy(FIXTURE, db)
    s = Store(db)
    canary.install(s, 4)
    return s


def test_bundle_round_trip_preserves_identity_counts_and_geometry(src, tmp_path):
    m = bundle.export_bundle(src, tmp_path / "b", "yaob-test")
    assert m["counts"]["canonical_live_nodes"] == 1240 and m["counts"]["diagnostic_nodes"] == 16 and len(m["packages"]) == 19
    r = bundle.import_bundle(tmp_path / "b", tmp_path / "dst.sqlite")
    assert r["matches_bundle"], r
    dst = Store(tmp_path / "dst.sqlite")
    a, b = src.counts(), dst.counts()
    for k in ("canonical_live_nodes", "diagnostic_nodes", "memberships", "multi_membership_nodes", "edges", "mpns"):
        assert a[k] == b[k], k
    # same canonical identity, same content hash, same position
    row = src.q1("SELECT n.uid, n.content_hash, l.x, l.y FROM node_v n JOIN layout l ON l.node_id=n.node_id WHERE n.rev_to IS NULL AND n.title='Synthetic topic 77'")
    other = dst.q1("SELECT n.content_hash, l.x, l.y FROM node_v n JOIN layout l ON l.node_id=n.node_id WHERE n.rev_to IS NULL AND n.uid=?", (row["uid"],))
    assert tuple(other) == (row["content_hash"], row["x"], row["y"])
    assert dst.mpn_by_slug("diagnostic-canary")["enabled"] is False            # disabled state survives
    assert dst.search_lexical("pythagorean")[0]["title"] == "Pythagorean theorem"
    assert dst.q1("SELECT SUM(n) FROM tiles WHERE level=0")[0] == dst.q1("SELECT COUNT(*) FROM layout")[0]


def test_tampered_package_is_refused(src, tmp_path):
    mid = int(src.mpn_by_slug("history")["id"])
    bundle.export_mpn(src, mid, tmp_path / "pkg")
    p = tmp_path / "pkg" / "nodes.jsonl"
    p.write_text(p.read_text().replace("Synthetic", "Tampered", 1))
    with pytest.raises(ValueError, match="checksum"):
        bundle.import_mpn(Store(tmp_path / "x.sqlite"), tmp_path / "pkg")


def test_package_cannot_bring_code(src, tmp_path):
    mid = int(src.mpn_by_slug("mathematics-statistics")["id"])
    bundle.export_mpn(src, mid, tmp_path / "pkg")
    p = tmp_path / "pkg" / "nodes.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    proc = next(r for r in rows if r["type"] == "procedure")
    proc["cells"]["thinking"]["impl"] = "os.system"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    man = json.loads((tmp_path / "pkg" / "manifest.json").read_text())
    man["files"]["nodes.jsonl"] = {"sha256": bundle.sha256_file(p), "bytes": p.stat().st_size}       # even with a matching checksum
    (tmp_path / "pkg" / "manifest.json").write_text(json.dumps(man))
    from mpnstudio.store import schemas
    dst = Store(tmp_path / "x.sqlite")
    schemas.seed_schemas(dst)
    with pytest.raises(ValueError, match="cannot bring code"):
        bundle.import_mpn(dst, tmp_path / "pkg")
    assert dst.counts()["canonical_live_nodes"] == 0


def test_importing_a_package_twice_does_not_duplicate_nodes(src, tmp_path):
    mid = int(src.mpn_by_slug("history")["id"])
    m = bundle.export_mpn(src, mid, tmp_path / "pkg")
    from mpnstudio.store import schemas
    dst = Store(tmp_path / "x.sqlite")
    schemas.seed_schemas(dst)
    r1 = bundle.import_mpn(dst, tmp_path / "pkg")
    n1 = dst.counts()["canonical_live_nodes"]
    r2 = bundle.import_mpn(dst, tmp_path / "pkg")
    assert r1["nodes_created"] == m["counts"]["nodes"] and r2["nodes_created"] == 0 and dst.counts()["canonical_live_nodes"] == n1
    assert m["installs_code"] is False and m["dependencies"]
