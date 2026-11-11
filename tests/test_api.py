import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "dev-fixture.sqlite"
pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="run scripts/dev_fixture.py first")


@pytest.fixture
def api(tmp_path, monkeypatch):
    db = tmp_path / "m.sqlite"
    shutil.copy(FIXTURE, db)
    monkeypatch.setenv("MPN_DATA", str(tmp_path))
    monkeypatch.setenv("MPN_DB", str(db))
    monkeypatch.setenv("MPN_NO_SYSTEMD", "1")
    monkeypatch.setenv("MPN_RUNTIME_URL", "http://127.0.0.1:9")      # nothing listens: runtime down
    import importlib

    from mpnstudio.api import app as appmod
    importlib.reload(appmod)
    c = TestClient(appmod.create_app(db), base_url="http://testserver")
    c.store = c.app.state.store
    return c


def first_node(api, q="Synthetic topic 7"):
    return api.get("/api/search", params={"q": q, "mode": "lexical"}).json()["results"][0]


def test_status_and_board(api):
    s = api.get("/api/status").json()
    assert s["counts"]["canonical_live_nodes"] == 1240 and s["runtime"]["available"] is False
    b = api.get("/api/board").json()
    assert len(b["mpns"]) == 18 and len(b["bounds"]) == 4 and b["bundles"]


def test_viewport_overview_then_nodes(api):
    b = api.get("/api/board").json()["bounds"]
    far = api.post("/api/viewport", json={"bbox": b, "scale": 0.05, "cap": 500}).json()
    assert far["level"] == "cells" and len(far["cells"]) <= 500
    m = api.get("/api/board").json()["mpns"][0]
    near = api.post("/api/viewport", json={"bbox": [m["x"] - 150, m["y"] - 150, m["x"] + 150, m["y"] + 150], "scale": 3, "edges": True}).json()
    assert near["level"] == "nodes" and "edges" in near


def test_search_jump_inspect_edit_persists(api):
    hit = first_node(api)
    assert "x" in hit and "mpn" in hit                                   # enough to jump there without loading the board
    n = api.get(f"/api/nodes/{hit['id']}").json()
    assert set(n["cells"]) == {"knowledge", "thinking", "reasoning"} and n["sources"] and n["state"]["index"] == "missing"
    cells = n["cells"]
    cells["knowledge"]["statements"][0]["text"] = "Edited statement about marigolds."
    r = api.patch(f"/api/nodes/{hit['id']}", json={"cells": cells, "summary": "Edited."}).json()
    assert r["changed"] and r["effects"]["content_changed"] and "trained node state" in r["effects"]["after_training"]
    assert api.get("/api/search", params={"q": "marigolds", "mode": "lexical"}).json()["results"][0]["id"] == hit["id"]
    # a second process opening the same file sees it (restart persistence)
    from mpnstudio.store.db import Store
    assert Store(api.store.path).get_node(int(hit["id"]))["summary"] == "Edited."


def test_invalid_edit_is_refused_with_reasons(api):
    hit = first_node(api)
    r = api.patch(f"/api/nodes/{hit['id']}", json={"cells": {"knowledge": {"statements": "not a list"}}})
    assert r.status_code == 422 and r.json()["detail"]["problems"]
    r = api.patch(f"/api/nodes/{hit['id']}", json={"cells": {"knowledge": {"statements": [{"text": "x"}]}, "thinking": {"impl": "os.system"}}})
    assert r.status_code == 422 and "not a built-in procedure" in r.json()["detail"]["problems"][0]


def test_create_connect_share_unshare_undo_restore(api):
    mp = api.get("/api/mpns").json()["mpns"]
    hist = next(m for m in mp if m["slug"] == "history")
    geo = next(m for m in mp if m["slug"] == "geography-geopolitics")
    before = api.get("/api/status").json()["counts"]["canonical_live_nodes"]
    r = api.post("/api/nodes", json={"title": "Congress of Vienna", "summary": "1814-15 conference that redrew European borders.",
                                     "memberships": [{"mpn_id": hist["id"]}, {"mpn_id": geo["id"]}],
                                     "cells": {"knowledge": {"statements": [{"text": "The Congress of Vienna met from 1814 to 1815."}]}, "thinking": {}, "reasoning": {}},
                                     "sources": [{"title": "Editor's note", "kind": "user"}]}).json()
    nid = r["id"]
    assert [m["role"] for m in r["node"]["memberships"]] == ["primary", "shared"] and r["node"]["position"]
    assert api.get("/api/status").json()["counts"]["canonical_live_nodes"] == before + 1      # one node, two memberships
    other = first_node(api)["id"]
    e = api.post("/api/edges", json={"src": nid, "dst": other, "type": "related_to"}).json()
    assert api.get(f"/api/nodes/{nid}/edges").json()["total"] == 1
    r = api.delete(f"/api/nodes/{nid}/memberships/{hist['id']}").json()
    assert r["node_status"] == "live" and [m["mpn_slug"] for m in r["memberships"]] == ["geography-geopolitics"]
    assert api.post("/api/undo").json()["undone"]
    assert len(api.get(f"/api/nodes/{nid}").json()["memberships"]) == 2
    assert api.delete(f"/api/nodes/{nid}").json()["changed"]
    assert nid not in [r["id"] for r in api.get("/api/search", params={"q": "Congress of Vienna", "mode": "lexical"}).json()["results"]]
    assert api.get(f"/api/nodes/{nid}").json()["status"] == "tombstone"
    assert api.post(f"/api/nodes/{nid}/restore").json()["changed"]
    assert api.get(f"/api/nodes/{nid}").json()["status"] == "live" and api.get(f"/api/nodes/{nid}/edges").json()["total"] == 1
    assert api.delete(f"/api/edges/{e['id']}").status_code == 200


def test_duplicate_reference_vs_canonical(api):
    hit = first_node(api)
    before = api.get("/api/status").json()["counts"]["canonical_live_nodes"]
    r = api.post(f"/api/nodes/{hit['id']}/duplicate", json={"mode": "reference"}).json()
    assert r["node_id"] == hit["id"] and api.get("/api/status").json()["counts"]["canonical_live_nodes"] == before
    r = api.post(f"/api/nodes/{hit['id']}/duplicate", json={"mode": "canonical"}).json()
    assert r["id"] != hit["id"] and api.get("/api/status").json()["counts"]["canonical_live_nodes"] == before + 1
    assert any(e["type"] == "derived_from" for e in r["node"]["edges"])


def test_move_does_not_create_a_revision(api):
    hit = first_node(api)
    head = api.get("/api/health").json()["head"]
    r = api.post("/api/layout/move", json={"moves": [{"id": hit["id"], "x": 5.0, "y": 6.0}]}).json()
    assert r["moved"] == 1 and api.get("/api/health").json()["head"] == head
    assert api.get(f"/api/nodes/{hit['id']}").json()["position"] == {"x": 5.0, "y": 6.0}


def test_mpn_disable_uninstall_install_roundtrip(api):
    bio = next(m for m in api.get("/api/mpns").json()["mpns"] if m["slug"] == "biology")
    counts = api.get("/api/status").json()["counts"]
    imp = api.get(f"/api/mpns/{bio['id']}/impact", params={"action": "uninstall"}).json()
    assert imp["nodes_only_here"] > 0 and imp["shared_nodes"] > 0 and "tombstoned" in imp["explanation"]
    assert api.post(f"/api/mpns/{bio['id']}/disable").json()["mpn"]["enabled"] is False
    assert api.post(f"/api/mpns/{bio['id']}/enable").json()["mpn"]["enabled"] is True
    api.post(f"/api/mpns/{bio['id']}/uninstall")
    after = api.get("/api/status").json()["counts"]
    assert after["canonical_live_nodes"] == counts["canonical_live_nodes"] - imp["nodes_only_here"]     # shared nodes survive
    assert after["tombstones"] == imp["nodes_only_here"]
    assert api.post(f"/api/mpns/{bio['id']}/install").json()["mpn"]["installed"] is True
    assert api.get("/api/status").json()["counts"]["canonical_live_nodes"] == counts["canonical_live_nodes"]


def test_schema_editor_flow(api):
    s = api.get("/api/schemas").json()["schemas"]
    s3 = next(x for x in s if x["id"] == "3cn")["spec"]
    renamed = dict(s3, roles=[dict(r, label="Strategy") if r["key"] == "thinking" else r for r in s3["roles"]])
    v = api.post("/api/schemas/validate", json=renamed).json()
    assert v["problems"] == [] and v["diff"]["breaking"] is False and "Runtime semantics unchanged" in v["diff"]["changes"][0]["effect"]
    bad = dict(s3, roles=[dict(r, runtime_role="none") if r["key"] == "thinking" else r for r in s3["roles"]])
    assert api.post("/api/schemas/validate", json=bad).json()["diff"]["breaking"] is True
    assert api.put("/api/schemas", json=bad).status_code == 400
    prev = api.get("/api/schemas/migration", params={"from_id": "3cn", "from_version": 1, "to_id": "5cn", "to_version": 1}).json()
    assert prev["can_apply"] and prev["nodes"] == 1240
    hit = first_node(api)
    r = api.post("/api/schemas/migrate", json={"from_id": "3cn", "from_version": 1, "to_id": "5cn", "to_version": 1, "node_ids": [hit["id"]]}).json()
    assert r["migrated"] == 1
    n = api.get(f"/api/nodes/{hit['id']}").json()
    assert n["schema"]["id"] == "5cn" and len(n["schema"]["roles"]) == 5 and n["problems"] == []


def test_jobs_api_and_log(api):
    import time

    from mpnstudio.train import jobs
    jid = api.post("/api/jobs", json={"kind": "sleep", "params": {"steps": 2, "step_s": 0.05, "gpu": False, "memory_estimate_gib": 0}}).json()["id"]
    jobs.worker_loop(once=True)
    for _ in range(100):
        j = api.get(f"/api/jobs/{jid}").json()
        if j["state"] == "succeeded":
            break
        time.sleep(0.1)
    assert j["state"] == "succeeded" and "step 2/2" in api.get(f"/api/jobs/{jid}/log").json()["text"]


def test_runtime_down_is_reported_not_faked(api):
    r = api.post("/api/run", json={"messages": [{"role": "user", "content": "hi"}]})
    assert "event: error" in r.text and "runtime is not running" in r.text
    assert api.post("/v1/chat/completions", json={"messages": []}).status_code == 503
    assert "text only" in api.get("/api/search", params={"q": "synthetic"}).json()["notice"]


def test_budget_validation(api):
    b = api.get("/api/budget").json()
    assert set(b["presets"]) == {"conservative", "balanced", "expanded", "specialists (experimental)"}
    assert b["budget"]["max_active_adapters"] == 0 and b["budget"]["context_tokens"] == 8192 and b["budget"]["output_tokens"] == 1024
    assert api.put("/api/budget", json={"selected_nodes": 2, "text_nodes": 5}).status_code == 400
    assert api.put("/api/budget", json={"preset": "conservative", "apply_preset": True}).json()["budget"]["selected_nodes"] == 4
    # an answer cap that leaves no room for the question is refused when saving (the 4,096 / 4,096 report)
    r = api.put("/api/budget", json={"context_tokens": 4096, "output_tokens": 4096})
    assert r.status_code == 400 and "no room for the question" in r.text
    # explicit zero survives saving and reading back; the experimental preset turns one specialist on
    assert api.put("/api/budget", json={"max_active_adapters": 0}).json()["budget"]["max_active_adapters"] == 0
    assert api.get("/api/budget").json()["budget"]["max_active_adapters"] == 0
    exp = api.put("/api/budget", json={"preset": "specialists (experimental)", "apply_preset": True}).json()["budget"]
    assert exp["max_active_adapters"] == 1
    assert api.put("/api/budget", json={"preset": "balanced", "apply_preset": True}).json()["budget"]["max_active_adapters"] == 0


def test_cross_site_write_is_refused(api):
    r = api.post("/api/undo", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
