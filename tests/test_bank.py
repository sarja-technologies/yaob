"""Router packaging and MPN residency (needs torch; CPU is enough)."""
import json

import pytest

torch = pytest.importorskip("torch")

from mpnstudio.model.bank import MpnResidency, SharedRouter, package_router  # noqa: E402
from mpnstudio.model.router import Router, RouterSpec, router_loss  # noqa: E402


@pytest.fixture
def packed(tmp_path):
    torch.manual_seed(0)
    spec = RouterSpec(n_rows=60, n_mpn=3, d_enc=32, d_key=8)
    r = Router(spec)
    with torch.no_grad():
        r.key_delta.normal_(0, 0.1)
        r.gate_bias.normal_(0, 0.5)
    node_rows = {1000 + i: i for i in range(60)}
    primary = {1000 + i: 1 + i % 3 for i in range(60)}          # MPN ids 1..3
    shared = {1000: [2], 1001: [3]}
    mpn_index = {1: 0, 2: 1, 3: 2}
    node_emb = torch.nn.functional.normalize(torch.randn(60, 32), dim=-1)
    hashes = {n: f"h{n}" for n in node_rows}
    manifest = package_router(tmp_path, r, node_rows, hashes, primary, shared, mpn_index, node_emb)
    return tmp_path, r, spec, manifest, node_emb, node_rows


def test_param_counts_are_exact():
    spec = RouterSpec(n_rows=60, n_mpn=3, d_enc=32, d_key=8)
    pc = Router(spec).param_counts()
    assert pc["per_node_each"] == spec.d_key + 1
    assert pc["per_node_total"] == spec.n_rows * pc["per_node_each"]
    assert pc["per_mpn_total"] == spec.n_mpn * spec.d_key
    assert pc["shared"] + pc["per_node_total"] + pc["per_mpn_total"] == pc["total"]


def test_router_loss_has_a_gradient_into_per_node_rows():
    torch.manual_seed(1)
    spec = RouterSpec(n_rows=20, n_mpn=2, d_enc=16, d_key=8)
    r = Router(spec)
    B, K = 6, 3
    q = torch.randn(B, 16)
    pos = torch.arange(B)
    neg = torch.randint(0, 20, (B, K))
    emb = torch.randn(20, 16)
    loss, parts = router_loss(r, q, pos, emb[pos], neg, emb[neg], torch.zeros(B, 2).scatter_(1, (pos % 2)[:, None], 1))
    loss.backward()
    assert r.key_delta.grad[pos].abs().sum() > 0 and r.gate_bias.grad[pos].abs().sum() > 0
    assert set(parts) == {"nce", "mpn", "gate"}


def test_shards_partition_nodes_and_keep_shared_members(packed):
    d, _, _, manifest, _, _ = packed
    assert sum(s["nodes"] for s in manifest["shards"].values()) == 60
    assert manifest["shards"]["2"]["shared_members"] == 1 and manifest["shards"]["3"]["shared_members"] == 1
    assert all(len(s["sha256"]) == 64 for s in manifest["shards"].values())


def test_packaged_keys_equal_the_training_module(packed):
    d, r, spec, _, node_emb, node_rows = packed
    shared = SharedRouter(d, "cpu")
    cache = MpnResidency(d, 10**9)
    shard, ev = cache.acquire(2)
    assert ev["state"] == "loaded now" and ev["adapter"] is False
    nid = int(shard.tensors["node_ids"][3])
    row = node_rows[nid]
    with torch.no_grad():
        want = r.node_keys(torch.tensor([row]), node_emb[row][None])[0]
        q = torch.randn(32)
        want_mpn = r.mpn_logits(r.query_keys(q[None]))
        got_mpn = shared.router.mpn_logits(shared.router.query_keys(q[None]))
    assert torch.allclose(want, shard.tensors["keys"][3].float(), atol=2e-3)          # float16 storage
    assert torch.allclose(want_mpn, got_mpn, atol=1e-5)
    assert float(shard.tensors["gate_bias"][3]) == pytest.approx(float(r.gate_bias[row]), abs=1e-6)
    # shared members carry their home MPN, so the runtime can find the full entry
    assert shard.tensors["member_ids"].tolist() == [1000] and shard.tensors["member_home"].tolist() == [1]


def test_residency_respects_budget_pins_and_reports(packed):
    d, _, _, manifest, _, _ = packed
    one = manifest["shards"]["1"]["bytes"]
    cache = MpnResidency(d, int(one * 2.3))
    a, ea = cache.acquire(1)
    b, eb = cache.acquire(2)
    assert ea["state"] == eb["state"] == "loaded now" and len(cache.residency()) == 2
    c, ec = cache.acquire(3)                           # both others are pinned: nothing can be evicted
    assert c is None and ec["state"] == "over budget"
    cache.release(1)
    c, ec = cache.acquire(3)                           # now the unpinned, least recently used MPN goes
    assert c is not None and ec["evicted"] == [1] and {r["mpn_id"] for r in cache.residency()} == {"2", "3"}
    again, e2 = cache.acquire(2)
    assert e2["state"] == "already resident"
    cache.release(2); cache.release(2); cache.release(3)
    assert cache.set_limits(one // 2, 8) and cache.bytes() == 0        # shrinking the budget evicts
    assert cache.acquire(1)[1]["state"] == "over budget"
    assert cache.acquire(99)[1]["state"] == "no trained state"


def test_adapters_load_and_unload_through_callbacks(packed, tmp_path):
    d, _, _, manifest, _, _ = packed
    adir = d / "adapters" / "2"
    adir.mkdir(parents=True)
    (adir / "adapter_config.json").write_text("{}")
    (adir / "adapter_model.safetensors").write_bytes(b"\0" * 4096)
    (adir / "adapter.json").write_text(json.dumps({"rank": 4, "trainable_params": 1024}))
    loaded, unloaded = [], []
    cache = MpnResidency(d, 10**9, load_adapter=lambda m, p: loaded.append(m) or f"mpn{m}", unload_adapter=unloaded.append)
    r, ev = cache.acquire(2)
    assert ev["adapter"] is True and r.adapter == "mpn2" and r.adapter_bytes >= 4096 and loaded == [2]
    r1, ev1 = cache.acquire(1)
    assert ev1["adapter"] is False                      # MPN 1 has no specialist
    r3, ev3 = cache.acquire(3, want_adapter=False)
    assert ev3["adapter"] is False
    cache.release(2)
    assert cache.drop(2) == [2] and unloaded == ["mpn2"]
    assert cache.drop(1) == []                          # pinned: stays


def test_bundle_round_trip_restores_the_checkpoint_under_new_ids(tmp_path):
    """Export a store with a router+adapter checkpoint, import it into a fresh store, and check that every
    node's routing key survives under its new id and that the adapter is attached to the right MPN."""
    import shutil
    from pathlib import Path
    from mpnstudio.store import bundle
    from mpnstudio.store.db import Store
    fixture = Path(__file__).resolve().parents[1] / "data" / "dev-fixture.sqlite"
    if not fixture.exists():
        pytest.skip("run scripts/dev_fixture.py first")
    shutil.copy(fixture, tmp_path / "src.sqlite")
    src = Store(tmp_path / "src.sqlite")
    rows = src.q("SELECT node_id, content_hash FROM node_v WHERE rev_to IS NULL AND status='live' ORDER BY node_id")
    node_rows = {r[0]: i for i, r in enumerate(rows)}
    node_hash = {r[0]: r[1] for r in rows}
    primary, shared = {}, {}
    for r in src.q("SELECT node_id, mpn_id, role FROM member_v WHERE rev_to IS NULL"):
        (primary.__setitem__(r[0], r[1]) if r[2] == "primary" else shared.setdefault(r[0], []).append(r[1]))
    mpn_ids = sorted(int(m["id"]) for m in src.mpns())
    torch.manual_seed(3)
    spec = RouterSpec(n_rows=len(rows), n_mpn=len(mpn_ids), d_enc=16, d_key=8)
    r = Router(spec)
    with torch.no_grad():
        r.key_delta.normal_(0, 0.2)
    emb = torch.nn.functional.normalize(torch.randn(len(rows), 16), dim=-1)
    ck = tmp_path / "ck"
    from mpnstudio.model.bank import save_tensors
    import numpy as np
    package_router(ck, r, node_rows, node_hash, primary, shared, {m: i for i, m in enumerate(mpn_ids)}, emb)
    save_tensors(ck / "router.safetensors", r.state_dict(), {"spec": spec.to_json()})
    np.save(ck / "node_emb.f16.npy", emb.half().numpy())
    (ck / "nodes.json").write_text(json.dumps({str(n): {"row": node_rows[n], "hash": node_hash[n], "key_trained": True, "adapter_trained": primary.get(n) == mpn_ids[0]} for n in node_rows}))
    (ck / "manifest.json").write_text(json.dumps({"id": "ck-test", "rev": src.head(), "core": {"name": "test-core", "revision": "x"}}))
    ad = ck / "adapters" / str(mpn_ids[0])
    ad.mkdir(parents=True)
    (ad / "adapter_config.json").write_text("{}")
    (ad / "adapter_model.safetensors").write_bytes(b"\0" * 64)
    (ad / "adapter.json").write_text(json.dumps({"rank": 4, "nodes": {str(n): node_hash[n] for n in node_rows if primary.get(n) == mpn_ids[0]}}))
    src.conn().execute("INSERT INTO checkpoints(id, bank_id, kind, path, status, created) VALUES('ck-test','main','router+adapters',?, 'active', datetime('now'))", (str(ck),))
    src.set_setting("active_checkpoint", {"id": "ck-test"})

    m = bundle.export_bundle(src, tmp_path / "b", "yaob-test", bundle.active_checkpoint_dir(src))
    assert m["checkpoint"]["adapters"] == [src.mpns()[0]["slug"]] or m["checkpoint"]["adapters"]
    res = bundle.import_bundle(tmp_path / "b", tmp_path / "dst" / "dst.sqlite")
    assert res["matches_bundle"] and res["checkpoint"]["trained_and_unchanged"] == len(rows)
    dst = Store(tmp_path / "dst" / "dst.sqlite")
    out = Path(res["checkpoint"]["path"])
    cache = MpnResidency(out, 10**9)
    slug_src = {int(x["id"]): x["slug"] for x in src.mpns()}
    id_dst = {x["slug"]: int(x["id"]) for x in dst.mpns()}
    uid_src = {r[0]: r[1] for r in src.q("SELECT node_id, uid FROM node_v WHERE rev_to IS NULL")}
    id_by_uid_dst = {r[1]: r[0] for r in dst.q("SELECT node_id, uid FROM node_v WHERE rev_to IS NULL")}
    some = list(node_rows)[::97]
    for n in some:
        new_mpn = id_dst[slug_src[primary[n]]]
        shard, _ = cache.acquire(new_mpn)
        new_id = id_by_uid_dst[uid_src[n]]
        i = shard.tensors["node_ids"].tolist().index(new_id)
        want = r.node_keys(torch.tensor([node_rows[n]]), emb[node_rows[n]][None])[0].detach()
        assert torch.allclose(shard.tensors["keys"][i].float(), want, atol=3e-3)
        cache.release(new_mpn)
    assert (out / "adapters" / str(id_dst[slug_src[mpn_ids[0]]]) / "adapter_config.json").exists()
