import pytest

from mpnstudio.store.db import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "m.sqlite")
    with s.edit(kind="import", message="seed") as tx:
        hist = tx.put_mpn("history", "History")
        geo = tx.put_mpn("geography-geopolitics", "Geography and geopolitics")
        bio = tx.put_mpn("biology", "Biology")
        src = tx.put_source("wikipedia", "reference", "Battle of Waterloo", "https://en.wikipedia.org/wiki/Battle_of_Waterloo", "CC BY-SA 4.0", revision="1")
        w = tx.put_node(title="Battle of Waterloo", summary="1815 battle in which Napoleon was defeated.", origin="reference",
                        cells={"knowledge": {"statements": [{"text": "Napoleon was defeated at Waterloo in 1815."}]}})
        tx.set_membership(w, hist)
        tx.link_source(w, src, "lead")
        c = tx.put_node(title="Cell (biology)", summary="Basic unit of life.", origin="reference", cells={"knowledge": {"statements": []}})
        tx.set_membership(c, bio)
    s.ids = dict(hist=hist, geo=geo, bio=bio, w=w, c=c)
    return s


def test_counts_and_search(store):
    assert store.counts()["canonical_live_nodes"] == 2
    hits = store.search_lexical("napoleon")
    assert [h["title"] for h in hits] == ["Battle of Waterloo"]
    assert store.search_lexical("wikipedia waterloo")[0]["id"] == str(store.ids["w"])


def test_shared_identity_one_node_many_memberships(store):
    i = store.ids
    with store.edit(message="share") as tx:
        tx.set_membership(i["w"], i["geo"])
    assert store.counts()["canonical_live_nodes"] == 2          # not duplicated
    assert store.counts()["multi_membership_nodes"] == 1
    roles = {m["mpn_slug"]: m["role"] for m in store.get_node(i["w"])["memberships"]}
    assert roles == {"history": "primary", "geography-geopolitics": "shared"}
    with store.edit(message="unshare") as tx:
        tx.remove_membership(i["w"], i["hist"])
    n = store.get_node(i["w"])
    assert n["status"] == "live"
    assert {m["mpn_slug"]: m["role"] for m in n["memberships"]} == {"geography-geopolitics": "primary"}
    with pytest.raises(ValueError):
        with store.edit() as tx:
            tx.remove_membership(i["w"], i["geo"])


def test_revision_pinning_reads_old_state(store):
    i = store.ids
    pinned = store.head()
    with store.edit(message="edit") as tx:
        tx.update_node(i["w"], summary="Changed.")
    assert store.get_node(i["w"])["summary"] == "Changed."
    assert store.get_node(i["w"], rev=pinned)["summary"].startswith("1815")
    assert store.search_lexical("changed") and not store.search_lexical("which")


def test_delete_is_recoverable_and_undo_redo(store):
    i = store.ids
    with store.edit(message="delete") as tx:
        tx.delete_node(i["w"])
    assert store.counts()["canonical_live_nodes"] == 1 and store.counts()["tombstones"] == 1
    assert store.search_lexical("waterloo") == []
    assert store.undo() is not None
    assert store.get_node(i["w"])["status"] == "live"
    assert store.search_lexical("waterloo")
    assert store.redo() is not None
    assert store.get_node(i["w"])["status"] == "tombstone"
    assert store.undo() is not None
    assert store.get_node(i["w"])["status"] == "live"


def test_undo_restores_removed_membership_and_edge(store):
    i = store.ids
    with store.edit() as tx:
        tx.set_membership(i["w"], i["geo"])
        e = tx.put_edge(i["w"], i["c"], "related_to")
    with store.edit() as tx:
        tx.remove_membership(i["w"], i["geo"])
        tx.remove_edge(e)
    assert len(store.get_node(i["w"])["memberships"]) == 1 and store.edge_count(i["w"]) == 0
    store.undo()
    assert len(store.get_node(i["w"])["memberships"]) == 2 and store.edge_count(i["w"]) == 1


def test_revert_refuses_when_changed_later(store):
    i = store.ids
    with store.edit() as tx:
        tx.update_node(i["w"], summary="A")
    first = store.head()
    with store.edit() as tx:
        tx.update_node(i["w"], summary="B")
    with pytest.raises(ValueError):
        store.revert(first)


def test_empty_edit_leaves_no_revision(store):
    h = store.head()
    with store.edit() as tx:
        pass
    assert store.head() == h


def test_mpn_disable_is_versioned(store):
    i = store.ids
    before = store.head()
    with store.edit(kind="mpn") as tx:
        tx.set_mpn_state(i["bio"], enabled=False)
    assert store.mpn(i["bio"])["enabled"] is False
    assert store.mpn(i["bio"], rev=before)["enabled"] is True
    store.undo()
    assert store.mpn(i["bio"])["enabled"] is True


def test_ids_are_strings_at_the_boundary(store):
    n = store.get_node(store.ids["w"])
    assert isinstance(n["id"], str) and isinstance(n["memberships"][0]["mpn_id"], str)
