"""Model operations shared by the HTTP API and the CLI.

Both front ends call these functions, so the GUI and the command line always act on the same
revisions, jobs and rules. Each mutating operation returns the revision it created and what it
made stale (indexes, learned state), so callers can show the difference between a change that
is live at once and one that needs reprocessing or retraining.
"""
from __future__ import annotations

import json

from ..runtime import procedures
from ..store import layout, schemas
from ..store.db import Store, content_hash, iid, sid

BOARD = 1


class OpError(ValueError):
    def __init__(self, message: str, status: int = 400, detail: dict | None = None):
        super().__init__(message)
        self.status, self.detail = status, detail or {}


def _node_or_404(store: Store, node_id: int, rev: int | None = None) -> dict:
    n = store.get_node(node_id, rev)
    if n is None:
        raise OpError(f"node {node_id} does not exist", 404)
    return n


# ---------------------------------------------------------------------------- derived state
def learned_state(store: Store, node_id: int, node: dict | None = None) -> dict:
    """Where a node stands with respect to indexes and learned tensors."""
    node = node or store.get_node(node_id, full=False)
    d = store.q1("SELECT * FROM derived_state WHERE node_id=?", (node_id,))
    rows = store.q("SELECT b.bank_id, b.row, b.state FROM bank_rows b WHERE b.node_id=?", (node_id,))
    h = node["content_hash"] if node else None
    out = {
        "index": "missing" if d is None or d["embed_hash"] is None else ("current" if d["embed_hash"] == h else "stale"),
        "learned": "untrained" if d is None or d["learned_hash"] is None else ("trained" if d["learned_hash"] == h else "stale"),
        "checkpoint": d["learned_checkpoint"] if d else None,
        "rows": [{"bank": r["bank_id"], "row": r["row"], "state": r["state"]} for r in rows],
    }
    out["explain"] = {
        ("current", "trained"): "Indexed, and trained on the current content: its routing key, and its MPN's specialist where one exists.",
        ("current", "untrained"): "Indexed. Not trained yet: it is routed by a key derived from its text and read as text until a training job covers it.",
        ("current", "stale"): "Indexed. Content changed after training: it is routed by its text-derived key, and its MPN's specialist still holds the old content until it is retrained.",
        ("stale", "trained"): "Content changed: waiting to be re-indexed.",
        ("stale", "stale"): "Content changed: waiting to be re-indexed, and its trained state is out of date.",
        ("stale", "untrained"): "Content changed: waiting to be re-indexed.",
        ("missing", "untrained"): "New: findable by text search now, by meaning after indexing.",
    }.get((out["index"], out["learned"]), "")
    return out


def dirty_summary(store: Store) -> dict:
    live = store.q1("SELECT COUNT(*) FROM node_v WHERE rev_to IS NULL AND status='live'")[0]
    idx = store.q1("SELECT COUNT(*) FROM node_v n LEFT JOIN derived_state d ON d.node_id=n.node_id "
                   "WHERE n.rev_to IS NULL AND n.status='live' AND (d.embed_hash IS NULL OR d.embed_hash<>n.content_hash)")[0]
    stale = store.q1("SELECT COUNT(*) FROM node_v n JOIN derived_state d ON d.node_id=n.node_id "
                     "WHERE n.rev_to IS NULL AND n.status='live' AND d.learned_hash IS NOT NULL AND d.learned_hash<>n.content_hash")[0]
    untrained = store.q1("SELECT COUNT(*) FROM node_v n LEFT JOIN derived_state d ON d.node_id=n.node_id "
                         "WHERE n.rev_to IS NULL AND n.status='live' AND d.learned_hash IS NULL")[0]
    return {"live_nodes": live, "to_index": idx, "trained_stale": stale, "untrained": untrained, "trained_current": live - stale - untrained}


# ---------------------------------------------------------------------------- nodes
def node_detail(store: Store, node_id: int, rev: int | None = None) -> dict:
    n = _node_or_404(store, node_id, rev)
    n["edge_count"] = store.edge_count(node_id, rev)
    n["edges"] = store.edges_of(node_id, rev, limit=40)
    n["state"] = learned_state(store, node_id, n)
    spec = schemas.get_schema(store, n["schema"]["id"], n["schema"]["version"])
    n["schema"]["name"] = spec["name"] if spec else n["schema"]["id"]
    n["schema"]["roles"] = [{"key": r["key"], "label": r["label"], "runtime_role": r["runtime_role"], "description": r.get("description", ""),
                             "fields": r.get("fields", [])} for r in (spec["roles"] if spec else [])]
    n["problems"] = schemas.validate_cells(spec, n["cells"]) if spec else ["unknown schema"]
    pos = layout.node_position(store, BOARD, node_id)
    n["position"] = {"x": pos[0], "y": pos[1]} if pos else None
    # resolve referral targets (stored as portable uids) to ids for navigation
    for ref in n["cells"].get("reasoning", {}).get("referrals", []) or []:
        if isinstance(ref.get("node"), str) and ref["node"].startswith("yaob:"):
            target = store.node_by_uid(ref["node"])
            ref["id"] = sid(target)
    n["cost"] = node_cost(store, n)
    return n


def node_cost(store: Store, n: dict) -> dict:
    """Storage and context cost of one node. Text tokens are estimated at 4 characters per token."""
    text = " ".join(s.get("text", "") for s in n["cells"].get("knowledge", {}).get("statements", []) or [])
    spec = store.get_setting("bank_spec") or {}
    # per node: the routing-key residual (Thinking cell) and the gate bias (Reasoning cell). Content is carried by the
    # MPN's specialist adapter, which all of that MPN's nodes share; it is not counted here.
    per_node = (spec.get("d_key", 0) + 1) if spec else 0
    return {"record_bytes": len(json.dumps(n["cells"], ensure_ascii=False).encode()),
            "text_tokens_estimate": max(1, len(text) // 4),
            "learned_parameters": per_node, "learned_bytes_f32": per_node * 4}


def validate_node_payload(store: Store, payload: dict, existing: dict | None = None) -> tuple[dict, list[str]]:
    sch = payload.get("schema") or (existing or {}).get("schema") or {"id": "3cn", "version": 1}
    spec = schemas.get_schema(store, sch["id"], sch.get("version"))
    if spec is None:
        raise OpError(f"unknown schema {sch}")
    cells = payload.get("cells", (existing or {}).get("cells", {}))
    problems = schemas.validate_cells(spec, cells)
    impl = (cells.get("thinking") or {}).get("impl")
    if impl and impl not in procedures.REGISTRY:
        problems.append(f"Thinking: '{impl}' is not a built-in procedure. A procedure node can only name an implementation that ships with MPN Studio.")
    return spec, problems


def create_node(store: Store, payload: dict, actor: str = "local") -> dict:
    title = (payload.get("title") or "").strip()
    if not title:
        raise OpError("A node needs a title.")
    mems = payload.get("memberships") or []
    if not mems:
        raise OpError("A node needs at least one MPN membership.")
    cells = payload.get("cells") or {"knowledge": {"statements": []}, "thinking": {}, "reasoning": {}}
    payload = dict(payload, cells=cells)
    spec, problems = validate_node_payload(store, payload)
    if problems:
        raise OpError("The node does not match its schema.", 422, {"problems": problems})
    srcs = payload.get("sources") or []
    with store.edit(actor=actor, message=f"create node “{title}”") as tx:
        nid = tx.put_node(title=title, summary=payload.get("summary", ""), aliases=payload.get("aliases") or [], type=payload.get("type", "knowledge"),
                          schema_id=spec["id"], schema_version=spec["version"], origin=payload.get("origin", "user"),
                          validation="source_backed" if srcs else "unverified", cells=cells)
        for i, m in enumerate(mems):
            tx.set_membership(nid, iid(m["mpn_id"]), iid(m["domain_id"]) if m.get("domain_id") else None,
                              m.get("role") or ("primary" if i == 0 else "shared"), {"basis": "set by " + actor})
        for s in srcs:
            if not (s.get("title") or s.get("url")):
                continue
            src_id = tx.put_source(s.get("kind", "user"), "user", s.get("title") or s.get("url"), s.get("url", ""), s.get("license", ""),
                                   s.get("license_url", ""), s.get("revision", ""), s.get("retrieved_at", ""), s.get("attribution", ""))
            tx.link_source(nid, src_id, s.get("locator", ""))
        rev = tx.rev
    pos = payload.get("position")
    layout.place_node(store, BOARD, nid, (pos["x"], pos["y"]) if pos else None)
    return {"id": sid(nid), "rev": rev, "node": node_detail(store, nid),
            "effects": {"live_now": ["text search", "inspector", "board", "retrieval by text"],
                        "after_indexing": ["search by meaning", "learned routing"], "after_training": ["trained node state"]}}


def update_node(store: Store, node_id: int, payload: dict, actor: str = "local") -> dict:
    cur = _node_or_404(store, node_id)
    if cur["status"] == "tombstone":
        raise OpError("This node is deleted. Restore it before editing.", 409)
    changes = {k: payload[k] for k in ("title", "summary", "aliases", "type", "cells", "status", "validation") if k in payload}
    if "status" in changes and changes["status"] not in ("live", "disabled"):
        raise OpError("status can be set to live or disabled; use delete and restore for removal")
    if "schema" in payload:
        changes["schema_id"], changes["schema_version"] = payload["schema"]["id"], payload["schema"]["version"]
    if "cells" in changes or "schema" in payload:
        _, problems = validate_node_payload(store, payload, cur)
        if problems:
            raise OpError("The node does not match its schema.", 422, {"problems": problems})
    if not changes:
        return {"id": sid(node_id), "rev": store.head(), "node": node_detail(store, node_id), "changed": False}
    with store.edit(actor=actor, message=f"edit node “{changes.get('title', cur['title'])}”") as tx:
        tx.update_node(node_id, **changes)
        rev = tx.rev
    new = store.get_node(node_id, full=False)
    content_changed = new["content_hash"] != cur["content_hash"]
    return {"id": sid(node_id), "rev": rev, "node": node_detail(store, node_id), "changed": True,
            "effects": {"content_changed": content_changed,
                        "live_now": ["text search", "inspector", "retrieval by text"] + ([] if content_changed else ["everything: no derived state depends on this change"]),
                        "after_indexing": ["search by meaning", "learned routing"] if content_changed else [],
                        "after_training": ["trained node state"] if content_changed else []}}


def delete_preview(store: Store, node_id: int) -> dict:
    n = _node_or_404(store, node_id)
    state = learned_state(store, node_id, n)
    return {"node": {"id": n["id"], "title": n["title"]}, "memberships": n["memberships"], "edges": store.edge_count(node_id),
            "learned_rows": state["rows"], "checkpoint": state["checkpoint"],
            "note": "Deletion is recoverable. The node becomes a tombstone, leaves search and routing at once, and can be restored with undo or Restore. "
                    "Its tensor row is kept but no longer read; it is only reclaimed by an explicit compaction."}


def delete_node(store: Store, node_id: int, actor: str = "local") -> dict:
    n = _node_or_404(store, node_id)
    if n["status"] == "tombstone":
        return {"id": sid(node_id), "rev": store.head(), "changed": False}
    with store.edit(actor=actor, message=f"delete node “{n['title']}”") as tx:
        tx.delete_node(node_id)
        rev = tx.rev
    return {"id": sid(node_id), "rev": rev, "changed": True}


def restore_node(store: Store, node_id: int, actor: str = "local") -> dict:
    n = _node_or_404(store, node_id)
    if n["status"] != "tombstone":
        return {"id": sid(node_id), "rev": store.head(), "changed": False}
    with store.edit(actor=actor, message=f"restore node “{n['title']}”") as tx:
        tx.update_node(node_id, status="live")
        rev = tx.rev
    layout.place_node(store, BOARD, node_id)
    return {"id": sid(node_id), "rev": rev, "changed": True, "node": node_detail(store, node_id)}


def duplicate_node(store: Store, node_id: int, mode: str, actor: str = "local", position: dict | None = None) -> dict:
    """mode 'canonical': a new node with its own identity, recording where it was copied from.
    mode 'reference': another visual occurrence of the same node on the board; the model is unchanged."""
    n = _node_or_404(store, node_id)
    if mode == "reference":
        pos = position or {}
        base = layout.node_position(store, BOARD, node_id) or (0.0, 0.0)
        x, y = pos.get("x", base[0] + 60), pos.get("y", base[1] + 60)
        cur = store.conn().execute("INSERT INTO board_objects(board_id, kind, x, y, props, created) VALUES(?,?,?,?,?,datetime('now'))",
                                   (BOARD, "reference", x, y, json.dumps({"node_id": sid(node_id)})))
        return {"mode": "reference", "object_id": sid(cur.lastrowid), "node_id": sid(node_id), "rev": store.head(),
                "note": "A reference shows the same canonical node in a second place. Editing either edits the one node."}
    if mode != "canonical":
        raise OpError("mode must be 'canonical' or 'reference'")
    cells = json.loads(json.dumps(n["cells"]))
    with store.edit(actor=actor, message=f"duplicate node “{n['title']}”") as tx:
        new = tx.put_node(title=n["title"] + " (copy)", summary=n["summary"], aliases=n["aliases"], type=n["type"], schema_id=n["schema"]["id"],
                          schema_version=n["schema"]["version"], origin="user", validation=n["validation"], cells=cells)
        for m in n["memberships"]:
            tx.set_membership(new, iid(m["mpn_id"]), iid(m["domain_id"]) if m["domain_id"] else None, m["role"],
                              {"basis": "copied", "from": n["id"]})
        for s in n["sources"]:
            tx.link_source(new, iid(s["id"]), s["locator"])
        tx.put_edge(new, node_id, "derived_from", 1.0, "user", {"copied_at_rev": tx.rev, "from_uid": n["uid"]})
        rev = tx.rev
    base = layout.node_position(store, BOARD, node_id)
    layout.place_node(store, BOARD, new, (position["x"], position["y"]) if position else ((base[0] + 40, base[1] + 40) if base else None))
    return {"mode": "canonical", "id": sid(new), "rev": rev, "node": node_detail(store, new),
            "note": "A new canonical node. It starts untrained and is linked to its origin by a derived_from edge."}


def node_history(store: Store, node_id: int) -> list[dict]:
    rows = store.q("SELECT v.rev_from, v.rev_to, v.title, v.status, v.content_hash, r.created, r.actor, r.message, r.kind "
                   "FROM node_v v JOIN revisions r ON r.rev=v.rev_from WHERE v.node_id=? ORDER BY v.rev_from DESC LIMIT 100", (node_id,))
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------- memberships and edges
def set_membership(store: Store, node_id: int, mpn_id: int, role: str | None = None, domain_id: int | None = None, actor: str = "local") -> dict:
    n = _node_or_404(store, node_id)
    if store.mpn(mpn_id) is None:
        raise OpError(f"MPN {mpn_id} does not exist", 404)
    with store.edit(actor=actor, message=f"membership of “{n['title']}”") as tx:
        tx.set_membership(node_id, mpn_id, domain_id, role, {"basis": "set by " + actor})
        rev = tx.rev
    return {"rev": rev, "memberships": store.memberships(node_id)}


def remove_membership(store: Store, node_id: int, mpn_id: int, actor: str = "local") -> dict:
    n = _node_or_404(store, node_id)
    try:
        with store.edit(actor=actor, message=f"remove a membership of “{n['title']}”") as tx:
            tx.remove_membership(node_id, mpn_id)
            rev = tx.rev
    except KeyError:
        raise OpError("The node is not a member of that MPN.", 404)
    except ValueError as e:
        raise OpError(str(e), 409)
    return {"rev": rev, "memberships": store.memberships(node_id), "node_status": store.get_node(node_id, full=False)["status"]}


EDGE_TYPES = ("related_to", "part_of", "has_part", "instance_of", "subclass_of", "causes", "has_effect", "depends_on", "contradicts",
              "applies", "refers_to", "follows", "derived_from", "mentions")


def create_edge(store: Store, src: int, dst: int, type_: str, actor: str = "local", weight: float = 1.0) -> dict:
    a, b = _node_or_404(store, src), _node_or_404(store, dst)
    if not type_ or len(type_) > 40:
        raise OpError("An edge needs a relation type.")
    try:
        with store.edit(actor=actor, message=f"connect “{a['title']}” to “{b['title']}” ({type_})") as tx:
            eid = tx.put_edge(src, dst, type_, weight, "user", {"basis": "set by " + actor})
            rev = tx.rev
    except ValueError as e:
        raise OpError(str(e))
    return {"id": sid(eid), "rev": rev}


def remove_edge(store: Store, edge_id: int, actor: str = "local") -> dict:
    try:
        with store.edit(actor=actor, message="disconnect") as tx:
            tx.remove_edge(edge_id)
            rev = tx.rev
    except KeyError:
        raise OpError("No such edge.", 404)
    return {"rev": rev}


# ---------------------------------------------------------------------------- MPNs
def mpn_impact(store: Store, mpn_id: int, action: str) -> dict:
    """What an action on an MPN would do, before doing it."""
    m = store.mpn(mpn_id)
    if m is None:
        raise OpError("No such MPN.", 404)
    only = store.q1(
        "SELECT COUNT(*) FROM member_v a JOIN node_v n ON n.node_id=a.node_id AND n.rev_to IS NULL AND n.status<>'tombstone' "
        "WHERE a.rev_to IS NULL AND a.mpn_id=? AND NOT EXISTS(SELECT 1 FROM member_v b WHERE b.node_id=a.node_id AND b.rev_to IS NULL AND b.mpn_id<>a.mpn_id)",
        (mpn_id,))[0]
    shared_rows = store.q(
        "SELECT a.node_id, n.title, a.role FROM member_v a JOIN node_v n ON n.node_id=a.node_id AND n.rev_to IS NULL AND n.status<>'tombstone' "
        "WHERE a.rev_to IS NULL AND a.mpn_id=? AND EXISTS(SELECT 1 FROM member_v b WHERE b.node_id=a.node_id AND b.rev_to IS NULL AND b.mpn_id<>a.mpn_id) LIMIT 2000",
        (mpn_id,))
    cross = store.q1(
        "SELECT COUNT(*) FROM edge_v e JOIN member_v a ON a.node_id=e.src AND a.rev_to IS NULL AND a.role='primary' "
        "JOIN member_v b ON b.node_id=e.dst AND b.rev_to IS NULL AND b.role='primary' WHERE e.rev_to IS NULL AND (a.mpn_id=? OR b.mpn_id=?) AND a.mpn_id<>b.mpn_id",
        (mpn_id, mpn_id))[0]
    trained = store.q1("SELECT COUNT(*) FROM member_v a JOIN derived_state d ON d.node_id=a.node_id WHERE a.rev_to IS NULL AND a.mpn_id=? AND a.role='primary' AND d.learned_hash IS NOT NULL",
                       (mpn_id,))[0]
    text = {
        "disable": f"The runtime stops consulting {m['name']}. {only} nodes that belong only to it become unreachable; {len(shared_rows)} shared nodes stay reachable through their other MPNs. Nothing is deleted. The core's own pretrained knowledge of this subject is not affected.",
        "enable": f"The runtime may consult {m['name']} again.",
        "uninstall": f"{m['name']} is removed from the runtime and its package files are released. {only} nodes that belong only to it are tombstoned (recoverable). {len(shared_rows)} shared nodes keep their identity and move to another of their MPNs. Trained state for {trained} of its nodes stops contributing. The shared router was trained with this MPN present; that influence remains until it is retrained. Knowledge the core learned in pretraining is not removed.",
        "install": f"{m['name']} is made available to the runtime again and its tombstoned nodes are restored.",
    }.get(action, "")
    return {"mpn": {"id": m["id"], "name": m["name"], "installed": m["installed"], "enabled": m["enabled"]}, "action": action,
            "nodes_only_here": only, "shared_nodes": len(shared_rows), "shared_examples": [r["title"] for r in shared_rows[:8]],
            "cross_mpn_edges": cross, "trained_nodes": trained, "explanation": text,
            "entangled": action == "uninstall" and trained > 0}


def set_mpn(store: Store, mpn_id: int, action: str, actor: str = "local") -> dict:
    m = store.mpn(mpn_id)
    if m is None:
        raise OpError("No such MPN.", 404)
    impact = mpn_impact(store, mpn_id, action)
    if action in ("enable", "disable"):
        if not m["installed"]:
            raise OpError("This MPN is not installed.", 409)
        with store.edit(actor=actor, kind="mpn", message=f"{action} {m['name']}") as tx:
            tx.set_mpn_state(mpn_id, enabled=action == "enable")
            rev = tx.rev
    elif action == "uninstall":
        with store.edit(actor=actor, kind="mpn", message=f"uninstall {m['name']}") as tx:
            rows = tx.c.execute("SELECT node_id FROM member_v WHERE mpn_id=? AND rev_to IS NULL", (mpn_id,)).fetchall()
            for (nid,) in rows:
                others = tx.c.execute("SELECT COUNT(*) FROM member_v WHERE node_id=? AND rev_to IS NULL AND mpn_id<>?", (nid, mpn_id)).fetchone()[0]
                if others:
                    tx.remove_membership(nid, mpn_id)        # shared identity survives; another membership becomes primary
                else:
                    st = tx.c.execute("SELECT status FROM node_v WHERE node_id=? AND rev_to IS NULL", (nid,)).fetchone()[0]
                    if st != "tombstone":
                        tx.delete_node(nid)                  # recoverable
            tx.set_mpn_state(mpn_id, installed=False, enabled=False)
            rev = tx.rev
    elif action == "install":
        if m["installed"]:
            return {"rev": store.head(), "changed": False, "mpn": m}
        # reinstall = revert the uninstall revision when nothing touched those entities since; otherwise only the flag
        un = store.q1("SELECT rev FROM revisions WHERE kind='mpn' AND message=? ORDER BY rev DESC LIMIT 1", (f"uninstall {m['name']}",))
        rev = None
        if un is not None:
            try:
                rev = store.revert(un[0], actor, "redo")
            except ValueError:
                rev = None
        if rev is None:
            with store.edit(actor=actor, kind="mpn", message=f"install {m['name']}") as tx:
                tx.set_mpn_state(mpn_id, installed=True, enabled=True)
                rev = tx.rev
    else:
        raise OpError("action must be enable, disable, install or uninstall")
    return {"rev": rev, "changed": True, "mpn": store.mpn(mpn_id), "impact": impact}


def create_mpn(store: Store, payload: dict, actor: str = "local") -> dict:
    import re
    name = (payload.get("name") or "").strip()
    if not name:
        raise OpError("An MPN needs a name.")
    slug = re.sub(r"[^a-z0-9]+", "-", (payload.get("slug") or name).lower()).strip("-")[:48]
    if not slug:
        raise OpError("An MPN needs a slug of letters or digits.")
    if store.mpn_by_slug(slug):
        raise OpError(f"An MPN with the slug '{slug}' already exists.", 409)
    with store.edit(actor=actor, kind="mpn", message=f"create MPN {name}") as tx:
        mid = tx.put_mpn(slug, name, "user", payload.get("description", ""), manifest={"package": f"user.mpn.{slug}", "class": "knowledge-only"})
        rev = tx.rev
    place_mpn_region(store, mid)
    return {"rev": rev, "mpn": store.mpn(mid)}


def place_mpn_region(store: Store, mpn_id: int) -> None:
    """Give a new, empty MPN a region beside the existing ones so it can be seen and dropped into."""
    if store.q1("SELECT 1 FROM regions WHERE board_id=? AND kind='mpn' AND ref_id=?", (BOARD, mpn_id)):
        return
    b = layout.bounds(store, BOARD)
    r = 220.0
    store.conn().execute("INSERT INTO regions VALUES(?,?,?,?,?,?)", (BOARD, "mpn", mpn_id, b[2] + r + 80, (b[1] + b[3]) / 2, r))


# ---------------------------------------------------------------------------- schemas
def schema_migration_preview(store: Store, schema_id: str, from_version: int, to_id: str, to_version: int) -> dict:
    old = schemas.get_schema(store, schema_id, from_version)
    new = schemas.get_schema(store, to_id, to_version)
    if not old or not new:
        raise OpError("Unknown schema version.", 404)
    d = schemas.diff_schemas(old, new)
    n = store.q1("SELECT COUNT(*) FROM node_v WHERE schema_id=? AND schema_version=? AND rev_to IS NULL AND status<>'tombstone'", (schema_id, from_version))[0]
    sample = store.q("SELECT node_id, cells FROM node_v WHERE schema_id=? AND schema_version=? AND rev_to IS NULL AND status<>'tombstone' LIMIT 200", (schema_id, from_version))
    failing = 0
    example = None
    for r in sample:
        moved = schemas.migrate_cells(old, new, json.loads(r["cells"]))
        probs = schemas.validate_cells(new, moved)
        if probs:
            failing += 1
            example = example or {"node_id": sid(r["node_id"]), "problems": probs[:3]}
    return {"from": {"id": schema_id, "version": from_version}, "to": {"id": to_id, "version": to_version}, "diff": d, "nodes": n,
            "sampled": len(sample), "sample_failing": failing, "example_failure": example,
            "can_apply": failing == 0,
            "note": "Cells and fields present in both versions are carried over unchanged; new cells start empty. "
                    "Learned state is kept because the knowledge, thinking and reasoning cells are unchanged." if not d["retrain_required"]
                    else "This migration changes a cell the learned state depends on: affected nodes are marked for retraining."}


def schema_migrate(store: Store, schema_id: str, from_version: int, to_id: str, to_version: int, node_ids: list[int] | None = None,
                   actor: str = "local") -> dict:
    prev = schema_migration_preview(store, schema_id, from_version, to_id, to_version)
    if not prev["can_apply"]:
        raise OpError("Some nodes would not validate after migration.", 422, prev)
    old = schemas.get_schema(store, schema_id, from_version)
    new = schemas.get_schema(store, to_id, to_version)
    where = "schema_id=? AND schema_version=? AND rev_to IS NULL AND status<>'tombstone'"
    args: list = [schema_id, from_version]
    if node_ids:
        where += f" AND node_id IN ({','.join('?' * len(node_ids))})"
        args += node_ids
    rows = store.q(f"SELECT node_id, cells FROM node_v WHERE {where}", args)
    skipped = 0
    with store.edit(actor=actor, kind="schema", message=f"migrate {len(rows)} nodes {schema_id} v{from_version} -> {to_id} v{to_version}") as tx:
        for r in rows:
            moved = schemas.migrate_cells(old, new, json.loads(r["cells"]))
            if schemas.validate_cells(new, moved):
                skipped += 1
                continue
            tx.update_node(r["node_id"], cells=moved, schema_id=to_id, schema_version=to_version)
        rev = tx.rev
    return {"rev": rev, "migrated": len(rows) - skipped, "skipped": skipped, "preview": prev}


# ---------------------------------------------------------------------------- search
def search(store: Store, q: str, mode: str = "hybrid", limit: int = 20, offset: int = 0, filters: dict | None = None,
           semantic_fn=None) -> dict:
    """Lexical (FTS5/BM25), semantic (routing index, when the runtime is up) or both, merged by reciprocal rank."""
    import time
    t0 = time.perf_counter()
    q = (q or "").strip()
    filters = {k: v for k, v in (filters or {}).items() if v}
    out: dict = {"query": q, "mode": mode, "results": [], "semantic_available": semantic_fn is not None}
    if not q and not filters:
        return out
    if q.isdigit() or q.startswith("yaob:"):              # direct id / uid lookup
        nid = int(q) if q.isdigit() else store.node_by_uid(q)
        n = store.get_node(nid, full=False) if nid else None
        if n and n["status"] != "tombstone":
            out["results"] = [{"id": n["id"], "title": n["title"], "type": n["type"], "status": n["status"], "validation": n["validation"],
                               "summary": n["summary"][:240], "score": 1.0, "match": "id"}]
    lex = []
    if q and mode in ("lexical", "hybrid") and not out["results"]:
        lex = store.search_lexical(q, limit=limit + offset + 20, filters=filters)
        if len(lex) < 5:
            seen = {r["id"] for r in lex}
            lex += [r for r in store.search_lexical(q, limit=limit + offset + 20, filters=filters, any_terms=True) if r["id"] not in seen]
    sem = []
    if q and mode in ("semantic", "hybrid") and semantic_fn is not None and not out["results"]:
        try:
            sem = semantic_fn(q, limit + offset + 20, filters)
        except Exception as e:
            out["semantic_error"] = str(e)[:200]
            out["semantic_available"] = False
    if not q and filters and not out["results"]:
        lex = filter_only(store, filters, limit + offset)
    if not out["results"]:
        score: dict[str, float] = {}
        info: dict[str, dict] = {}
        for rank, r in enumerate(lex):
            score[r["id"]] = score.get(r["id"], 0) + 1 / (30 + rank)
            info.setdefault(r["id"], dict(r, match="text"))
        for rank, r in enumerate(sem):
            score[r["id"]] = score.get(r["id"], 0) + 1 / (30 + rank)
            if r["id"] in info:
                info[r["id"]]["match"] = "text+meaning"
            else:
                info[r["id"]] = dict(r, match="meaning")
        # a node whose title is the query comes first, then titles that start with it (rank fusion alone put
        # "Dutch East India Company" above "East India Company")
        ql = q.lower()
        for k, r in info.items():
            t = (r.get("title") or "").lower()
            if q and t == ql:
                score[k] += 1.0
            elif q and t.startswith(ql):
                score[k] += 0.01
        ranked = sorted(score, key=lambda k: -score[k])[offset:offset + limit]
        out["results"] = [dict(info[k], score=round(score[k], 5)) for k in ranked]
    ids = [iid(r["id"]) for r in out["results"]]
    if ids:
        ph = ",".join("?" * len(ids))
        mp = {r[0]: (r[1], r[2]) for r in store.q(
            f"SELECT m.node_id, p.name, p.mpn_id FROM member_v m JOIN mpn_v p ON p.mpn_id=m.mpn_id AND p.rev_to IS NULL "
            f"WHERE m.rev_to IS NULL AND m.role='primary' AND m.node_id IN ({ph})", ids)}
        ps = {r[0]: (r[1], r[2]) for r in store.q(f"SELECT node_id, x, y FROM layout WHERE board_id=? AND node_id IN ({ph})", (BOARD, *ids))}
        for r in out["results"]:
            k = iid(r["id"])
            if k in mp:
                r["mpn"], r["mpn_id"] = mp[k][0], sid(mp[k][1])
            if k in ps:
                r["x"], r["y"] = ps[k]
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


def filter_only(store: Store, f: dict, limit: int) -> list[dict]:
    where, args = ["n.rev_to IS NULL"], []
    for k in ("status", "type", "validation", "origin"):
        if f.get(k):
            where.append(f"n.{k}=?")
            args.append(f[k])
    if not f.get("status"):
        where.append("n.status<>'tombstone'")
    if f.get("mpn_id"):
        where.append("EXISTS(SELECT 1 FROM member_v m WHERE m.node_id=n.node_id AND m.rev_to IS NULL AND m.mpn_id=?)")
        args.append(iid(f["mpn_id"]))
    rows = store.q(f"SELECT n.node_id, n.title, n.type, n.status, n.validation, n.summary FROM node_v n WHERE {' AND '.join(where)} ORDER BY n.title LIMIT ?", (*args, limit))
    return [{"id": sid(r["node_id"]), "title": r["title"], "type": r["type"], "status": r["status"], "validation": r["validation"],
             "summary": r["summary"][:240], "score": 0.0} for r in rows]


__all__ = [n for n in dir() if not n.startswith("_")]
_ = content_hash
