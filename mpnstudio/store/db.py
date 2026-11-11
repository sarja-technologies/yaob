"""The Yaob model store: versioned nodes, memberships, edges and MPNs in SQLite.

Reads take an optional revision. `rev=None` means head. A request that must see
one consistent network pins `store.head()` once and passes it to every read.

Writes go through `store.edit(...)`, which opens one transaction, allocates one
new revision and closes or opens version rows. Undo is generic: it reopens what
a revision closed and closes what it opened, as a new revision.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from .schema import DDL, SCHEMA_VERSION, VERSIONED

ENTITY_KEYS = {
    "node_v": ("node_id",), "member_v": ("node_id", "mpn_id"), "edge_v": ("edge_id",),
    "mpn_v": ("mpn_id",), "domain_v": ("domain_id",), "node_source_v": ("node_id", "source_id"),
}
STOPWORDS = frozenset("the and for was are were with from that this what which who whom how why when where did does has have had its not into than then "
                      "you your our their his her they them can could would should about over under between during after before".split())
NODE_TYPES = ("knowledge", "procedure", "rule", "tool", "learned_memory")
NODE_STATUS = ("live", "disabled", "tombstone")
VALIDATION = ("unverified", "source_backed", "checked", "failed")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sid(i: int | None) -> str | None:
    """Ids cross every API boundary as decimal strings."""
    return None if i is None else str(i)


def iid(s: Any) -> int:
    if isinstance(s, bool):
        raise ValueError("bad id")
    if isinstance(s, int):
        return s
    if isinstance(s, str) and s.isdigit():
        return int(s)
    raise ValueError(f"bad id: {s!r}")


def content_hash(title: str, summary: str, aliases: list, cells: dict, type_: str) -> str:
    """Hash of everything learned state and indexes are derived from."""
    blob = json.dumps([title, summary, sorted(aliases), cells, type_], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def cell_text(cells: dict) -> str:
    """Flatten cell content to searchable text."""
    out: list[str] = []

    def walk(v):
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            for k, x in v.items():
                if k in ("source", "source_id", "impl", "learned"):
                    continue
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(str(v))
    walk(cells)
    return "\n".join(out)


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._wlock = threading.RLock()
        with self.conn() as c:
            c.executescript(DDL)
            c.execute("INSERT OR IGNORE INTO meta VALUES('schema_version', ?)", (str(SCHEMA_VERSION),))
            c.execute("INSERT OR IGNORE INTO meta VALUES('created', ?)", (now(),))
            if c.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 0:
                c.execute("INSERT INTO revisions(rev, parent, created, actor, kind, message) VALUES(0, NULL, ?, 'system', 'system', 'empty store')", (now(),))
            c.commit()
        # stores created before the summary table existed: build it once
        if self.q1("SELECT 1 FROM summary_counts LIMIT 1") is None and self.q1("SELECT 1 FROM member_v LIMIT 1") is not None:
            self.rebuild_summaries()

    # ------------------------------------------------------------------ connections
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "c", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA foreign_keys=OFF")
            c.execute("PRAGMA busy_timeout=30000")
            c.execute("PRAGMA cache_size=-65536")     # 64 MiB page cache per connection, bounded
            c.execute("PRAGMA mmap_size=0")           # no memory map: keeps resident memory accountable
            self._local.c = c
        return c

    def q(self, sql: str, args: Iterable = ()) -> list[sqlite3.Row]:
        return self.conn().execute(sql, tuple(args)).fetchall()

    def q1(self, sql: str, args: Iterable = ()) -> sqlite3.Row | None:
        return self.conn().execute(sql, tuple(args)).fetchone()

    def head(self) -> int:
        return self.q1("SELECT MAX(rev) FROM revisions")[0]

    @staticmethod
    def at(rev: int | None, alias: str = "") -> tuple[str, tuple]:
        """SQL predicate selecting the version valid at `rev` (None = head)."""
        p = f"{alias}." if alias else ""
        if rev is None:
            return f"{p}rev_to IS NULL", ()
        return f"({p}rev_from <= ? AND ({p}rev_to IS NULL OR {p}rev_to > ?))", (rev, rev)

    # ------------------------------------------------------------------ settings/meta
    def get_setting(self, key: str, default=None):
        r = self.q1("SELECT value FROM settings WHERE key=?", (key,))
        return json.loads(r[0]) if r else default

    def set_setting(self, key: str, value) -> None:
        with self._wlock:
            self.conn().execute("INSERT INTO settings VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                                (key, json.dumps(value)))

    # ------------------------------------------------------------------ edit transactions
    @contextmanager
    def edit(self, actor: str = "local", kind: str = "edit", message: str = "", undo_of: int | None = None):
        with self._wlock:
            c = self.conn()
            c.execute("BEGIN IMMEDIATE")
            try:
                parent = c.execute("SELECT MAX(rev) FROM revisions").fetchone()[0]
                rev = parent + 1
                c.execute("INSERT INTO revisions(rev, parent, created, actor, kind, message, undo_of) VALUES(?,?,?,?,?,?,?)",
                          (rev, parent, now(), actor, kind, message, undo_of))
                tx = Tx(self, c, rev)
                yield tx
                if not tx.touched and kind not in ("import", "system"):
                    c.execute("ROLLBACK")      # an edit that changed nothing leaves no revision behind
                    tx.rev = parent
                    return
                self._apply_summary_delta(c, tx.nodes_changed, parent)
                c.execute("UPDATE revisions SET stats=? WHERE rev=?", (json.dumps(tx.stats), rev))
                c.execute("COMMIT")
            except BaseException:
                c.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------------ incremental summaries
    @staticmethod
    def _contribution(c: sqlite3.Connection, node_id: int, rev: int | None) -> list[tuple[str, int, int, int]]:
        """What one node adds to the live counts at a revision: [(kind, ref_id, live, shared)]."""
        w, a = Store.at(rev)
        r = c.execute(f"SELECT status FROM node_v WHERE node_id=? AND {w}", (node_id, *a)).fetchone()
        if r is None or r[0] != "live":
            return []
        out = []
        for m in c.execute(f"SELECT mpn_id, domain_id, role FROM member_v WHERE node_id=? AND {w}", (node_id, *a)):
            out.append(("mpn", m[0], 1, 1 if m[2] == "shared" else 0))
            if m[1] is not None:
                out.append(("domain", m[1], 1, 0))
        return out

    def _apply_summary_delta(self, c: sqlite3.Connection, nodes: set[int], before_rev: int) -> None:
        """Bring per-MPN and per-domain live counts up to date for the nodes an edit touched. Cost is proportional
        to the number of touched nodes, not to the size of the network."""
        if not nodes:
            return
        delta: dict[tuple[str, int], list[int]] = {}
        for nid in nodes:
            for kind, ref, live, shared in self._contribution(c, nid, before_rev):
                d = delta.setdefault((kind, ref), [0, 0])
                d[0] -= live
                d[1] -= shared
            for kind, ref, live, shared in self._contribution(c, nid, None):
                d = delta.setdefault((kind, ref), [0, 0])
                d[0] += live
                d[1] += shared
        for (kind, ref), (dl, ds) in delta.items():
            if dl or ds:
                c.execute("INSERT INTO summary_counts(kind, ref_id, live, shared) VALUES(?,?,?,?) "
                          "ON CONFLICT(kind, ref_id) DO UPDATE SET live=live+excluded.live, shared=shared+excluded.shared", (kind, ref, dl, ds))

    def rebuild_summaries(self) -> None:
        """Recompute every count from scratch (after a bulk load done outside `edit`, or to verify the incremental path)."""
        c = self.conn()
        with self._wlock:
            c.execute("BEGIN IMMEDIATE")
            try:
                c.execute("DELETE FROM summary_counts")
                c.execute("INSERT INTO summary_counts(kind, ref_id, live, shared) SELECT 'mpn', m.mpn_id, COUNT(*), SUM(m.role='shared') FROM member_v m "
                          "JOIN node_v n ON n.node_id=m.node_id AND n.rev_to IS NULL WHERE m.rev_to IS NULL AND n.status='live' GROUP BY m.mpn_id")
                c.execute("INSERT INTO summary_counts(kind, ref_id, live, shared) SELECT 'domain', m.domain_id, COUNT(*), 0 FROM member_v m "
                          "JOIN node_v n ON n.node_id=m.node_id AND n.rev_to IS NULL WHERE m.rev_to IS NULL AND n.status='live' AND m.domain_id IS NOT NULL GROUP BY m.domain_id")
                c.execute("COMMIT")
            except BaseException:
                c.execute("ROLLBACK")
                raise

    def summary_counts(self, kind: str) -> dict[int, tuple[int, int]]:
        return {r[0]: (r[1], r[2]) for r in self.q("SELECT ref_id, live, shared FROM summary_counts WHERE kind=?", (kind,))}

    def _next_id(self, c: sqlite3.Connection, kind: str, n: int = 1) -> int:
        r = c.execute("SELECT next FROM ids WHERE kind=?", (kind,)).fetchone()
        first = r[0] if r else 1
        c.execute("INSERT INTO ids VALUES(?,?) ON CONFLICT(kind) DO UPDATE SET next=excluded.next", (kind, first + n))
        return first

    # ------------------------------------------------------------------ revisions, undo, redo
    def revisions(self, limit: int = 50, before: int | None = None) -> list[dict]:
        if before is None:
            rows = self.q("SELECT * FROM revisions ORDER BY rev DESC LIMIT ?", (limit,))
        else:
            rows = self.q("SELECT * FROM revisions WHERE rev < ? ORDER BY rev DESC LIMIT ?", (before, limit))
        return [dict(r, stats=json.loads(r["stats"])) for r in rows]

    def revision_changes(self, rev: int) -> dict:
        """What a revision opened and closed, by table."""
        out = {}
        for table in VERSIONED:
            opened = self.q1(f"SELECT COUNT(*) FROM {table} WHERE rev_from=?", (rev,))[0]
            closed = self.q1(f"SELECT COUNT(*) FROM {table} WHERE rev_to=?", (rev,))[0]
            if opened or closed:
                out[table] = {"opened": opened, "closed": closed}
        return out

    def revert(self, rev: int, actor: str = "local", kind: str = "undo") -> int:
        """Create a new revision that undoes exactly what `rev` did.

        Refuses when a later revision already changed one of the same entities,
        because blindly reopening an old version would silently discard that work.
        """
        with self._wlock:
            for table, keys in ENTITY_KEYS.items():
                kl = ", ".join(keys)
                stale = self.q1(f"SELECT {kl} FROM {table} WHERE rev_from=? AND rev_to IS NOT NULL LIMIT 1", (rev,))
                if stale is not None:
                    raise ValueError(f"revision {rev} cannot be reverted: {table} {tuple(stale)} was changed again later")
                on = " AND ".join(f"b.{k} IS a.{k}" for k in keys)
                clash = self.q1(f"SELECT a.{keys[0]} FROM {table} a JOIN {table} b ON {on} "
                                f"WHERE a.rev_to=? AND b.rev_to IS NULL AND b.rev_from<>? LIMIT 1", (rev, rev))
                if clash is not None:
                    raise ValueError(f"revision {rev} cannot be reverted: {table} {clash[0]} was recreated later")
            with self.edit(actor=actor, kind=kind, message=f"{kind} of revision {rev}", undo_of=rev) as tx:
                c = tx.c
                touched_nodes: set[int] = set()
                for table, (idcol, cols) in VERSIONED.items():
                    collist = ", ".join(cols)
                    opened = c.execute(f"SELECT vid, {idcol} FROM {table} WHERE rev_from=? AND rev_to IS NULL", (rev,)).fetchall()
                    closed = c.execute(f"SELECT {collist} FROM {table} WHERE rev_to=?", (rev,)).fetchall()
                    for r in opened:
                        c.execute(f"UPDATE {table} SET rev_to=? WHERE vid=?", (tx.rev, r[0]))
                    for r in closed:
                        c.execute(f"INSERT INTO {table}({collist}, rev_from, rev_to) VALUES({','.join('?' * len(cols))}, ?, NULL)",
                                  (*tuple(r), tx.rev))
                    if table in ("node_v", "member_v", "node_source_v"):
                        touched_nodes.update(r[1] for r in opened)
                        touched_nodes.update(r[0] for r in closed)
                    if opened or closed:
                        tx.touched = True
                        tx.stats[table] = {"closed": len(opened), "reopened": len(closed)}
                for nid in touched_nodes:
                    tx._reindex_fts(nid)
                tx.nodes_changed.update(touched_nodes)
            return tx.rev

    def undo_target(self) -> int | None:
        """The most recent user revision that has not been undone."""
        undone: set[int] = set()
        for r in self.q("SELECT rev, kind, undo_of FROM revisions ORDER BY rev DESC LIMIT 500"):
            if r["rev"] in undone:
                continue
            if r["kind"] == "undo":
                undone.add(r["undo_of"])
                continue
            if r["kind"] == "redo":
                # a redo re-applies; it is itself undoable
                return r["rev"]
            if r["kind"] in ("edit", "mpn", "schema"):
                return r["rev"]
        return None

    def redo_target(self) -> int | None:
        """The most recent undo that has not itself been reverted, if nothing was edited since."""
        reverted: set[int] = set()
        for r in self.q("SELECT rev, kind, undo_of FROM revisions ORDER BY rev DESC LIMIT 500"):
            if r["rev"] in reverted:
                continue
            if r["kind"] == "redo":
                reverted.add(r["undo_of"])
                continue
            if r["kind"] == "undo":
                return r["rev"]
            if r["kind"] in ("edit", "mpn", "schema"):
                return None
        return None

    def undo(self, actor: str = "local") -> int | None:
        t = self.undo_target()
        return None if t is None else self.revert(t, actor, "undo")

    def redo(self, actor: str = "local") -> int | None:
        t = self.redo_target()
        return None if t is None else self.revert(t, actor, "redo")

    # ------------------------------------------------------------------ reads
    def _node_row(self, r: sqlite3.Row, full: bool = True) -> dict:
        d = {"id": sid(r["node_id"]), "uid": r["uid"], "type": r["type"], "title": r["title"], "summary": r["summary"],
             "status": r["status"], "validation": r["validation"], "origin": r["origin"],
             "schema": {"id": r["schema_id"], "version": r["schema_version"]},
             "content_hash": r["content_hash"], "rev": r["rev_from"]}
        if full:
            d["aliases"] = json.loads(r["aliases"])
            d["cells"] = json.loads(r["cells"])
        return d

    def get_node(self, node_id: int, rev: int | None = None, full: bool = True) -> dict | None:
        w, a = self.at(rev)
        r = self.q1(f"SELECT * FROM node_v WHERE node_id=? AND {w}", (node_id, *a))
        if r is None:
            return None
        d = self._node_row(r, full)
        if full:
            d["memberships"] = self.memberships(node_id, rev)
            d["sources"] = self.node_sources(node_id, rev)
        return d

    def get_nodes(self, ids: list[int], rev: int | None = None, full: bool = False) -> list[dict]:
        if not ids:
            return []
        w, a = self.at(rev)
        out = []
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            rows = self.q(f"SELECT * FROM node_v WHERE node_id IN ({','.join('?' * len(chunk))}) AND {w}", (*chunk, *a))
            out += [self._node_row(r, full) for r in rows]
        order = {n: k for k, n in enumerate(ids)}
        out.sort(key=lambda d: order[int(d["id"])])
        return out

    def node_by_uid(self, uid: str) -> int | None:
        r = self.q1("SELECT node_id FROM node_v WHERE uid=? AND rev_to IS NULL", (uid,))
        return r[0] if r else None

    def memberships(self, node_id: int, rev: int | None = None) -> list[dict]:
        w, a = self.at(rev, "m")
        wm, am = self.at(rev, "p")
        rows = self.q(
            f"SELECT m.mpn_id, m.domain_id, m.role, m.evidence, p.slug, p.name, p.enabled, p.installed "
            f"FROM member_v m JOIN mpn_v p ON p.mpn_id=m.mpn_id AND {wm} WHERE m.node_id=? AND {w} ORDER BY m.role, p.name",
            (*am, node_id, *a))
        return [{"mpn_id": sid(r["mpn_id"]), "domain_id": sid(r["domain_id"]), "role": r["role"], "mpn_slug": r["slug"],
                 "mpn_name": r["name"], "mpn_enabled": bool(r["enabled"]), "mpn_installed": bool(r["installed"]),
                 "evidence": json.loads(r["evidence"])} for r in rows]

    def node_sources(self, node_id: int, rev: int | None = None) -> list[dict]:
        w, a = self.at(rev, "ns")
        rows = self.q(f"SELECT s.*, ns.locator FROM node_source_v ns JOIN sources s ON s.id=ns.source_id WHERE ns.node_id=? AND {w} ORDER BY s.id",
                      (node_id, *a))
        return [{"id": sid(r["id"]), "kind": r["kind"], "origin": r["origin"], "title": r["title"], "url": r["url"],
                 "license": r["license"], "license_url": r["license_url"], "revision": r["revision"],
                 "retrieved_at": r["retrieved_at"], "attribution": r["attribution"], "locator": r["locator"]} for r in rows]

    def edges_of(self, node_id: int, rev: int | None = None, limit: int = 200, offset: int = 0) -> list[dict]:
        w, a = self.at(rev, "e")
        wn, an = self.at(rev, "n")
        rows = self.q(
            f"SELECT e.edge_id, e.src, e.dst, e.type, e.weight, e.origin, e.evidence, n.title, n.type AS ntype, n.status "
            f"FROM edge_v e JOIN node_v n ON n.node_id = CASE WHEN e.src=? THEN e.dst ELSE e.src END AND {wn} "
            f"WHERE (e.src=? OR e.dst=?) AND {w} ORDER BY e.weight DESC, e.edge_id LIMIT ? OFFSET ?",
            (node_id, *an, node_id, node_id, *a, limit, offset))
        return [{"id": sid(r["edge_id"]), "src": sid(r["src"]), "dst": sid(r["dst"]), "type": r["type"], "weight": r["weight"],
                 "origin": r["origin"], "evidence": json.loads(r["evidence"]),
                 "other": {"id": sid(r["dst"] if r["src"] == node_id else r["src"]), "title": r["title"], "type": r["ntype"], "status": r["status"]},
                 "direction": "out" if r["src"] == node_id else "in"} for r in rows]

    def edge_count(self, node_id: int, rev: int | None = None) -> int:
        w, a = self.at(rev)
        return self.q1(f"SELECT COUNT(*) FROM edge_v WHERE (src=? OR dst=?) AND {w}", (node_id, node_id, *a))[0]

    def mpns(self, rev: int | None = None) -> list[dict]:
        w, a = self.at(rev)
        wm, am = self.at(rev, "m")
        wn, an = self.at(rev, "n")
        if rev is None or rev == self.head():
            counts = self.summary_counts("mpn")           # maintained incrementally; no scan
        else:
            counts = {r[0]: (r[1], r[2]) for r in self.q(
                f"SELECT m.mpn_id, COUNT(*), SUM(m.role='shared') FROM member_v m JOIN node_v n ON n.node_id=m.node_id AND {wn} "
                f"WHERE {wm} AND n.status='live' GROUP BY m.mpn_id", (*an, *am))}
        out = []
        for r in self.q(f"SELECT * FROM mpn_v WHERE {w} ORDER BY name", a):
            n, shared = counts.get(r["mpn_id"], (0, 0))
            out.append({"id": sid(r["mpn_id"]), "slug": r["slug"], "name": r["name"], "kind": r["kind"],
                        "description": r["description"], "version": r["version"], "manifest": json.loads(r["manifest"]),
                        "installed": bool(r["installed"]), "enabled": bool(r["enabled"]),
                        "node_count": n, "shared_count": shared or 0, "rev": r["rev_from"]})
        return out

    def mpn(self, mpn_id: int, rev: int | None = None) -> dict | None:
        for m in self.mpns(rev):
            if m["id"] == str(mpn_id):
                return m
        return None

    def mpn_by_slug(self, slug: str, rev: int | None = None) -> dict | None:
        for m in self.mpns(rev):
            if m["slug"] == slug:
                return m
        return None

    def domains(self, mpn_id: int | None = None, rev: int | None = None) -> list[dict]:
        w, a = self.at(rev, "d")
        wm, am = self.at(rev, "m")
        wn, an = self.at(rev, "n")
        if rev is None or rev == self.head():
            counts = {k: v[0] for k, v in self.summary_counts("domain").items()}
        else:
            counts = dict(self.q(
                f"SELECT m.domain_id, COUNT(*) FROM member_v m JOIN node_v n ON n.node_id=m.node_id AND {wn} "
                f"WHERE {wm} AND n.status='live' AND m.domain_id IS NOT NULL GROUP BY m.domain_id", (*an, *am)))
        sql = f"SELECT d.* FROM domain_v d WHERE {w}"
        args: tuple = a
        if mpn_id is not None:
            sql += " AND d.mpn_id=?"
            args = (*a, mpn_id)
        rows = self.q(sql + " ORDER BY d.mpn_id, d.depth, d.name", args)
        # a domain's count includes its sub-domains (nodes are filed under the deepest one)
        total = dict(counts)
        parent = {r["domain_id"]: r["parent_id"] for r in rows}
        for did, c in counts.items():
            p, seen = parent.get(did), 0
            while p is not None and seen < 8:
                total[p] = total.get(p, 0) + c
                p, seen = parent.get(p), seen + 1
        return [{"id": sid(r["domain_id"]), "mpn_id": sid(r["mpn_id"]), "parent_id": sid(r["parent_id"]), "slug": r["slug"],
                 "name": r["name"], "depth": r["depth"], "node_count": total.get(r["domain_id"], 0)}
                for r in rows]

    def counts(self, rev: int | None = None) -> dict:
        """The numbers the product reports. Canonical live nodes are counted once, whatever their memberships."""
        w, a = self.at(rev)
        by_status = dict(self.q(f"SELECT status, COUNT(*) FROM node_v WHERE {w} GROUP BY status", a))
        by_type = dict(self.q(f"SELECT type, COUNT(*) FROM node_v WHERE {w} AND status='live' GROUP BY type", a))
        by_origin = dict(self.q(f"SELECT origin, COUNT(*) FROM node_v WHERE {w} AND status='live' GROUP BY origin", a))
        wm, am = self.at(rev, "m")
        wn, an = self.at(rev, "n")
        multi = self.q1(
            f"SELECT COUNT(*) FROM (SELECT m.node_id FROM member_v m JOIN node_v n ON n.node_id=m.node_id AND {wn} "
            f"WHERE {wm} AND n.status='live' GROUP BY m.node_id HAVING COUNT(*)>1)", (*an, *am))[0]
        diag = by_origin.get("diagnostic", 0)
        kp = self.q1(f"SELECT COUNT(*) FROM node_v WHERE {w} AND status='live' AND origin<>'diagnostic' AND type IN ('knowledge','procedure')", a)[0]
        by_type = dict(self.q(f"SELECT type, COUNT(*) FROM node_v WHERE {w} AND status='live' AND origin<>'diagnostic' GROUP BY type", a))
        return {
            "rev": self.head() if rev is None else rev,
            # Diagnostic canary nodes are invented test fixtures. They are reported on their own and never counted as model nodes.
            "canonical_live_nodes": by_status.get("live", 0) - diag,
            "canonical_live_knowledge_procedure_nodes": kp,
            "diagnostic_nodes": diag,
            "live_by_type": by_type, "live_by_origin": by_origin,
            "disabled_nodes": by_status.get("disabled", 0), "tombstones": by_status.get("tombstone", 0),
            "memberships": self.q1(f"SELECT COUNT(*) FROM member_v WHERE {w}", a)[0],
            "multi_membership_nodes": multi,
            "edges": self.q1(f"SELECT COUNT(*) FROM edge_v WHERE {w}", a)[0],
            "mpns": self.q1(f"SELECT COUNT(*) FROM mpn_v WHERE {w}", a)[0],
            "domains": self.q1(f"SELECT COUNT(*) FROM domain_v WHERE {w}", a)[0],
            "sources": self.q1("SELECT COUNT(*) FROM sources")[0],
        }

    # ------------------------------------------------------------------ lexical search (head)
    @staticmethod
    def fts_query(text: str) -> str:
        import re
        terms = re.findall(r"[\w][\w'\-]*", text, flags=re.UNICODE)[:16]
        if not terms:
            return ""
        return " ".join('"' + t.replace('"', "") + '"' + ("*" if i == len(terms) - 1 else "") for i, t in enumerate(terms))

    def search_lexical(self, text: str, limit: int = 20, offset: int = 0, filters: dict | None = None, any_terms: bool = False) -> list[dict]:
        fq = self.fts_query(text)
        if not fq:
            return []
        if any_terms:
            import re
            words = [w for w in re.findall(r"[\w][\w'\-]*", text, flags=re.UNICODE) if len(w) > 2 and w.lower() not in STOPWORDS][:16]
            if not words:
                return []
            fq = " OR ".join('"' + w.replace('"', "") + '"' for w in words)
        f = filters or {}
        where, args = ["node_fts MATCH ?"], [fq]
        join = "JOIN node_v n ON n.node_id = node_fts.rowid AND n.rev_to IS NULL"
        if f.get("status"):
            where.append("n.status=?"); args.append(f["status"])
        else:
            where.append("n.status<>'tombstone'")
        if f.get("type"):
            where.append("n.type=?"); args.append(f["type"])
        if f.get("validation"):
            where.append("n.validation=?"); args.append(f["validation"])
        if f.get("origin"):
            where.append("n.origin=?"); args.append(f["origin"])
        if f.get("mpn_id"):
            where.append("EXISTS(SELECT 1 FROM member_v m WHERE m.node_id=n.node_id AND m.rev_to IS NULL AND m.mpn_id=?)")
            args.append(iid(f["mpn_id"]))
        sql = (f"SELECT n.node_id, n.title, n.type, n.status, n.validation, n.summary, "
               f"bm25(node_fts, 10.0, 6.0, 3.0, 1.0, 0.5) AS score, "
               f"snippet(node_fts, -1, '[', ']', ' … ', 12) AS snip "
               f"FROM node_fts {join} WHERE {' AND '.join(where)} ORDER BY score LIMIT ? OFFSET ?")
        try:
            rows = self.q(sql, (*args, limit, offset))
        except sqlite3.OperationalError:
            return []
        return [{"id": sid(r["node_id"]), "title": r["title"], "type": r["type"], "status": r["status"],
                 "validation": r["validation"], "summary": r["summary"][:240], "score": -r["score"], "snippet": r["snip"]} for r in rows]


class Tx:
    """One edit transaction = one revision."""

    def __init__(self, store: Store, c: sqlite3.Connection, rev: int):
        self.store, self.c, self.rev = store, c, rev
        self.touched = False
        self.stats: dict = {}
        self.nodes_changed: set[int] = set()

    def _bump(self, key: str, n: int = 1):
        self.touched = True
        self.stats[key] = self.stats.get(key, 0) + n

    # -------------------------------------------------- sources and schemas (not versioned: append-only facts)
    def put_source(self, kind: str, origin: str, title: str, url: str = "", license: str = "", license_url: str = "",
                   revision: str = "", retrieved_at: str = "", attribution: str = "") -> int:
        r = self.c.execute("SELECT id FROM sources WHERE kind=? AND url=? AND revision=?", (kind, url, revision)).fetchone()
        if r:
            return r[0]
        cur = self.c.execute(
            "INSERT INTO sources(kind, origin, title, url, license, license_url, revision, retrieved_at, attribution) VALUES(?,?,?,?,?,?,?,?,?)",
            (kind, origin, title, url, license, license_url, revision, retrieved_at, attribution))
        self._bump("sources")
        return cur.lastrowid

    # -------------------------------------------------- MPNs and domains
    def put_mpn(self, slug: str, name: str, kind: str = "domain", description: str = "", version: str = "0.1.0",
                manifest: dict | None = None, installed: bool = True, enabled: bool = True, mpn_id: int | None = None) -> int:
        if mpn_id is None:
            r = self.c.execute("SELECT mpn_id FROM mpn_v WHERE slug=? AND rev_to IS NULL", (slug,)).fetchone()
            mpn_id = r[0] if r else self.store._next_id(self.c, "mpn")
        self.c.execute("UPDATE mpn_v SET rev_to=? WHERE mpn_id=? AND rev_to IS NULL", (self.rev, mpn_id))
        self.c.execute(
            "INSERT INTO mpn_v(mpn_id, slug, name, kind, description, version, manifest, installed, enabled, rev_from) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (mpn_id, slug, name, kind, description, version, json.dumps(manifest or {}), int(installed), int(enabled), self.rev))
        self._bump("mpns")
        return mpn_id

    def set_mpn_state(self, mpn_id: int, **changes) -> None:
        r = self.c.execute("SELECT * FROM mpn_v WHERE mpn_id=? AND rev_to IS NULL", (mpn_id,)).fetchone()
        if r is None:
            raise KeyError(f"no MPN {mpn_id}")
        d = dict(r)
        for k, v in changes.items():
            if k not in ("installed", "enabled", "name", "description", "version", "manifest"):
                raise ValueError(k)
            d[k] = json.dumps(v) if k == "manifest" else (int(v) if k in ("installed", "enabled") else v)
        self.put_mpn(d["slug"], d["name"], d["kind"], d["description"], d["version"], json.loads(d["manifest"]),
                     bool(d["installed"]), bool(d["enabled"]), mpn_id)

    def put_domain(self, mpn_id: int, slug: str, name: str, parent_id: int | None = None, depth: int = 1,
                   domain_id: int | None = None) -> int:
        if domain_id is None:
            r = self.c.execute("SELECT domain_id FROM domain_v WHERE mpn_id=? AND slug=? AND rev_to IS NULL", (mpn_id, slug)).fetchone()
            if r:
                return r[0]
            domain_id = self.store._next_id(self.c, "domain")
        self.c.execute("UPDATE domain_v SET rev_to=? WHERE domain_id=? AND rev_to IS NULL", (self.rev, domain_id))
        self.c.execute("INSERT INTO domain_v(domain_id, mpn_id, parent_id, slug, name, depth, rev_from) VALUES(?,?,?,?,?,?,?)",
                       (domain_id, mpn_id, parent_id, slug, name, depth, self.rev))
        self._bump("domains")
        return domain_id

    # -------------------------------------------------- nodes
    def put_node(self, *, node_id: int | None = None, uid: str | None = None, type: str = "knowledge", title: str,
                 summary: str = "", aliases: list | None = None, schema_id: str = "3cn", schema_version: int = 1,
                 status: str = "live", validation: str = "unverified", origin: str = "user", cells: dict | None = None) -> int:
        if type not in NODE_TYPES:
            raise ValueError(f"unknown node type {type!r}")
        if status not in NODE_STATUS:
            raise ValueError(f"unknown status {status!r}")
        if validation not in VALIDATION:
            raise ValueError(f"unknown validation state {validation!r}")
        if not title or not title.strip():
            raise ValueError("a node needs a title")
        aliases = aliases or []
        cells = cells or {}
        if node_id is None:
            node_id = self.store._next_id(self.c, "node")
        if uid is None:
            uid = f"yaob:u:{node_id}"
        h = content_hash(title, summary, aliases, cells, type)
        self.c.execute("UPDATE node_v SET rev_to=? WHERE node_id=? AND rev_to IS NULL", (self.rev, node_id))
        self.c.execute(
            "INSERT INTO node_v(node_id, uid, type, title, summary, aliases, schema_id, schema_version, status, validation, origin, cells, content_hash, rev_from) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (node_id, uid, type, title.strip(), summary, json.dumps(aliases, ensure_ascii=False), schema_id, schema_version,
             status, validation, origin, json.dumps(cells, ensure_ascii=False), h, self.rev))
        self._bump("nodes")
        self.nodes_changed.add(node_id)
        self._reindex_fts(node_id)
        return node_id

    def update_node(self, node_id: int, **changes) -> int:
        r = self.c.execute("SELECT * FROM node_v WHERE node_id=? AND rev_to IS NULL", (node_id,)).fetchone()
        if r is None:
            raise KeyError(f"no node {node_id}")
        d = {"uid": r["uid"], "type": r["type"], "title": r["title"], "summary": r["summary"], "aliases": json.loads(r["aliases"]),
             "schema_id": r["schema_id"], "schema_version": r["schema_version"], "status": r["status"],
             "validation": r["validation"], "origin": r["origin"], "cells": json.loads(r["cells"])}
        for k, v in changes.items():
            if k not in d:
                raise ValueError(f"cannot set {k}")
            d[k] = v
        return self.put_node(node_id=node_id, **d)

    def delete_node(self, node_id: int) -> None:
        """Recoverable deletion: the node becomes a tombstone. Its history, memberships and edges stay readable at
        earlier revisions and are restored by undo. Nothing is physically removed."""
        self.update_node(node_id, status="tombstone")

    # -------------------------------------------------- memberships
    def set_membership(self, node_id: int, mpn_id: int, domain_id: int | None = None, role: str | None = None,
                       evidence: dict | None = None) -> None:
        cur = self.c.execute("SELECT * FROM member_v WHERE node_id=? AND rev_to IS NULL", (node_id,)).fetchall()
        existing = next((m for m in cur if m["mpn_id"] == mpn_id), None)
        if role is None:
            role = existing["role"] if existing else ("primary" if not cur else "shared")
        if role not in ("primary", "shared"):
            raise ValueError(role)
        if role == "primary":   # exactly one primary membership per node
            for m in cur:
                if m["role"] == "primary" and m["mpn_id"] != mpn_id:
                    self.c.execute("UPDATE member_v SET rev_to=? WHERE vid=?", (self.rev, m["vid"]))
                    self.c.execute("INSERT INTO member_v(node_id, mpn_id, domain_id, role, evidence, rev_from) VALUES(?,?,?,?,?,?)",
                                   (node_id, m["mpn_id"], m["domain_id"], "shared", m["evidence"], self.rev))
        if existing:
            self.c.execute("UPDATE member_v SET rev_to=? WHERE vid=?", (self.rev, existing["vid"]))
            if evidence is None:
                evidence = json.loads(existing["evidence"])
            if domain_id is None:
                domain_id = existing["domain_id"]
        self.c.execute("INSERT INTO member_v(node_id, mpn_id, domain_id, role, evidence, rev_from) VALUES(?,?,?,?,?,?)",
                       (node_id, mpn_id, domain_id, role, json.dumps(evidence or {}), self.rev))
        self._bump("memberships")
        self.nodes_changed.add(node_id)

    def remove_membership(self, node_id: int, mpn_id: int) -> None:
        """Remove one membership. The canonical node and its other memberships are untouched. A node always keeps
        at least one membership; if the primary one is removed, the next shared one becomes primary."""
        cur = self.c.execute("SELECT * FROM member_v WHERE node_id=? AND rev_to IS NULL", (node_id,)).fetchall()
        target = next((m for m in cur if m["mpn_id"] == mpn_id), None)
        if target is None:
            raise KeyError("no such membership")
        others = [m for m in cur if m["mpn_id"] != mpn_id]
        if not others:
            raise ValueError("this is the node's only membership; delete the node or add another membership first")
        self.c.execute("UPDATE member_v SET rev_to=? WHERE vid=?", (self.rev, target["vid"]))
        if target["role"] == "primary":
            promote = others[0]
            self.c.execute("UPDATE member_v SET rev_to=? WHERE vid=?", (self.rev, promote["vid"]))
            self.c.execute("INSERT INTO member_v(node_id, mpn_id, domain_id, role, evidence, rev_from) VALUES(?,?,?,?,?,?)",
                           (node_id, promote["mpn_id"], promote["domain_id"], "primary", promote["evidence"], self.rev))
        self._bump("memberships_removed")
        self.nodes_changed.add(node_id)

    # -------------------------------------------------- edges
    def put_edge(self, src: int, dst: int, type: str, weight: float = 1.0, origin: str = "user", evidence: dict | None = None,
                 edge_id: int | None = None) -> int:
        if src == dst:
            raise ValueError("an edge needs two different nodes")
        if edge_id is None:
            r = self.c.execute("SELECT edge_id FROM edge_v WHERE src=? AND dst=? AND type=? AND rev_to IS NULL", (src, dst, type)).fetchone()
            if r:
                return r[0]
            edge_id = self.store._next_id(self.c, "edge")
        else:
            self.c.execute("UPDATE edge_v SET rev_to=? WHERE edge_id=? AND rev_to IS NULL", (self.rev, edge_id))
        self.c.execute("INSERT INTO edge_v(edge_id, src, dst, type, weight, origin, evidence, rev_from) VALUES(?,?,?,?,?,?,?,?)",
                       (edge_id, src, dst, type, weight, origin, json.dumps(evidence or {}), self.rev))
        self._bump("edges")
        return edge_id

    def remove_edge(self, edge_id: int) -> None:
        n = self.c.execute("UPDATE edge_v SET rev_to=? WHERE edge_id=? AND rev_to IS NULL", (self.rev, edge_id)).rowcount
        if not n:
            raise KeyError(f"no edge {edge_id}")
        self._bump("edges_removed")

    # -------------------------------------------------- node sources
    def link_source(self, node_id: int, source_id: int, locator: str = "") -> None:
        r = self.c.execute("SELECT vid FROM node_source_v WHERE node_id=? AND source_id=? AND rev_to IS NULL", (node_id, source_id)).fetchone()
        if r:
            return
        self.c.execute("INSERT INTO node_source_v(node_id, source_id, locator, rev_from) VALUES(?,?,?,?)",
                       (node_id, source_id, locator, self.rev))
        self._bump("node_sources")
        self.nodes_changed.add(node_id)
        self._reindex_fts(node_id)

    def unlink_source(self, node_id: int, source_id: int) -> None:
        self.c.execute("UPDATE node_source_v SET rev_to=? WHERE node_id=? AND source_id=? AND rev_to IS NULL", (self.rev, node_id, source_id))
        self._bump("node_sources_removed")
        self.nodes_changed.add(node_id)
        self._reindex_fts(node_id)

    # -------------------------------------------------- head lexical index
    def _reindex_fts(self, node_id: int) -> None:
        self.c.execute("DELETE FROM node_fts WHERE rowid=?", (node_id,))
        r = self.c.execute("SELECT * FROM node_v WHERE node_id=? AND rev_to IS NULL", (node_id,)).fetchone()
        if r is None:
            return
        # tombstones stay indexed so the deleted filter can find them; default searches exclude them by status
        srcs = self.c.execute(
            "SELECT s.title, s.kind FROM node_source_v ns JOIN sources s ON s.id=ns.source_id WHERE ns.node_id=? AND ns.rev_to IS NULL",
            (node_id,)).fetchall()
        self.c.execute("INSERT INTO node_fts(rowid, title, aliases, summary, body, sources) VALUES(?,?,?,?,?,?)",
                       (node_id, r["title"], " ; ".join(json.loads(r["aliases"])), r["summary"],
                        cell_text(json.loads(r["cells"])), " ; ".join(f"{s['kind']} {s['title']}" for s in srcs)))
