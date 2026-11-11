"""MPN Studio HTTP API and web UI server.

This process holds no model. It owns the model store (SQLite), board geometry, jobs and the
static web UI, and it proxies run/chat requests to the runtime process on loopback. If the
runtime is down, the board, editing and text search keep working and chat reports that the
runtime is unavailable.

Access: trusted-network only. Every request is refused unless the TCP peer address lies inside
MPN_ALLOWED_NETS. There is no password. Cross-site browser writes are refused by an Origin check.
"""
from __future__ import annotations

import ipaddress
import json
import os
import time
from pathlib import Path

import httpx
import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse

from .. import __version__
from ..store import layout, schemas
from ..store.db import Store, iid, now, sid
from ..train import jobs
from . import ops

BOARD = ops.BOARD
RUNTIME_URL = os.environ.get("MPN_RUNTIME_URL", "http://127.0.0.1:8841")
ALLOWED = [ipaddress.ip_network(n.strip()) for n in os.environ.get("MPN_ALLOWED_NETS", "127.0.0.0/8,::1/128").split(",") if n.strip()]
WEB_DIR = Path(os.environ.get("MPN_WEB", Path(__file__).resolve().parents[2] / "web-dist"))
if not WEB_DIR.exists():
    WEB_DIR = Path(__file__).resolve().parents[2] / "web" / "dist"

DEFAULT_BUDGET = {
    "preset": "balanced",
    "total_gib": 0,                 # 0 = derive from available memory minus reserve
    "reserve_gib": 16.0,            # left for the OS and other services
    "resident_cache_mib": 512,      # learned MPN state kept in memory
    "max_resident_mpns": 8,
    "max_active_mpns": 3,           # MPNs one request may consult
    # Specialists applied at once. 0 since held-out eval-001 (docs/measurements/heldout-eval-001): routing + node text
    # without specialists scored 85 % vs 81 % with one, and abstained on 100 % vs 50 % of unanswerable questions.
    # Specialists stay available as the "specialists (experimental)" preset. Pilot 2: four stacked fell to about 20 %.
    "max_active_adapters": 0,
    "retrieval_candidates": 64,
    "selected_nodes": 8,            # nodes chosen by the router
    "text_nodes": 3,                # of those, how many are also given as text
    "traversal_depth": 1,
    "referral_mode": "relation",    # "plain" is the rule before 12 Nov 2026 (see runtime/engine.py)
    "fact_focus": True,             # a node's facts that the question names are rendered first
    "context_tokens": 8192,         # min(this, the core's max_context) is the context one request gets
    "output_tokens": 1024,          # answer cap; a client's max_tokens is honoured up to this
    "concurrent_requests": 2,
    "prefetch_mib": 0,
    "disk_cache_gib": 4.0,
}
PRESETS = {
    "conservative": {"resident_cache_mib": 128, "max_resident_mpns": 3, "max_active_mpns": 2, "max_active_adapters": 0, "retrieval_candidates": 24, "selected_nodes": 4,
                     "text_nodes": 2, "traversal_depth": 0, "context_tokens": 2048, "output_tokens": 256, "concurrent_requests": 1},
    "balanced": {k: DEFAULT_BUDGET[k] for k in ("resident_cache_mib", "max_resident_mpns", "max_active_mpns", "max_active_adapters", "retrieval_candidates",
                                                "selected_nodes", "text_nodes", "traversal_depth", "context_tokens", "output_tokens", "concurrent_requests")},
    "expanded": {"resident_cache_mib": 2048, "max_resident_mpns": 18, "max_active_mpns": 5, "max_active_adapters": 0, "retrieval_candidates": 160, "selected_nodes": 16,
                 "text_nodes": 5, "traversal_depth": 2, "context_tokens": 16384, "output_tokens": 2048, "concurrent_requests": 3},
}
# The configuration used before eval-001 showed specialists hurting abstention. For research; not a default.
PRESETS["specialists (experimental)"] = dict(PRESETS["balanced"], max_active_adapters=1)


def get_budget(store: Store) -> dict:
    from ..runtime.budget import repair_budget
    b = dict(DEFAULT_BUDGET)
    b.update(store.get_setting("budget") or {})
    b, note = repair_budget(b)          # a budget saved before the context/output check may hold e.g. 4096/4096
    if note:
        b["repair_note"] = note
    return b


def create_app(db_path: str | Path | None = None) -> FastAPI:
    store = Store(db_path or jobs.db_path())
    schemas.seed_schemas(store)
    layout.ensure_tiles(store, BOARD)
    app = FastAPI(title="MPN Studio", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.store = store
    client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=2.0))

    # ------------------------------------------------------------------ access control
    @app.middleware("http")
    async def guard(request: Request, call_next):
        peer = request.client.host if request.client else ""
        try:
            addr = ipaddress.ip_address(peer)
            ok = any(addr in n for n in ALLOWED)
        except ValueError:
            ok = peer == "testclient"
        if not ok:
            return PlainTextResponse("This address is outside the networks allowed to use MPN Studio.", status_code=403)
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            site = request.headers.get("sec-fetch-site", "")
            host = request.headers.get("host", "")
            if site == "cross-site" or (origin and origin != "null" and origin.split("://", 1)[-1] != host) or origin == "null":
                return PlainTextResponse("Cross-site requests are refused.", status_code=403)
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/") or request.url.path.startswith("/v1/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.exception_handler(ops.OpError)
    async def op_error(_: Request, e: ops.OpError):
        return JSONResponse({"error": str(e), **({"detail": e.detail} if e.detail else {})}, status_code=e.status)

    @app.exception_handler(ValueError)
    async def value_error(_: Request, e: ValueError):
        return JSONResponse({"error": str(e)}, status_code=400)

    def nid(s: str) -> int:
        try:
            return iid(s)
        except ValueError:
            raise HTTPException(400, "ids are decimal strings")

    async def runtime_get(path: str, timeout: float = 3.0):
        try:
            r = await client.get(RUNTIME_URL + path, timeout=timeout)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    async def semantic(q: str, limit: int, filters: dict):
        r = await client.post(RUNTIME_URL + "/semantic", json={"q": q, "limit": limit, "filters": filters}, timeout=8.0)
        r.raise_for_status()
        return r.json()["results"]

    # ------------------------------------------------------------------ status
    @app.get("/api/health")
    async def health():
        return {"ok": True, "version": __version__, "head": store.head(), "time": now()}

    @app.get("/api/status")
    async def status():
        rt = await runtime_get("/status")
        vm = psutil.virtual_memory()
        return {"version": __version__, "counts": store.counts(), "dirty": ops.dirty_summary(store),
                "runtime": rt or {"available": False, "reason": "The runtime process is not answering on " + RUNTIME_URL},
                "memory": {"total_gib": round(vm.total / 2**30, 1), "available_gib": round(vm.available / 2**30, 1),
                           "api_process_mib": round(psutil.Process().memory_info().rss / 2**20, 1),
                           "swap_used_gib": round(psutil.swap_memory().used / 2**30, 2)},
                "budget": get_budget(store), "undo": store.undo_target(), "redo": store.redo_target(),
                "jobs_active": store.q1("SELECT COUNT(*) FROM jobs WHERE state IN ('queued','running','cancelling')")[0],
                "corpus": store.get_setting("corpus"), "release": store.get_setting("active_release")}

    # ------------------------------------------------------------------ board
    @app.get("/api/board")
    async def board():
        b = store.q1("SELECT * FROM boards WHERE id=?", (BOARD,))
        return {"id": sid(BOARD), "name": b["name"] if b else "Yaob network", "view": json.loads(b["view"]) if b else {},
                "bounds": layout.bounds(store, BOARD), "spacing": layout.SPACING,
                "mpns": layout.region_list(store, BOARD, None, kinds=("mpn",)), "bundles": layout.mpn_bundles(store),
                "saved_views": [dict(r, id=sid(r["id"])) for r in map(dict, store.q("SELECT id, name, x, y, zoom, created FROM saved_views WHERE board_id=? ORDER BY name", (BOARD,)))],
                "objects": [{"id": sid(r["id"]), "kind": r["kind"], "x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"], "props": json.loads(r["props"])}
                            for r in store.q("SELECT * FROM board_objects WHERE board_id=?", (BOARD,))],
                "head": store.head()}

    @app.post("/api/viewport")
    async def viewport(body: dict):
        t0 = time.perf_counter()
        bbox = [float(v) for v in body["bbox"]]
        out = layout.viewport(store, BOARD, bbox, float(body.get("scale", 1.0)), int(body.get("cap", 2500)), int(body.get("labels", 160)))
        if body.get("edges") and out.get("level") == "nodes":
            out["edges"] = layout.viewport_edges(store, [iid(n["id"]) for n in out["nodes"]], int(body.get("edge_cap", 1500)))
        out["head"] = store.head()
        out["ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return out

    @app.put("/api/board/view")
    async def save_view_state(body: dict):
        store.conn().execute("UPDATE boards SET view=? WHERE id=?", (json.dumps({k: body.get(k) for k in ("x", "y", "zoom", "panels")}), BOARD))
        return {"ok": True}

    @app.post("/api/board/views")
    async def add_view(body: dict):
        name = (body.get("name") or "").strip()
        if not name:
            raise ops.OpError("A saved view needs a name.")
        cur = store.conn().execute("INSERT INTO saved_views(board_id, name, x, y, zoom, created) VALUES(?,?,?,?,?,?)",
                                   (BOARD, name[:80], float(body["x"]), float(body["y"]), float(body["zoom"]), now()))
        return {"id": sid(cur.lastrowid)}

    @app.delete("/api/board/views/{view_id}")
    async def del_view(view_id: str):
        store.conn().execute("DELETE FROM saved_views WHERE id=? AND board_id=?", (nid(view_id), BOARD))
        return {"ok": True}

    @app.post("/api/layout/move")
    async def move(body: dict):
        moves = [(nid(m["id"]), float(m["x"]), float(m["y"])) for m in body.get("moves", [])][:5000]
        return {"moved": layout.move_nodes(store, BOARD, moves), "head": store.head(),
                "note": "Geometry only. No model revision was created."}

    @app.post("/api/layout/recompute")
    async def recompute():
        return layout.compute_layout(store, BOARD)

    @app.post("/api/board/objects")
    async def add_object(body: dict):
        kind = body.get("kind")
        if kind not in ("group", "note", "panel"):
            raise ops.OpError("kind must be group, note or panel")
        cur = store.conn().execute("INSERT INTO board_objects(board_id, kind, x, y, w, h, props, created) VALUES(?,?,?,?,?,?,?,?)",
                                   (BOARD, kind, float(body["x"]), float(body["y"]), float(body.get("w", 0)), float(body.get("h", 0)),
                                    json.dumps(body.get("props", {})), now()))
        return {"id": sid(cur.lastrowid)}

    @app.patch("/api/board/objects/{oid}")
    async def patch_object(oid: str, body: dict):
        r = store.q1("SELECT * FROM board_objects WHERE id=? AND board_id=?", (nid(oid), BOARD))
        if r is None:
            raise HTTPException(404)
        props = json.loads(r["props"])
        props.update(body.get("props", {}))
        store.conn().execute("UPDATE board_objects SET x=?, y=?, w=?, h=?, props=? WHERE id=?",
                             (float(body.get("x", r["x"])), float(body.get("y", r["y"])), float(body.get("w", r["w"])), float(body.get("h", r["h"])),
                              json.dumps(props), nid(oid)))
        return {"ok": True}

    @app.delete("/api/board/objects/{oid}")
    async def del_object(oid: str):
        store.conn().execute("DELETE FROM board_objects WHERE id=? AND board_id=?", (nid(oid), BOARD))
        return {"ok": True}

    # ------------------------------------------------------------------ nodes
    @app.get("/api/nodes/{node_id}")
    async def node(node_id: str, rev: int | None = None):
        return ops.node_detail(store, nid(node_id), rev)

    @app.post("/api/nodes")
    async def node_create(body: dict):
        return ops.create_node(store, body)

    @app.patch("/api/nodes/{node_id}")
    async def node_update(node_id: str, body: dict):
        return ops.update_node(store, nid(node_id), body)

    @app.get("/api/nodes/{node_id}/delete-preview")
    async def node_delete_preview(node_id: str):
        return ops.delete_preview(store, nid(node_id))

    @app.delete("/api/nodes/{node_id}")
    async def node_delete(node_id: str):
        return ops.delete_node(store, nid(node_id))

    @app.post("/api/nodes/{node_id}/restore")
    async def node_restore(node_id: str):
        return ops.restore_node(store, nid(node_id))

    @app.post("/api/nodes/{node_id}/duplicate")
    async def node_duplicate(node_id: str, body: dict):
        return ops.duplicate_node(store, nid(node_id), body.get("mode", "canonical"), position=body.get("position"))

    @app.get("/api/nodes/{node_id}/edges")
    async def node_edges(node_id: str, offset: int = 0, limit: int = 50):
        return {"total": store.edge_count(nid(node_id)), "edges": store.edges_of(nid(node_id), limit=min(limit, 200), offset=offset)}

    @app.get("/api/nodes/{node_id}/history")
    async def node_hist(node_id: str):
        return {"versions": ops.node_history(store, nid(node_id))}

    @app.post("/api/nodes/validate")
    async def node_validate(body: dict):
        _, problems = ops.validate_node_payload(store, body)
        return {"problems": problems}

    @app.put("/api/nodes/{node_id}/memberships/{mpn_id}")
    async def member_put(node_id: str, mpn_id: str, body: dict | None = None):
        body = body or {}
        return ops.set_membership(store, nid(node_id), nid(mpn_id), body.get("role"), nid(body["domain_id"]) if body.get("domain_id") else None)

    @app.delete("/api/nodes/{node_id}/memberships/{mpn_id}")
    async def member_del(node_id: str, mpn_id: str):
        return ops.remove_membership(store, nid(node_id), nid(mpn_id))

    @app.post("/api/edges")
    async def edge_create(body: dict):
        return ops.create_edge(store, nid(body["src"]), nid(body["dst"]), body.get("type", "related_to"))

    @app.delete("/api/edges/{edge_id}")
    async def edge_delete(edge_id: str):
        return ops.remove_edge(store, nid(edge_id))

    @app.get("/api/edge-types")
    async def edge_types():
        used = [r[0] for r in store.q("SELECT type FROM edge_v WHERE rev_to IS NULL GROUP BY type ORDER BY COUNT(*) DESC LIMIT 60")]
        return {"types": list(dict.fromkeys(list(ops.EDGE_TYPES) + used))}

    # ------------------------------------------------------------------ search
    @app.get("/api/search")
    async def search(q: str = "", mode: str = "hybrid", limit: int = 20, offset: int = 0, type: str = "", status: str = "",
                     validation: str = "", origin: str = "", mpn_id: str = ""):
        filters = {"type": type, "status": status, "validation": validation, "origin": origin, "mpn_id": mpn_id}
        sem_results = None
        sem_err = None
        if q.strip() and mode in ("semantic", "hybrid"):
            try:
                sem_results = await semantic(q, min(limit, 100) + offset + 20, {k: v for k, v in filters.items() if v})
            except Exception as e:
                sem_err = "The runtime is not available, so results are by text only." if mode == "hybrid" else "Search by meaning needs the runtime, which is not available."
                del e
        out = ops.search(store, q, mode, min(limit, 100), offset, filters, (lambda *_: sem_results) if sem_results is not None else None)
        if sem_err:
            out["notice"] = sem_err
        return out

    # ------------------------------------------------------------------ MPNs
    @app.get("/api/mpns")
    async def mpns(rev: int | None = None):
        rt = await runtime_get("/residency")
        res = {m["mpn_id"]: m for m in (rt or {}).get("mpns", [])}
        out = []
        for m in store.mpns(rev):
            mid = iid(m["id"])
            trained = store.q1("SELECT COUNT(*) FROM member_v a JOIN node_v n ON n.node_id=a.node_id AND n.rev_to IS NULL AND n.status='live' "
                               "JOIN derived_state d ON d.node_id=a.node_id WHERE a.rev_to IS NULL AND a.mpn_id=? AND a.role='primary' AND d.learned_hash = n.content_hash", (mid,))[0]
            own = store.q1("SELECT COUNT(*) FROM member_v a JOIN node_v n ON n.node_id=a.node_id AND n.rev_to IS NULL AND n.status='live' "
                           "WHERE a.rev_to IS NULL AND a.mpn_id=? AND a.role='primary'", (mid,))[0]
            m["trained_nodes"], m["primary_nodes"] = trained, own
            m["training"] = "untrained" if trained == 0 else ("trained" if trained == own else "partly trained")
            m["resident"] = bool(res.get(m["id"], {}).get("resident"))
            m["resident_bytes"] = res.get(m["id"], {}).get("bytes", 0)
            m["validated"] = bool(m["manifest"].get("evaluation", {}).get("validated"))
            out.append(m)
        return {"mpns": out, "runtime_available": rt is not None}

    @app.post("/api/mpns")
    async def mpn_create(body: dict):
        return ops.create_mpn(store, body)

    @app.get("/api/mpns/{mpn_id}")
    async def mpn_get(mpn_id: str):
        m = store.mpn(nid(mpn_id))
        if m is None:
            raise HTTPException(404)
        m["domains"] = store.domains(nid(mpn_id))
        return m

    @app.get("/api/mpns/{mpn_id}/impact")
    async def mpn_impact(mpn_id: str, action: str):
        return ops.mpn_impact(store, nid(mpn_id), action)

    @app.post("/api/mpns/{mpn_id}/{action}")
    async def mpn_action(mpn_id: str, action: str):
        out = ops.set_mpn(store, nid(mpn_id), action)
        try:
            await client.post(RUNTIME_URL + "/invalidate", json={"reason": f"mpn {action}", "mpn_id": mpn_id}, timeout=3.0)
        except Exception:
            pass
        return out

    # ------------------------------------------------------------------ packages and bundles (files live under the data directory)
    def _safe_dir(kind: str, name: str) -> Path:
        import re
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}", name or ""):
            raise ops.OpError("Use a name of letters, digits, dot, dash or underscore.")
        return jobs.data_root() / kind / name

    @app.post("/api/mpns/{mpn_id}/export")
    async def mpn_export(mpn_id: str, body: dict | None = None):
        from ..store import bundle
        m = store.mpn(nid(mpn_id))
        if m is None:
            raise HTTPException(404)
        out = _safe_dir("packages", (body or {}).get("name") or f"{m['slug']}-{m['version']}-rev{store.head()}")
        if out.exists():
            raise ops.OpError(f"{out} already exists.", 409)
        man = bundle.export_mpn(store, nid(mpn_id), out, None, bundle.active_checkpoint_dir(store))
        return {"path": str(out), "manifest": {k: v for k, v in man.items() if k != "files"}, "files": len(man["files"])}

    @app.get("/api/packages")
    async def packages():
        root = jobs.data_root() / "packages"
        out = []
        for d in sorted(root.glob("*/manifest.json")) if root.exists() else []:
            try:
                man = json.loads(d.read_text())
                out.append({"name": d.parent.name, "path": str(d.parent), "slug": man.get("slug"), "title": man.get("name"), "version": man.get("version"),
                            "class": man.get("class"), "counts": man.get("counts"), "licences": man.get("licences"), "installs_code": man.get("installs_code", False),
                            "installed": store.mpn_by_slug(man.get("slug", "")) is not None})
            except Exception:
                continue
        return {"packages": out, "directory": str(root)}

    @app.post("/api/packages/{name}/install")
    async def package_install(name: str):
        from ..store import bundle
        pkg = _safe_dir("packages", name)
        if not (pkg / "manifest.json").exists():
            raise HTTPException(404)
        return bundle.import_mpn(store, pkg)

    @app.post("/api/bundles")
    async def bundle_export(body: dict):
        from ..store import bundle
        out = _safe_dir("bundles", body.get("name") or f"yaob-n10k-rev{store.head()}")
        if out.exists():
            raise ops.OpError(f"{out} already exists.", 409)
        card = jobs.data_root() / "MODEL_CARD.md"
        man = bundle.export_bundle(store, out, body.get("model_name") or "yaob-n10k", bundle.active_checkpoint_dir(store),
                                   store.get_setting("core_reference") or {}, card.read_text() if card.exists() else None)
        return {"path": str(out), "revision": man["revision"], "counts": man["counts"], "files": len(man["files"]), "checkpoint": man["checkpoint"]}

    # ------------------------------------------------------------------ schemas
    @app.get("/api/schemas")
    async def schema_list():
        return {"schemas": schemas.list_schemas(store), "runtime_roles": schemas.RUNTIME_ROLES, "field_types": schemas.FIELD_TYPES}

    @app.post("/api/schemas/validate")
    async def schema_validate(body: dict):
        problems = schemas.validate_schema(body)
        prev = schemas.get_schema(store, body.get("id", ""), body.get("version")) or (
            schemas.get_schema(store, body.get("id", ""), body.get("version", 1) - 1) if isinstance(body.get("version"), int) else None)
        return {"problems": problems, "diff": schemas.diff_schemas(prev, body) if prev and not problems else None,
                "compared_with": {"id": prev["id"], "version": prev["version"]} if prev else None}

    @app.put("/api/schemas")
    async def schema_save(body: dict):
        return schemas.save_schema(store, body)

    @app.get("/api/schemas/migration")
    async def schema_migration(from_id: str, from_version: int, to_id: str, to_version: int):
        return ops.schema_migration_preview(store, from_id, from_version, to_id, to_version)

    @app.post("/api/schemas/migrate")
    async def schema_migrate(body: dict):
        ids = [nid(x) for x in body.get("node_ids") or []] or None
        return ops.schema_migrate(store, body["from_id"], int(body["from_version"]), body["to_id"], int(body["to_version"]), ids)

    # ------------------------------------------------------------------ revisions
    @app.get("/api/revisions")
    async def revisions(limit: int = 40, before: int | None = None):
        return {"head": store.head(), "revisions": store.revisions(min(limit, 200), before), "undo": store.undo_target(), "redo": store.redo_target(),
                "releases": [dict(r, manifest=json.loads(r["manifest"])) for r in map(dict, store.q("SELECT * FROM releases ORDER BY created DESC"))]}

    @app.post("/api/undo")
    async def undo():
        try:
            rev = store.undo()
        except ValueError as e:
            raise ops.OpError(str(e), 409)
        return {"rev": rev, "undone": rev is not None, "head": store.head(), "undo": store.undo_target(), "redo": store.redo_target()}

    @app.post("/api/redo")
    async def redo():
        try:
            rev = store.redo()
        except ValueError as e:
            raise ops.OpError(str(e), 409)
        return {"rev": rev, "redone": rev is not None, "head": store.head(), "undo": store.undo_target(), "redo": store.redo_target()}

    @app.post("/api/releases")
    async def release_create(body: dict):
        name = (body.get("name") or "").strip()
        if not name or store.q1("SELECT 1 FROM releases WHERE name=?", (name,)):
            raise ops.OpError("A release needs a new, non-empty name.")
        rev = int(body.get("rev") or store.head())
        store.conn().execute("INSERT INTO releases(name, rev, created, manifest) VALUES(?,?,?,?)",
                             (name, rev, now(), json.dumps({"counts": store.counts(rev), "note": body.get("note", "")})))
        return {"name": name, "rev": rev}

    # ------------------------------------------------------------------ jobs
    @app.get("/api/jobs")
    async def job_list(limit: int = 30):
        jobs.reconcile(store)
        return {"jobs": jobs.list_jobs(store, min(limit, 100)), "memory": jobs.memory_status(store)}

    @app.post("/api/jobs")
    async def job_submit(body: dict):
        kind = body.get("kind")
        if kind not in ("train_bank", "evaluate", "reindex", "package", "sleep", "retrain_nodes"):
            raise ops.OpError("Unknown job kind.")
        return {"id": jobs.submit(store, kind, body.get("params") or {})}

    @app.get("/api/jobs/{job_id}")
    async def job_get(job_id: str):
        j = jobs.get(store, job_id)
        if j is None:
            raise HTTPException(404)
        return j

    @app.get("/api/jobs/{job_id}/log")
    async def job_log(job_id: str, offset: int = 0, limit: int = 65536):
        j = jobs.get(store, job_id)
        if j is None:
            raise HTTPException(404)
        p = Path(j["log_path"])
        if not p.exists():
            return {"offset": 0, "next": 0, "text": "", "size": 0}
        size = p.stat().st_size
        if offset < 0:
            offset = max(0, size + offset)
        with p.open("rb") as f:
            f.seek(min(offset, size))
            data = f.read(min(limit, 262144))
        return {"offset": offset, "next": offset + len(data), "size": size, "text": data.decode("utf-8", "replace")}

    @app.post("/api/jobs/{job_id}/cancel")
    async def job_cancel(job_id: str):
        return jobs.cancel(store, job_id)

    @app.post("/api/jobs/{job_id}/resume")
    async def job_resume(job_id: str):
        return {"id": jobs.resume(store, job_id)}

    @app.get("/api/checkpoints")
    async def checkpoints():
        rows = store.q("SELECT * FROM checkpoints ORDER BY created DESC LIMIT 60")
        return {"checkpoints": [dict(r, metrics=json.loads(r["metrics"])) for r in map(dict, rows)],
                "banks": [dict(r, spec=json.loads(r["spec"])) for r in map(dict, store.q("SELECT * FROM banks"))]}

    @app.post("/api/checkpoints/{ck_id}/activate")
    async def checkpoint_activate(ck_id: str):
        r = store.q1("SELECT * FROM checkpoints WHERE id=?", (ck_id,))
        if r is None:
            raise HTTPException(404)
        try:
            resp = await client.post(RUNTIME_URL + "/activate", json={"checkpoint": ck_id}, timeout=120.0)
        except Exception:
            raise ops.OpError("The runtime is not available, so the checkpoint cannot be loaded now.", 503)
        return JSONResponse(resp.json(), status_code=resp.status_code)

    # ------------------------------------------------------------------ budget and runtime
    @app.get("/api/budget")
    async def budget_get():
        ms = jobs.memory_status(store)
        derived = {}
        for name, p in PRESETS.items():
            frac = {"conservative": 0.25, "expanded": 0.8}.get(name, 0.5)
            derived[name] = dict(p, total_gib=round(max(ms["available_gib"] - get_budget(store)["reserve_gib"], 0) * frac, 1))
        return {"budget": get_budget(store), "presets": derived, "memory": ms,
                "note": "Preset totals are derived from memory available now minus the reserve. On this machine CPU and GPU share one pool, so there is one total."}

    @app.put("/api/budget")
    async def budget_put(body: dict):
        b = get_budget(store)
        if body.get("preset") in PRESETS and body.get("apply_preset"):
            b.update(PRESETS[body["preset"]])
            b["preset"] = body["preset"]
        limits = {"total_gib": (0, 4096), "reserve_gib": (2, 512), "resident_cache_mib": (16, 262144), "max_resident_mpns": (1, 4096),
                  "max_active_mpns": (1, 64), "max_active_adapters": (0, 8), "retrieval_candidates": (4, 2000), "selected_nodes": (1, 64), "text_nodes": (0, 16),
                  "traversal_depth": (0, 4), "context_tokens": (512, 131072), "output_tokens": (16, 8192), "concurrent_requests": (1, 16),
                  "prefetch_mib": (0, 65536), "disk_cache_gib": (0, 4096)}
        for k, (lo, hi) in limits.items():
            if k in body:
                v = float(body[k])
                if not lo <= v <= hi:
                    raise ops.OpError(f"{k} must be between {lo} and {hi}.")
                b[k] = v if k.endswith("_gib") else int(v)
        if "referral_mode" in body:
            if body["referral_mode"] not in ("plain", "relation"):
                raise ops.OpError('referral_mode must be "plain" or "relation".')
            b["referral_mode"] = body["referral_mode"]
        if "fact_focus" in body:
            if not isinstance(body["fact_focus"], bool):
                raise ops.OpError("fact_focus must be true or false.")
            b["fact_focus"] = body["fact_focus"]
        if "preset" in body and not body.get("apply_preset"):
            b["preset"] = "custom"
        if b["text_nodes"] > b["selected_nodes"]:
            raise ops.OpError("text_nodes cannot exceed selected_nodes.")
        from ..runtime.budget import check_budget_tokens
        why = check_budget_tokens(b["context_tokens"], b["output_tokens"])
        if why:
            raise ops.OpError(why)
        b.pop("repair_note", None)
        store.set_setting("budget", b)
        try:
            await client.post(RUNTIME_URL + "/budget", json=b, timeout=5.0)
        except Exception:
            pass
        return {"budget": b}

    @app.get("/api/runtime")
    async def runtime_status():
        rt = await runtime_get("/status")
        return rt or {"available": False, "reason": "The runtime process is not answering on " + RUNTIME_URL}

    @app.post("/api/runtime/{action}")
    async def runtime_action(action: str, body: dict | None = None):
        if action not in ("load", "unload", "reindex", "invalidate", "evict"):
            raise HTTPException(404)
        try:
            r = await client.post(RUNTIME_URL + "/" + action, json=body or {}, timeout=900.0)
            return JSONResponse(r.json(), status_code=r.status_code)
        except Exception:
            raise ops.OpError("The runtime is not available.", 503)

    async def proxy_stream(path: str, body: dict):
        async def gen():
            try:
                async with client.stream("POST", RUNTIME_URL + path, json=body, timeout=httpx.Timeout(900.0, connect=2.0)) as r:
                    if r.status_code != 200:
                        text = (await r.aread()).decode("utf-8", "replace")
                        yield f"event: error\ndata: {json.dumps({'error': text[:500], 'status': r.status_code})}\n\n"
                        return
                    async for chunk in r.aiter_raw():
                        yield chunk
            except (httpx.ConnectError, httpx.ConnectTimeout):
                yield f"event: error\ndata: {json.dumps({'error': 'The runtime is not running, so nothing can be generated. The board and editing still work.', 'status': 503})}\n\n"
            except httpx.HTTPError as e:
                yield f"event: error\ndata: {json.dumps({'error': f'The connection to the runtime was lost: {type(e).__name__}', 'status': 502})}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    @app.post("/api/run")
    async def run(body: dict):
        body = dict(body, budget=get_budget(store))
        return await proxy_stream("/run", body)

    @app.post("/api/run/{request_id}/cancel")
    async def run_cancel(request_id: str):
        try:
            r = await client.post(RUNTIME_URL + f"/cancel/{request_id}", timeout=5.0)
            return r.json()
        except Exception:
            raise ops.OpError("The runtime is not available.", 503)

    # ------------------------------------------------------------------ OpenAI-compatible endpoint
    @app.get("/v1/models")
    async def v1_models():
        rel = store.get_setting("active_release") or {}
        return {"object": "list", "data": [{"id": "yaob-n10k", "object": "model", "owned_by": "fylume", "created": 0,
                                            "yaob": {"revision": store.head(), "release": rel}}]}

    @app.post("/v1/chat/completions")
    async def v1_chat(body: dict):
        body = dict(body, budget=get_budget(store))
        if body.get("stream"):
            return await proxy_stream("/v1/chat/completions", body)
        try:
            r = await client.post(RUNTIME_URL + "/v1/chat/completions", json=body, timeout=900.0)
            return JSONResponse(r.json(), status_code=r.status_code)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return JSONResponse({"error": {"message": "The Yaob runtime is not running.", "type": "runtime_unavailable"}}, status_code=503)

    # ------------------------------------------------------------------ web UI
    @app.get("/{path:path}")
    async def web(path: str):
        if path.startswith(("api/", "v1/")):
            raise HTTPException(404)
        f = (WEB_DIR / path).resolve()
        if path and f.is_file() and str(f).startswith(str(WEB_DIR.resolve())):
            headers = {"Cache-Control": "public, max-age=31536000, immutable"} if "/assets/" in f"/{path}" else {"Cache-Control": "no-cache"}
            return FileResponse(f, headers=headers)
        index = WEB_DIR / "index.html"
        if index.is_file():
            return FileResponse(index, headers={"Cache-Control": "no-cache"})
        return PlainTextResponse("MPN Studio API is running. The web UI has not been built (web/dist is missing).", status_code=200)

    return app


def main():
    import uvicorn
    host = os.environ.get("MPN_HOST", "127.0.0.1")
    port = int(os.environ.get("MPN_PORT", "8840"))
    uvicorn.run(create_app(), host=host, port=port, log_level="info", access_log=False, server_header=False)


if __name__ == "__main__":
    main()
