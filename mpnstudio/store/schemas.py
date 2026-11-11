"""Cell-node schemas: 3CN, 5CN and custom variants.

A schema is a list of cell roles. Each role has
  key            stable identifier used in stored node data and by the runtime
  label          what the editor shows; free to rename
  runtime_role   which runtime behaviour reads the cell: knowledge | thinking | reasoning | provenance | interface | none
  fields         typed fields with validation rules

`runtime_role` is what the runtime dispatches on. Relabelling a cell changes `label` only and
never its semantics. Changing `key`, `runtime_role` or a field's type is a structural change:
it needs a new schema version and a migration, and the compatibility check says so.
"""
from __future__ import annotations

import json
import re

from .db import Store, now

RUNTIME_ROLES = ("knowledge", "thinking", "reasoning", "provenance", "interface", "none")
FIELD_TYPES = ("text", "number", "boolean", "list", "object", "statements", "facts", "constraints", "checks", "referrals", "params")

S3CN = {
    "id": "3cn", "version": 1, "name": "Three Cell Node",
    "description": "The initial node schema: Knowledge, Thinking and Reasoning cells.",
    "roles": [
        {"key": "knowledge", "label": "Knowledge", "runtime_role": "knowledge",
         "description": "Provenance-backed content: sourced statements and structured facts. Read as text (retrieval) and through learned memory slots.",
         "fields": [{"key": "statements", "type": "statements", "required": True, "max_items": 12,
                     "description": "Sentences, each with a source."},
                    {"key": "facts", "type": "facts", "required": False, "description": "Property/value claims, each with a source."}]},
        {"key": "thinking", "label": "Thinking", "runtime_role": "thinking",
         "description": "How a query reaches and uses this node: entity class, cue terms, the answers it can give; for a procedure, the implementation and its parameters. The learned routing key belongs to this cell.",
         "fields": [{"key": "entity_class", "type": "list", "required": False}, {"key": "description", "type": "text", "required": False},
                    {"key": "cues", "type": "list", "required": False}, {"key": "answers", "type": "list", "required": False},
                    {"key": "impl", "type": "text", "required": False, "description": "Registered procedure key. Only built-in keys run."},
                    {"key": "params", "type": "params", "required": False}, {"key": "formula", "type": "text", "required": False},
                    {"key": "unit", "type": "text", "required": False}]},
        {"key": "reasoning", "label": "Reasoning", "runtime_role": "reasoning",
         "description": "Limits and follow-ups: constraints on answers, checks the runtime applies, referrals to related nodes. The learned relevance gate belongs to this cell.",
         "fields": [{"key": "constraints", "type": "constraints", "required": False}, {"key": "checks", "type": "checks", "required": False},
                    {"key": "referrals", "type": "referrals", "required": False}]},
    ],
}

S5CN = {
    "id": "5cn", "version": 1, "name": "Five Cell Node",
    "description": "3CN plus two explicitly defined cells: Provenance (review trail) and Interface (how the node presents and exchanges data).",
    "extends": {"id": "3cn", "version": 1},
    "roles": S3CN["roles"] + [
        {"key": "provenance", "label": "Provenance", "runtime_role": "provenance",
         "description": "Review trail for the node: who checked it, when, against what, and with which outcome. The runtime reports it with citations; it does not change routing.",
         "fields": [{"key": "reviews", "type": "list", "required": False, "description": "Entries {by, on, outcome, note}."},
                    {"key": "confidence", "type": "number", "required": False, "min": 0, "max": 1}]},
        {"key": "interface", "label": "Interface", "runtime_role": "interface",
         "description": "Presentation and exchange: preferred answer format, units and locale. Applied when the runtime formats an answer that cites this node.",
         "fields": [{"key": "answer_format", "type": "text", "required": False}, {"key": "units", "type": "text", "required": False},
                    {"key": "locale", "type": "text", "required": False}]},
    ],
}


def validate_schema(spec: dict) -> list[str]:
    """Structural validation of a schema definition. Returns problems (empty = valid)."""
    errs = []
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,31}", str(spec.get("id", ""))):
        errs.append("id must be 2-32 characters: lower-case letters, digits, '-' or '_'")
    if not isinstance(spec.get("version"), int) or spec.get("version", 0) < 1:
        errs.append("version must be a positive integer")
    if not str(spec.get("name", "")).strip():
        errs.append("name is required")
    roles = spec.get("roles")
    if not isinstance(roles, list) or not roles:
        return errs + ["a schema needs at least one cell role"]
    seen = set()
    for i, r in enumerate(roles):
        where = f"cell {i + 1}"
        key = r.get("key", "")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", str(key)):
            errs.append(f"{where}: key must be lower-case letters, digits or '_' and start with a letter")
        if key in seen:
            errs.append(f"{where}: duplicate key {key!r}")
        seen.add(key)
        if not str(r.get("label", "")).strip():
            errs.append(f"{where}: label is required")
        if r.get("runtime_role") not in RUNTIME_ROLES:
            errs.append(f"{where}: runtime_role must be one of {', '.join(RUNTIME_ROLES)}")
        fkeys = set()
        for f in r.get("fields", []):
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", str(f.get("key", ""))):
                errs.append(f"{where}: field key {f.get('key')!r} is invalid")
            if f.get("key") in fkeys:
                errs.append(f"{where}: duplicate field {f.get('key')!r}")
            fkeys.add(f.get("key"))
            if f.get("type") not in FIELD_TYPES:
                errs.append(f"{where}: field {f.get('key')!r} has unknown type {f.get('type')!r}")
    for needed in ("knowledge",):
        if not any(r.get("runtime_role") == needed for r in roles):
            errs.append(f"a schema needs one cell with runtime_role '{needed}' so its nodes can hold sourced content")
    for role in ("knowledge", "thinking", "reasoning"):
        if sum(1 for r in roles if r.get("runtime_role") == role) > 1:
            errs.append(f"only one cell may have runtime_role '{role}'")
    return errs


def diff_schemas(old: dict, new: dict) -> dict:
    """Compatibility of `new` with `old`: what changed and whether existing nodes need migrating."""
    o = {r["key"]: r for r in old["roles"]}
    n = {r["key"]: r for r in new["roles"]}
    changes, breaking = [], False
    for k in o.keys() - n.keys():
        changes.append({"kind": "cell_removed", "cell": k, "breaking": True,
                        "effect": f"Content of cell '{k}' would be dropped from migrated nodes. Its runtime behaviour ({o[k]['runtime_role']}) stops."})
        breaking = True
    for k in n.keys() - o.keys():
        changes.append({"kind": "cell_added", "cell": k, "breaking": False,
                        "effect": f"New cell '{k}' starts empty on existing nodes. Runtime role: {n[k]['runtime_role']}."})
    for k in o.keys() & n.keys():
        a, b = o[k], n[k]
        if a["label"] != b["label"]:
            changes.append({"kind": "relabelled", "cell": k, "breaking": False,
                            "effect": f"Shown as '{b['label']}' instead of '{a['label']}'. Runtime semantics unchanged ({b['runtime_role']})."})
        if a["runtime_role"] != b["runtime_role"]:
            changes.append({"kind": "runtime_role_changed", "cell": k, "breaking": True,
                            "effect": f"Cell '{k}' would be read as {b['runtime_role']} instead of {a['runtime_role']}. This changes execution, not just the label."})
            breaking = True
        af = {f["key"]: f for f in a.get("fields", [])}
        bf = {f["key"]: f for f in b.get("fields", [])}
        for fk in af.keys() - bf.keys():
            changes.append({"kind": "field_removed", "cell": k, "field": fk, "breaking": True, "effect": f"Field '{fk}' would be dropped from cell '{k}'."})
            breaking = True
        for fk in bf.keys() - af.keys():
            req = bool(bf[fk].get("required"))
            changes.append({"kind": "field_added", "cell": k, "field": fk, "breaking": req,
                            "effect": f"Field '{fk}' added to cell '{k}'" + (" and required: existing nodes fail validation until it is filled." if req else ".")})
            breaking = breaking or req
        for fk in af.keys() & bf.keys():
            if af[fk]["type"] != bf[fk]["type"]:
                changes.append({"kind": "field_type_changed", "cell": k, "field": fk, "breaking": True,
                                "effect": f"Field '{fk}' changes type {af[fk]['type']} -> {bf[fk]['type']}; stored values need conversion."})
                breaking = True
    learned = any(c["kind"] in ("cell_removed", "runtime_role_changed") and c["cell"] in ("knowledge", "thinking", "reasoning") for c in changes)
    return {"changes": changes, "breaking": breaking, "migration_required": breaking,
            "retrain_required": learned,
            "summary": "No structural change." if not changes else
            ("Structural change: needs a new schema version and a migration." if breaking else "Compatible: applies without migration.")}


_TYPE_CHECK = {
    "text": lambda v: isinstance(v, str), "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool), "list": lambda v: isinstance(v, list), "object": lambda v: isinstance(v, dict),
    "params": lambda v: isinstance(v, dict),
    "statements": lambda v: isinstance(v, list) and all(isinstance(x, dict) and isinstance(x.get("text"), str) and x["text"].strip() for x in v),
    "facts": lambda v: isinstance(v, list) and all(isinstance(x, dict) and "property" in x and "value" in x for x in v),
    "constraints": lambda v: isinstance(v, list) and all(isinstance(x, dict) and isinstance(x.get("text"), str) for x in v),
    "checks": lambda v: isinstance(v, list) and all(isinstance(x, dict) and "kind" in x for x in v),
    "referrals": lambda v: isinstance(v, list) and all(isinstance(x, dict) and "node" in x for x in v),
}


def validate_cells(spec: dict, cells: dict) -> list[str]:
    """Validate a node's cells against a schema. Returns problems (empty = valid)."""
    errs = []
    roles = {r["key"]: r for r in spec["roles"]}
    for k in cells:
        if k not in roles:
            errs.append(f"cell '{k}' is not part of schema {spec['id']} v{spec['version']}")
    for k, r in roles.items():
        cell = cells.get(k, {})
        if not isinstance(cell, dict):
            errs.append(f"cell '{k}' must be an object")
            continue
        fields = {f["key"]: f for f in r.get("fields", [])}
        for fk, f in fields.items():
            if fk not in cell or cell[fk] in (None, "", []):
                if f.get("required"):
                    errs.append(f"{r['label']}: '{fk}' is required")
                continue
            v = cell[fk]
            if not _TYPE_CHECK[f["type"]](v):
                errs.append(f"{r['label']}: '{fk}' is not a valid {f['type']}")
                continue
            if "max_items" in f and isinstance(v, list) and len(v) > f["max_items"]:
                errs.append(f"{r['label']}: '{fk}' has more than {f['max_items']} items")
            if f["type"] == "number":
                if "min" in f and v < f["min"] or "max" in f and v > f["max"]:
                    errs.append(f"{r['label']}: '{fk}' is out of range")
        for fk in cell:
            if fk not in fields and fk != "learned":
                errs.append(f"{r['label']}: unknown field '{fk}'")
    return errs


def migrate_cells(old: dict, new: dict, cells: dict) -> dict:
    """Carry a node's cells from schema `old` to `new`: keep cells and fields that still exist, start new ones empty."""
    new_roles = {r["key"]: r for r in new["roles"]}
    out = {}
    for k, r in new_roles.items():
        src = cells.get(k, {}) if isinstance(cells.get(k), dict) else {}
        keep = {f["key"] for f in r.get("fields", [])} | {"learned"}
        out[k] = {fk: v for fk, v in src.items() if fk in keep}
    return out


def seed_schemas(store: Store) -> None:
    c = store.conn()
    for spec in (S3CN, S5CN):
        c.execute("INSERT OR IGNORE INTO cell_schemas(id, version, name, spec, status, created) VALUES(?,?,?,?,?,?)",
                  (spec["id"], spec["version"], spec["name"], json.dumps(spec), "active", now()))


def get_schema(store: Store, sid: str, version: int | None = None) -> dict | None:
    if version is None:
        r = store.q1("SELECT spec FROM cell_schemas WHERE id=? ORDER BY version DESC LIMIT 1", (sid,))
    else:
        r = store.q1("SELECT spec FROM cell_schemas WHERE id=? AND version=?", (sid, version))
    return json.loads(r[0]) if r else None


def list_schemas(store: Store) -> list[dict]:
    out = []
    for r in store.q("SELECT id, version, name, spec, status, created FROM cell_schemas ORDER BY id, version"):
        spec = json.loads(r["spec"])
        n = store.q1("SELECT COUNT(*) FROM node_v WHERE schema_id=? AND schema_version=? AND rev_to IS NULL AND status<>'tombstone'", (r["id"], r["version"]))[0]
        out.append({"id": r["id"], "version": r["version"], "name": r["name"], "status": r["status"], "created": r["created"],
                    "cells": len(spec["roles"]), "nodes": n, "spec": spec})
    return out


def save_schema(store: Store, spec: dict) -> dict:
    """Store a schema version. A version that is already used by nodes can only change labels and descriptions."""
    errs = validate_schema(spec)
    if errs:
        raise ValueError("; ".join(errs))
    existing = get_schema(store, spec["id"], spec["version"])
    if existing:
        d = diff_schemas(existing, spec)
        used = store.q1("SELECT COUNT(*) FROM node_v WHERE schema_id=? AND schema_version=? AND rev_to IS NULL", (spec["id"], spec["version"]))[0]
        if d["breaking"] and used:
            raise ValueError(f"{used} nodes use {spec['id']} v{spec['version']}; this is a structural change, so save it as version {spec['version'] + 1} and migrate")
        store.conn().execute("UPDATE cell_schemas SET name=?, spec=? WHERE id=? AND version=?", (spec["name"], json.dumps(spec), spec["id"], spec["version"]))
        return {"saved": True, "diff": d}
    store.conn().execute("INSERT INTO cell_schemas(id, version, name, spec, status, created) VALUES(?,?,?,?,?,?)",
                         (spec["id"], spec["version"], spec["name"], json.dumps(spec), "active", now()))
    prev = get_schema(store, spec["id"], spec["version"] - 1) if spec["version"] > 1 else None
    return {"saved": True, "diff": diff_schemas(prev, spec) if prev else None}
