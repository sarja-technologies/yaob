"""`mpn`: command-line client for MPN Studio.

It talks to the same HTTP API as the web UI, so both act on the same revisions, jobs and rules.
Set MPN_URL (default http://127.0.0.1:8840). Uses only the standard library.

Examples
  mpn status
  mpn search "battle of waterloo" --mode lexical
  mpn node show 4211
  mpn node create --title "Congress of Vienna" --mpn history --mpn geography-geopolitics --statement "The Congress of Vienna met from 1814 to 1815." --source-title "Editor's note"
  mpn member remove 10201 history
  mpn mpn impact biology uninstall
  mpn mpn disable biology
  mpn run "Who commanded the allied army at Waterloo?"
  mpn run "Convert 12 miles to kilometres" --mode yaob --evidence
  mpn job start train_bank exposures=24 batch=16
  mpn job log 261111-170233-ab12cd --follow
  mpn budget preset conservative
  mpn undo
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

URL = os.environ.get("MPN_URL", "http://127.0.0.1:8840").rstrip("/")


class ApiError(Exception):
    pass


def call(method: str, path: str, body: dict | None = None, query: dict | None = None, stream: bool = False):
    url = URL + path
    if query:
        url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v not in (None, "")})
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"} if data is not None else {})
    try:
        r = urllib.request.urlopen(req, timeout=900)
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "replace")
        try:
            j = json.loads(text)
            msg = j.get("error") or text
            if j.get("detail", {}).get("problems"):
                msg += "\n  - " + "\n  - ".join(j["detail"]["problems"])
        except Exception:
            msg = text
        raise ApiError(f"{e.code}: {msg}")
    except urllib.error.URLError as e:
        raise ApiError(f"cannot reach MPN Studio at {URL}: {e.reason}")
    if stream:
        return r
    return json.loads(r.read().decode() or "{}")


def out(x, as_json: bool = False):
    if as_json or not isinstance(x, str):
        print(json.dumps(x, indent=1, ensure_ascii=False))
    else:
        print(x)


def resolve_mpn(ref: str) -> str:
    if ref.isdigit():
        return ref
    for m in call("GET", "/api/mpns")["mpns"]:
        if m["slug"] == ref or m["name"].lower() == ref.lower():
            return m["id"]
    raise ApiError(f"no MPN called {ref!r}")


def kv(pairs: list[str]) -> dict:
    d = {}
    for p in pairs:
        if "=" not in p:
            raise ApiError(f"expected key=value, got {p!r}")
        k, v = p.split("=", 1)
        try:
            d[k] = json.loads(v)
        except Exception:
            d[k] = v
    return d


def sse_events(resp):
    ev, data = None, []
    for raw in resp:
        line = raw.decode("utf-8", "replace").rstrip("\n")
        if line.startswith("event:"):
            ev = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
        elif line == "" and ev:
            yield ev, json.loads("\n".join(data) or "{}")
            ev, data = None, []


def cmd_status(a):
    s = call("GET", "/api/status")
    if a.json:
        return out(s, True)
    c, d, rt = s["counts"], s["dirty"], s["runtime"]
    print(f"MPN Studio {s['version']} at {URL}   revision {c['rev']}")
    print(f"canonical live nodes   {c['canonical_live_nodes']:,}  ({', '.join(f'{k} {v:,}' for k, v in c['live_by_type'].items())})")
    print(f"memberships {c['memberships']:,} ({c['multi_membership_nodes']:,} nodes in more than one MPN)   edges {c['edges']:,}   MPNs {c['mpns']}   tombstones {c['tombstones']}")
    print(f"learned state          trained {d['trained_current']:,}   stale {d['trained_stale']:,}   untrained {d['untrained']:,}   to index {d['to_index']:,}")
    if rt.get("available"):
        core = rt.get("core") or {}
        print(f"runtime                core {core.get('name', 'not loaded')} {str(core.get('revision', ''))[:8]}   checkpoint {(rt.get('checkpoint') or {}).get('id', 'none')}   "
              f"process {rt['memory']['process_rss_mib']:.0f} MiB, torch {rt['memory']['torch_allocated_mib']:.0f} MiB")
    else:
        print(f"runtime                unavailable: {rt.get('reason')}")
    print(f"memory                 {s['memory']['available_gib']} of {s['memory']['total_gib']} GiB available   jobs active {s['jobs_active']}")


def cmd_search(a):
    r = call("GET", "/api/search", query={"q": a.query, "mode": a.mode, "limit": a.limit, "type": a.type, "status": a.status,
                                          "validation": a.validation, "mpn_id": resolve_mpn(a.mpn) if a.mpn else ""})
    if a.json:
        return out(r, True)
    if r.get("notice"):
        print(r["notice"])
    for x in r["results"]:
        print(f"{x['id']:>8}  {x['title'][:48]:48}  {x.get('mpn', '')[:28]:28}  {x.get('match', '')}")
    print(f"{len(r['results'])} results in {r.get('ms', 0)} ms")


def cmd_node(a):
    if a.action == "show":
        n = call("GET", f"/api/nodes/{a.id}", query={"rev": a.rev})
        if a.json:
            return out(n, True)
        print(f"{n['title']}  (node {n['id']}, {n['type']}, {n['status']}, {n['validation']}, schema {n['schema']['id']} v{n['schema']['version']}, revision {n['rev']})")
        print("memberships: " + ", ".join(f"{m['mpn_name']} ({m['role']})" for m in n["memberships"]))
        for role in n["schema"]["roles"]:
            cell = n["cells"].get(role["key"], {})
            print(f"\n[{role['label']}]  runtime role: {role['runtime_role']}")
            for k, v in cell.items():
                if isinstance(v, list):
                    for item in v[:12]:
                        print("  - " + (item.get("text") or f"{item.get('property', item.get('relation', item.get('kind', '')))}: {item.get('value', item.get('title', ''))}" if isinstance(item, dict) else str(item)))
                elif v not in ("", None, {}):
                    print(f"  {k}: {v}")
        print("\nsources:")
        for s in n["sources"]:
            print(f"  {s['title']}  [{s['license']}]  {s['url']}")
        print(f"\nstate: {n['state']['explain']}  (index {n['state']['index']}, learned {n['state']['learned']})   edges {n['edge_count']}")
    elif a.action == "create":
        body = {"title": a.title, "summary": a.summary or "", "type": a.type or "knowledge",
                "memberships": [{"mpn_id": resolve_mpn(m)} for m in a.mpn or []],
                "cells": {"knowledge": {"statements": [{"text": s} for s in a.statement or []]}, "thinking": {}, "reasoning": {}},
                "sources": [{"title": a.source_title, "url": a.source_url or "", "license": a.source_license or ""}] if a.source_title or a.source_url else []}
        r = call("POST", "/api/nodes", body)
        out(r if a.json else f"created node {r['id']} at revision {r['rev']}; live now: {', '.join(r['effects']['live_now'])}; after indexing: {', '.join(r['effects']['after_indexing'])}", a.json)
    elif a.action == "edit":
        cur = call("GET", f"/api/nodes/{a.id}")
        body = {}
        if a.title:
            body["title"] = a.title
        if a.summary is not None:
            body["summary"] = a.summary
        if a.statement:
            cells = cur["cells"]
            cells.setdefault("knowledge", {})["statements"] = [{"text": s} for s in a.statement]
            body["cells"] = cells
        if a.set_status:
            body["status"] = a.set_status
        r = call("PATCH", f"/api/nodes/{a.id}", body)
        out(r if a.json else f"revision {r['rev']}; changed: {r['changed']}" + (f"; after training: {', '.join(r['effects']['after_training'])}" if r.get("effects", {}).get("after_training") else ""), a.json)
    elif a.action == "delete":
        p = call("GET", f"/api/nodes/{a.id}/delete-preview")
        print(f"deleting “{p['node']['title']}”: {len(p['memberships'])} memberships, {p['edges']} edges. {p['note']}")
        out(call("DELETE", f"/api/nodes/{a.id}"), True)
    elif a.action == "restore":
        r = call("POST", f"/api/nodes/{a.id}/restore")
        print(f"revision {r['rev']}; restored: {r['changed']}")
    elif a.action == "history":
        out(call("GET", f"/api/nodes/{a.id}/history"), True)


def cmd_member(a):
    mid = resolve_mpn(a.mpn)
    r = call("PUT", f"/api/nodes/{a.node}/memberships/{mid}", {"role": a.role} if a.role else {}) if a.action == "add" else call("DELETE", f"/api/nodes/{a.node}/memberships/{mid}")
    print(f"revision {r['rev']}: " + ", ".join(f"{m['mpn_name']} ({m['role']})" for m in r["memberships"]))


def cmd_edge(a):
    if a.action == "add":
        r = call("POST", "/api/edges", {"src": a.src, "dst": a.dst, "type": a.type})
        print(f"edge {r['id']} at revision {r['rev']}")
    else:
        print(f"revision {call('DELETE', f'/api/edges/{a.src}')['rev']}")


def cmd_mpn(a):
    if a.action == "list":
        r = call("GET", "/api/mpns")
        if a.json:
            return out(r, True)
        for m in r["mpns"]:
            st = ("enabled" if m["enabled"] else "disabled") if m["installed"] else "not installed"
            print(f"{m['id']:>4}  {m['slug']:26}  {m['node_count']:6,} nodes  {m['shared_count']:5,} shared  {st:13}  {m['training']:14}  {'resident' if m['resident'] else ''}")
    elif a.action == "create":
        r = call("POST", "/api/mpns", {"name": a.ref, "description": a.description or ""})
        print(f"created MPN {r['mpn']['id']} ({r['mpn']['slug']}) at revision {r['rev']}")
    elif a.action == "impact":
        r = call("GET", f"/api/mpns/{resolve_mpn(a.ref)}/impact", query={"action": a.what})
        out(r if a.json else r["explanation"], a.json)
    else:
        mid = resolve_mpn(a.ref)
        print(call("GET", f"/api/mpns/{mid}/impact", query={"action": a.action})["explanation"])
        r = call("POST", f"/api/mpns/{mid}/{a.action}")
        print(f"done at revision {r['rev']}")


def cmd_bundle(a):
    if a.action == "export":
        out(call("POST", "/api/bundles", {"name": a.ref} if a.ref else {}), True)
    elif a.action == "packages":
        for p in call("GET", "/api/packages")["packages"]:
            print(f"{p['name']:44}  {p['title'] or '':30}  {p['counts']['nodes']:6,} nodes  {p['class']}  {'installed' if p['installed'] else ''}")
    elif a.action == "export-mpn":
        out(call("POST", f"/api/mpns/{resolve_mpn(a.ref)}/export"), True)
    elif a.action == "install":
        out(call("POST", f"/api/packages/{a.ref}/install"), True)


def cmd_schema(a):
    if a.action == "list":
        for s in call("GET", "/api/schemas")["schemas"]:
            print(f"{s['id']} v{s['version']}  {s['name']:20}  {s['cells']} cells  {s['nodes']:,} nodes")
    elif a.action == "validate":
        r = call("POST", "/api/schemas/validate", json.load(open(a.file)))
        for p in r["problems"]:
            print("problem: " + p)
        for c in (r.get("diff") or {}).get("changes", []):
            print(("structural: " if c["breaking"] else "compatible: ") + c["effect"])
        if r.get("diff"):
            print(r["diff"]["summary"])
    elif a.action == "save":
        out(call("PUT", "/api/schemas", json.load(open(a.file))), True)
    elif a.action == "migrate":
        f, t = a.file.split(":"), a.to.split(":")
        q = {"from_id": f[0], "from_version": int(f[1]), "to_id": t[0], "to_version": int(t[1])}
        prev = call("GET", "/api/schemas/migration", query=q)
        print(f"{prev['nodes']:,} nodes; {prev['diff']['summary']} {prev['note']}")
        if a.apply:
            r = call("POST", "/api/schemas/migrate", q)
            print(f"migrated {r['migrated']:,} nodes at revision {r['rev']}")


def cmd_job(a):
    if a.action == "list":
        r = call("GET", "/api/jobs")
        for j in r["jobs"]:
            pr = j["progress"]
            where = f"{pr.get('step', '')}/{pr.get('total', '')}" if "step" in pr else (pr.get("waiting") or pr.get("stage") or "")
            print(f"{j['id']}  {j['kind']:14}  {j['state']:11}  {where}")
    elif a.action == "start":
        print(call("POST", "/api/jobs", {"kind": a.ref, "params": kv(a.params)})["id"])
    elif a.action == "show":
        out(call("GET", f"/api/jobs/{a.ref}"), True)
    elif a.action == "cancel":
        print(call("POST", f"/api/jobs/{a.ref}/cancel")["state"])
    elif a.action == "resume":
        print(call("POST", f"/api/jobs/{a.ref}/resume")["id"])
    elif a.action == "log":
        off = -8000
        while True:
            r = call("GET", f"/api/jobs/{a.ref}/log", query={"offset": off})
            sys.stdout.write(r["text"])
            sys.stdout.flush()
            off = r["next"]
            if not a.follow or call("GET", f"/api/jobs/{a.ref}")["state"] not in ("queued", "running", "cancelling"):
                if not r["text"]:
                    break
                if not a.follow:
                    break
            time.sleep(1.5)


def cmd_checkpoint(a):
    if a.action == "list":
        for c in call("GET", "/api/checkpoints")["checkpoints"]:
            m = c["metrics"].get("trained", {})
            print(f"{c['id']}  {c['status']:9}  step {c['step']:6}  rev {c['rev']}  held-out memory {m.get('memory_gold', '')}  routed {m.get('memory_routed', '')}  {c['path']}")
    else:
        out(call("POST", f"/api/checkpoints/{a.ref}/activate"), True)


def cmd_run(a):
    body = {"messages": [{"role": "user", "content": a.question}], "mode": a.mode}
    if a.rev:
        body["rev"] = a.rev
    resp = call("POST", "/api/run", body, stream=True)
    route = None
    for ev, d in sse_events(resp):
        if a.json:
            print(json.dumps({"event": ev, "data": d}, ensure_ascii=False))
            continue
        if ev == "token":
            sys.stdout.write(d["text"])
            sys.stdout.flush()
        elif ev == "route":
            route = d
        elif ev == "error":
            print(f"\nerror: {d['error']}")
            sys.exit(1)
        elif ev == "validation" and a.evidence:
            val = d
        elif ev == "done":
            print()
            t, u, m = d["timing"], d["usage"], d["memory"]
            if a.evidence and route is not None:
                print("\nconsulted")
                print("  MPNs: " + (", ".join(f"{x['name']} ({x['score']})" for x in route["mpns"] if x["active"]) or "none"))
                if route.get("skipped"):
                    print("  skipped: " + ", ".join(f"{x['name']} ({x['reason']})" for x in route["skipped"]))
                for s in route.get("shards", []):
                    print(f"  shard {s.get('name', s['mpn_id'])}: {s['state']}, {s['bytes'] / 2**20:.1f} MiB, {s['ms']} ms")
                for n in route.get("nodes", []):
                    print(f"  node {n['id']} {n['title'][:44]:44} {'+'.join(n['read_as']) or '-':12} gate {n['gate']}  via {n['via']}  [{n['mpn']}]")
                for p in route.get("procedures", []):
                    print(f"  procedure {p['title']}: {json.dumps(p['args'])} -> {json.dumps(p.get('result')) if p['ok'] else p.get('error')}")
                for r in route.get("budget", {}).get("reductions", []):
                    print(f"  budget: {r}")
                for c in val.get("citations", []):
                    if c["source"]:
                        print(f"  [{c['n']}] {c['title']}: {c['source']['url']} ({c['source']['license']})")
                for c in val.get("checks", []):
                    print(f"  check {c['kind']}: {'ok' if c['ok'] else 'FAILED'}: {c['detail']}")
            print(f"\n{u.get('completion_tokens', 0)} tokens, first token {t.get('ttft_ms')} ms, total {t.get('total_ms')} ms, route {t.get('route_ms')} ms"
                  f" | prompt {u.get('prompt_tokens')} tokens ({u.get('memory_tokens', 0)} memory, {u.get('text_evidence_tokens', 0)} text evidence)"
                  f" | torch peak {m.get('torch_peak_mib')} MiB, estimate {m.get('estimate_mib')} MiB")


def cmd_budget(a):
    if a.action == "show":
        return out(call("GET", "/api/budget"), True)
    if a.action == "preset":
        return out(call("PUT", "/api/budget", {"preset": a.values[0], "apply_preset": True})["budget"], True)
    out(call("PUT", "/api/budget", kv(a.values))["budget"], True)


def cmd_runtime(a):
    out(call("GET", "/api/runtime") if a.action == "status" else call("POST", f"/api/runtime/{a.action}", kv(a.params) if a.params else {}), True)


def cmd_simple(a):
    if a.cmd == "undo":
        r = call("POST", "/api/undo")
        print(f"undone; head is revision {r['head']}" if r["undone"] else "nothing to undo")
    elif a.cmd == "redo":
        r = call("POST", "/api/redo")
        print(f"redone; head is revision {r['head']}" if r["redone"] else "nothing to redo")
    elif a.cmd == "revisions":
        for r in call("GET", "/api/revisions", query={"limit": a.limit})["revisions"]:
            print(f"{r['rev']:6}  {r['created']}  {r['kind']:7}  {r['actor']:14}  {r['message']}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mpn", description="MPN Studio command line (uses the HTTP API at MPN_URL).")
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("--mode", default="hybrid", choices=["hybrid", "lexical", "semantic"]); s.add_argument("--limit", type=int, default=15)
    s.add_argument("--type", default=""); s.add_argument("--status", default=""); s.add_argument("--validation", default=""); s.add_argument("--mpn", default=""); s.set_defaults(fn=cmd_search)
    n = sub.add_parser("node"); n.add_argument("action", choices=["show", "create", "edit", "delete", "restore", "history"]); n.add_argument("id", nargs="?")
    n.add_argument("--rev", type=int); n.add_argument("--title"); n.add_argument("--summary"); n.add_argument("--type"); n.add_argument("--mpn", action="append")
    n.add_argument("--statement", action="append"); n.add_argument("--source-title"); n.add_argument("--source-url"); n.add_argument("--source-license")
    n.add_argument("--set-status", choices=["live", "disabled"]); n.set_defaults(fn=cmd_node)
    m = sub.add_parser("member"); m.add_argument("action", choices=["add", "remove"]); m.add_argument("node"); m.add_argument("mpn"); m.add_argument("--role", choices=["primary", "shared"]); m.set_defaults(fn=cmd_member)
    e = sub.add_parser("edge"); e.add_argument("action", choices=["add", "remove"]); e.add_argument("src"); e.add_argument("dst", nargs="?"); e.add_argument("--type", default="related_to"); e.set_defaults(fn=cmd_edge)
    p = sub.add_parser("mpn"); p.add_argument("action", choices=["list", "create", "impact", "enable", "disable", "install", "uninstall"]); p.add_argument("ref", nargs="?"); p.add_argument("what", nargs="?", default="uninstall")
    p.add_argument("--description"); p.set_defaults(fn=cmd_mpn)
    bd = sub.add_parser("bundle", help="export a model bundle; list, export and install MPN packages"); bd.add_argument("action", choices=["export", "packages", "export-mpn", "install"]); bd.add_argument("ref", nargs="?"); bd.set_defaults(fn=cmd_bundle)
    sc = sub.add_parser("schema"); sc.add_argument("action", choices=["list", "validate", "save", "migrate"]); sc.add_argument("file", nargs="?"); sc.add_argument("--to"); sc.add_argument("--apply", action="store_true"); sc.set_defaults(fn=cmd_schema)
    j = sub.add_parser("job"); j.add_argument("action", choices=["list", "start", "show", "log", "cancel", "resume"]); j.add_argument("ref", nargs="?"); j.add_argument("params", nargs="*"); j.add_argument("--follow", action="store_true"); j.set_defaults(fn=cmd_job)
    c = sub.add_parser("checkpoint"); c.add_argument("action", choices=["list", "activate"]); c.add_argument("ref", nargs="?"); c.set_defaults(fn=cmd_checkpoint)
    r = sub.add_parser("run"); r.add_argument("question"); r.add_argument("--mode", default="yaob", choices=["yaob", "retrieval", "core"]); r.add_argument("--rev", type=int); r.add_argument("--evidence", action="store_true"); r.set_defaults(fn=cmd_run)
    b = sub.add_parser("budget"); b.add_argument("action", choices=["show", "set", "preset"]); b.add_argument("values", nargs="*"); b.set_defaults(fn=cmd_budget)
    rt = sub.add_parser("runtime"); rt.add_argument("action", choices=["status", "load", "unload", "reindex", "evict"]); rt.add_argument("params", nargs="*"); rt.set_defaults(fn=cmd_runtime)
    for name in ("undo", "redo"):
        sub.add_parser(name).set_defaults(fn=cmd_simple)
    rv = sub.add_parser("revisions"); rv.add_argument("--limit", type=int, default=20); rv.set_defaults(fn=cmd_simple)
    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except ApiError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except (BrokenPipeError, KeyboardInterrupt):
        sys.exit(130)


if __name__ == "__main__":
    main()
