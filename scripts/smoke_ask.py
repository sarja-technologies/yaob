"""Ask the runtime canary questions in every arm and report what came back. Used by scripts/smoke_runtime.sh.

The questions are built from the canary nodes' stored facts ("What is the <property> of <name>?"); the expected
value is the stored value. These are the training facts' own phrasing family, so this is a smoke test of the
path (does each arm run, does the adapter carry anything), not the held-out evaluation.

  python scripts/smoke_ask.py <db> <runtime-url> [n]
"""
from __future__ import annotations

import json
import sys
import urllib.request

sys.path.insert(0, ".")
from mpnstudio.store.db import Store  # noqa: E402

ARMS = {"core": ("core", {}), "retrieval": ("retrieval", {}), "yaob": ("yaob", {}),
        "yaob_closed_book": ("yaob", {"text": False}), "abl_no_adapters": ("yaob", {"adapters": False})}


def ask(url: str, q: str, mode: str, options: dict) -> dict:
    body = json.dumps({"messages": [{"role": "user", "content": q}], "mode": mode, "options": options, "budget": {"output_tokens": 48}}).encode()
    req = urllib.request.Request(url + "/run", data=body, headers={"content-type": "application/json"})
    out = {"text": "", "route": None, "error": None, "done": None}
    with urllib.request.urlopen(req, timeout=300) as r:
        ev = None
        for raw in r:
            line = raw.decode().rstrip("\n")
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                d = json.loads(line[6:])
                if ev == "token":
                    out["text"] += d["text"]
                elif ev == "route":
                    out["route"] = d
                elif ev == "error":
                    out["error"] = d["error"]
                elif ev == "done":
                    out["done"] = d
    return out


def main():
    db, url = sys.argv[1], sys.argv[2].rstrip("/")
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    s = Store(db)
    mid = int(s.get_setting("canary")["mpn_id"])
    ids = [r[0] for r in s.q("SELECT node_id FROM member_v WHERE mpn_id=? AND rev_to IS NULL ORDER BY node_id", (mid,))][:n]
    qs = []
    for nid in ids:
        node = s.get_node(nid, full=True)
        f = node["cells"]["knowledge"]["facts"][0]
        qs.append((f"What is the {f['property']} of {node['title']}?", str(f["value"])))
    summary = {}
    for arm, (mode, opts) in ARMS.items():
        ok = 0
        for q, want in qs:
            r = ask(url, q, mode, opts)
            hit = want.lower() in r["text"].lower()
            ok += hit
            if q == qs[0][0]:
                rt = r["route"] or {}
                print(json.dumps({"arm": arm, "q": q, "want": want, "reply": r["text"][:200], "error": r["error"],
                                  "mpns": [m["name"] for m in rt.get("mpns", []) if m.get("active")], "adapters": [a["name"] for a in rt.get("adapters", [])],
                                  "nodes": [(x["title"], x["read_as"]) for x in rt.get("nodes", [])][:3],
                                  "timing": (r["done"] or {}).get("timing"), "memory": (r["done"] or {}).get("memory")}, ensure_ascii=False))
        summary[arm] = f"{ok}/{len(qs)}"
    print(json.dumps({"smoke_summary": summary}))


if __name__ == "__main__":
    main()
