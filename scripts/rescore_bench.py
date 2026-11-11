"""Re-score stored benchmark replies with the current checkers (no model run).
Used when a checker is corrected, e.g. to accept another tool-call syntax. Coding tasks are re-executed."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpnstudio.evals import core_suite

tasks = {t["id"]: t for t in core_suite.TASKS}
for path in sys.argv[1:]:
    d = json.load(open(path))
    changed = []
    for r in d["rows"]:
        ok = bool(core_suite.score(tasks[r["id"]], r["reply"]))
        if ok != r["ok"]:
            changed.append((r["id"], r["ok"], ok))
            r["ok"] = ok
    cats = {}
    for r in d["rows"]:
        c = cats.setdefault(r["cat"], {"n": 0, "ok": 0})
        c["n"] += 1
        c["ok"] += r["ok"]
    d["quality"] = {"by_category": cats, "total_ok": sum(r["ok"] for r in d["rows"]), "total": len(d["rows"])}
    d.setdefault("rescored", []).append({"changed": changed})
    strict = [r for r in d["rows"] if r["cat"] in ("structured", "routing") and r["ok"] and core_suite.fenced(r["reply"])]
    d["quality"]["json_in_code_fence"] = len(strict)
    d["quality"]["total_ok_strict_json"] = d["quality"]["total_ok"] - len(strict)
    json.dump(d, open(path, "w"), indent=1, ensure_ascii=False)
    print(path, "changed:", changed, "total", d["quality"]["total_ok"], "/", d["quality"]["total"])
