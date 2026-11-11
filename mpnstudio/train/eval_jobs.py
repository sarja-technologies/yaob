"""Jobs that use the running runtime over loopback instead of loading a model themselves: evaluation and re-indexing.

Because they go through the same runtime that serves the UI, what they measure is the deployed system.
They hold no GPU memory of their own (`gpu: false`).
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from ..evals import heldout
from .jobs import JobContext, data_root, job_kind

RUNTIME = os.environ.get("MPN_RUNTIME_URL", "http://127.0.0.1:8841")


def _post(path: str, body: dict | None = None, timeout: int = 600) -> dict:
    req = urllib.request.Request(RUNTIME + path, data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


@job_kind("evaluate")
def evaluate(ctx: JobContext) -> dict:
    p = ctx.params
    suite_path = Path(p.get("suite") or (data_root() / "eval" / "heldout_v1.json"))
    if not suite_path.exists():
        suite = heldout.build_suite(ctx.store, int(p.get("seed", 20261111)))
        suite_path.parent.mkdir(parents=True, exist_ok=True)
        suite_path.write_text(json.dumps(suite, indent=1, ensure_ascii=False))
        ctx.log(f"built suite {suite_path}: {suite['sizes']}")
    suite = json.loads(suite_path.read_text())
    arms = [a for a in (p.get("arms") or ",".join(heldout.ARMS)).split(",") if a in heldout.ARMS]
    from ..api.app import get_budget
    budget = dict(get_budget(ctx.store), output_tokens=int(p.get("output_tokens", 128)))
    out = ctx.dir / "eval"
    prev = ctx.resume_dir()
    if prev and (prev / "eval" / "rows.jsonl").exists():          # resume: finished questions are not asked again
        out.mkdir(parents=True, exist_ok=True)
        (out / "rows.jsonl").write_text((prev / "eval" / "rows.jsonl").read_text())
        ctx.log(f"resuming evaluation of job {prev.name}")
    total = len(arms) * (int(p.get("limit", 0)) or len(suite["tasks"]))

    def log(msg):
        ctx.log(msg)
        done = sum(1 for _ in (out / "rows.jsonl").open()) if (out / "rows.jsonl").exists() else 0
        ctx.progress(stage="evaluating", step=done, total=total, force=True)
        if ctx.should_stop():
            raise KeyboardInterrupt
    try:
        rep = heldout.run_suite(ctx.store, suite, out, RUNTIME, arms, budget, int(p.get("limit", 0)), log)
    except KeyboardInterrupt:
        return {"cancelled": True, "rows": str(out / "rows.jsonl")}
    ctx.log(heldout.render_markdown(rep, suite))
    summary = {a: rep["arms"][a]["overall"] for a in rep["arms"]}
    ck = (rep.get("runtime") or {}).get("checkpoint") or {}
    if ck.get("id"):
        row = ctx.store.q1("SELECT metrics FROM checkpoints WHERE id=?", (ck["id"],))
        if row:
            m = json.loads(row[0])
            m["held_out_evaluation"] = {"job": ctx.id, "summary": summary, "report": str(out / "report.json")}
            ctx.store.conn().execute("UPDATE checkpoints SET metrics=? WHERE id=?", (json.dumps(m), ck["id"]))
    return {"report": str(out / "report.json"), "markdown": str(out / "report.md"), "summary": summary}


@job_kind("reindex")
def reindex(ctx: JobContext) -> dict:
    total = 0
    while True:
        r = _post("/reindex")
        total += r.get("indexed", 0)
        ctx.progress(stage="indexing", indexed=total, pending=r.get("pending", 0), force=True)
        if not r.get("pending") or not r.get("indexed") or ctx.should_stop():
            return {"indexed": total, "pending": r.get("pending", 0)}
