"""Durable jobs: a queue in the model store, a worker, and one transient systemd unit per job.

* A job is a row in `jobs`. Submitting returns at once; nothing depends on the browser or an SSH shell.
* The worker (`python -m mpnstudio.train.jobs worker`) admits queued jobs against the memory budget and
  starts each one as its own transient user unit, `mpn-studio-job-<id>`, so a job survives a restart of
  the worker and of the API.
* A job process writes progress and a heartbeat to its row, a log file, and atomic checkpoints.
* Cancel: the row goes to `cancelling`, the unit gets SIGTERM, the job finishes its current step, saves a
  resumable checkpoint and ends as `cancelled`.
* Interruption: a running job whose unit is gone is marked `interrupted`. `resume` queues a new attempt
  that continues from the last checkpoint (exact resume when optimizer, scheduler and step were saved;
  otherwise it is reported as a restart).

Without systemd (tests, other machines) jobs run as detached child processes with the same contract.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil

from ..store.db import Store, now

ACTIVE = ("queued", "running", "cancelling")
TERMINAL = ("succeeded", "failed", "cancelled", "interrupted")


def data_root() -> Path:
    return Path(os.environ.get("MPN_DATA", Path.home() / "mpn-studio" / "data"))


def db_path() -> Path:
    return Path(os.environ.get("MPN_DB", data_root() / "yaob-n10k.sqlite"))


def job_dir(job_id: str) -> Path:
    d = data_root() / "jobs" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def has_systemd() -> bool:
    return shutil.which("systemd-run") is not None and os.environ.get("MPN_NO_SYSTEMD") != "1" and bool(os.environ.get("XDG_RUNTIME_DIR"))


# ---------------------------------------------------------------------------- queue operations
def submit(store: Store, kind: str, params: dict, resume_of: str | None = None, attempt: int = 1) -> str:
    job_id = time.strftime("%y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    store.conn().execute(
        "INSERT INTO jobs(id, kind, state, params, created, resume_of, attempt) VALUES(?,?,?,?,?,?,?)",
        (job_id, kind, "queued", json.dumps(params), now(), resume_of, attempt))
    return job_id


def get(store: Store, job_id: str) -> dict | None:
    r = store.q1("SELECT * FROM jobs WHERE id=?", (job_id,))
    if r is None:
        return None
    d = dict(r)
    for k in ("params", "progress", "result"):
        d[k] = json.loads(d[k] or "{}")
    d["log_path"] = str(job_dir(job_id) / "log.txt")
    return d


def list_jobs(store: Store, limit: int = 50) -> list[dict]:
    return [get(store, r[0]) for r in store.q("SELECT id FROM jobs ORDER BY created DESC LIMIT ?", (limit,))]


def update(store: Store, job_id: str, **fields) -> None:
    sets, args = [], []
    for k, v in fields.items():
        sets.append(f"{k}=?")
        args.append(json.dumps(v) if k in ("params", "progress", "result") else v)
    store.conn().execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id=?", (*args, job_id))


def cancel(store: Store, job_id: str) -> dict:
    j = get(store, job_id)
    if j is None:
        raise KeyError(job_id)
    if j["state"] == "queued":
        update(store, job_id, state="cancelled", finished=now())
    elif j["state"] == "running":
        update(store, job_id, state="cancelling")
        _signal_job(j, signal.SIGTERM)
    return get(store, job_id)


def resume(store: Store, job_id: str) -> str:
    j = get(store, job_id)
    if j is None:
        raise KeyError(job_id)
    if j["state"] not in ("cancelled", "interrupted", "failed"):
        raise ValueError(f"job is {j['state']}; only a cancelled, interrupted or failed job can be resumed")
    params = dict(j["params"], resume_from=str(job_dir(job_id)))
    return submit(store, j["kind"], params, resume_of=job_id, attempt=j["attempt"] + 1)


def _unit_name(job_id: str) -> str:
    return f"mpn-studio-job-{job_id}"


def _unit_active(unit: str) -> bool:
    r = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True)
    return r.stdout.strip() in ("active", "activating", "deactivating")


def _signal_job(j: dict, sig: int) -> None:
    if j.get("unit") and has_systemd():
        subprocess.run(["systemctl", "--user", "kill", f"--signal={signal.Signals(sig).name}", j["unit"]], capture_output=True)
    elif j.get("pid") and psutil.pid_exists(j["pid"]):
        try:
            os.kill(j["pid"], sig)
        except ProcessLookupError:
            pass


def job_alive(j: dict) -> bool:
    if j.get("unit") and has_systemd():
        return _unit_active(j["unit"])
    return bool(j.get("pid")) and psutil.pid_exists(j["pid"]) and psutil.Process(j["pid"]).status() != psutil.STATUS_ZOMBIE


# ---------------------------------------------------------------------------- admission
def memory_status(store: Store | None = None) -> dict:
    vm = psutil.virtual_memory()
    budget = (store.get_setting("budget") if store else None) or {}
    reserve = float(budget.get("reserve_gib", 16.0))
    limit = float(budget.get("total_gib", 0) or 0)
    avail = vm.available / 2**30
    return {"total_gib": round(vm.total / 2**30, 1), "available_gib": round(avail, 1), "reserve_gib": reserve,
            "usable_gib": round(max(avail - reserve, 0.0), 1), "budget_total_gib": limit,
            "swap_used_gib": round(psutil.swap_memory().used / 2**30, 2)}


# Per-kind defaults: whether the job loads a model on the GPU, and how much memory it needs at peak.
# train_bank / retrain_nodes: pilot 2 measured a torch peak of 50-54 GiB training rank-16 LoRA on Qwen3-4B at batch 16
# with up to 72-token answers (core 7.5 GiB, activations and the PEFT graph the rest); at batch 8 about half the activations.
# 40 GiB is the estimate at the job's default batch of 8; a job that sets batch 16 should set memory_estimate_gib to about 56.
KIND_DEFAULTS = {"train_bank": {"gpu": True, "memory_estimate_gib": 40.0}, "retrain_nodes": {"gpu": True, "memory_estimate_gib": 40.0},
                 "evaluate": {"gpu": False, "memory_estimate_gib": 0.5}, "reindex": {"gpu": False, "memory_estimate_gib": 0.5},
                 "sleep": {"gpu": True, "memory_estimate_gib": 12.0}}


def admit(store: Store, j: dict) -> tuple[bool, str]:
    """Can this job start now? Jobs that use the GPU run one at a time and need their estimate to fit."""
    d = KIND_DEFAULTS.get(j["kind"], {"gpu": True, "memory_estimate_gib": 12.0})
    uses_gpu = j["params"].get("gpu", d["gpu"])
    if uses_gpu:
        running = [r[0] for r in store.q("SELECT id, kind, params FROM jobs WHERE state IN ('running','cancelling')")
                   if json.loads(r[2] or "{}").get("gpu", KIND_DEFAULTS.get(r[1], {"gpu": True})["gpu"])]
        if running:
            return False, f"waiting for job {running[0]} to finish"
    need = float(j["params"].get("memory_estimate_gib", d["memory_estimate_gib"]))
    ms = memory_status(store)
    if need > ms["usable_gib"]:
        return False, (f"needs about {need:.0f} GiB; {ms['available_gib']:.0f} GiB is available and {ms['reserve_gib']:.0f} GiB "
                       f"is reserved for the system and other services")
    return True, "admitted"


# ---------------------------------------------------------------------------- worker
def start_job(store: Store, j: dict) -> None:
    d = job_dir(j["id"])
    log = d / "log.txt"
    cmd = [sys.executable, "-u", "-m", "mpnstudio.train.jobs", "run", j["id"]]
    env = {k: v for k, v in os.environ.items() if k.startswith(("MPN_", "HF_", "CUDA", "PATH", "HOME", "LANG", "XDG_", "PYTHON"))}
    env["MPN_DB"] = store.path
    cwd = str(Path(__file__).resolve().parents[2])
    if has_systemd():
        unit = _unit_name(j["id"])
        args = ["systemd-run", "--user", f"--unit={unit}", "--collect", "--quiet", f"--working-directory={cwd}",
                "-p", f"StandardOutput=append:{log}", "-p", f"StandardError=append:{log}", "-p", "KillMode=mixed",
                "-p", "TimeoutStopSec=180", "-p", "Restart=no",
                # GPU memory on the Spark's unified memory is not counted as RSS, so the kernel cannot tell that a job is
                # the large consumer; a high score makes MPN Studio's own job the first to go, not a neighbour's service.
                "-p", "OOMScoreAdjust=900"]
        for k, v in env.items():
            args += ["--setenv", f"{k}={v}"]
        r = subprocess.run(args + cmd, capture_output=True, text=True)
        if r.returncode != 0:
            update(store, j["id"], state="failed", error=f"could not start unit: {r.stderr.strip()[:300]}", finished=now())
            return
        update(store, j["id"], state="running", started=now(), unit=unit, heartbeat=now())
    else:
        with open(log, "ab") as f:
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd, env=env, start_new_session=True)
        update(store, j["id"], state="running", started=now(), pid=p.pid, heartbeat=now())


def reconcile(store: Store) -> list[str]:
    """Running jobs whose process is gone were interrupted (crash, reboot, kill -9). Mark them so they can be resumed."""
    changed = []
    for r in store.q("SELECT id FROM jobs WHERE state IN ('running','cancelling')"):
        j = get(store, r[0])
        started = time.mktime(time.strptime(j["started"], "%Y-%m-%dT%H:%M:%SZ")) - time.timezone if j["started"] else 0
        if time.time() - started < 8:
            continue
        if not job_alive(j):
            j2 = get(store, j["id"])
            if j2["state"] in ("running", "cancelling"):       # the job did not record its own ending
                state = "cancelled" if j2["state"] == "cancelling" else "interrupted"
                update(store, j["id"], state=state, finished=now(),
                       error=None if state == "cancelled" else "the job process ended without reporting a result")
                changed.append(j["id"])
    return changed


def worker_loop(poll: float = 3.0, once: bool = False) -> None:
    store = Store(db_path())
    print(f"worker: db={store.path} systemd={has_systemd()}", flush=True)
    while True:
        for jid in reconcile(store):
            print(f"worker: job {jid} was interrupted", flush=True)
            j = get(store, jid)
            if j["state"] == "interrupted" and j["params"].get("auto_resume", True) and j["attempt"] < 5:
                new = resume(store, jid)
                print(f"worker: queued {new} to resume {jid}", flush=True)
        for r in store.q("SELECT id FROM jobs WHERE state='queued' ORDER BY created"):
            j = get(store, r[0])
            ok, why = admit(store, j)
            if j["progress"].get("waiting") != (None if ok else why):
                update(store, j["id"], progress=dict(j["progress"], waiting=None if ok else why))
            if ok:
                print(f"worker: starting {j['id']} ({j['kind']})", flush=True)
                start_job(store, j)
                break
        if once:
            return
        time.sleep(poll)


# ---------------------------------------------------------------------------- inside a job process
class JobContext:
    """What a job implementation uses to report progress, checkpoint and notice cancellation."""

    def __init__(self, store: Store, job: dict):
        self.store, self.job, self.id = store, job, job["id"]
        self.dir = job_dir(self.id)
        self.params = job["params"]
        self.cancelled = False
        self._last_beat = 0.0
        signal.signal(signal.SIGTERM, self._on_term)
        signal.signal(signal.SIGINT, self._on_term)

    def _on_term(self, *_):
        self.cancelled = True
        print("job: stop requested; finishing the current step and saving", flush=True)

    def log(self, msg: str | dict) -> None:
        print(msg if isinstance(msg, str) else json.dumps(msg), flush=True)

    def progress(self, **fields) -> None:
        if time.monotonic() - self._last_beat < 1.0 and not fields.get("force"):
            return
        self._last_beat = time.monotonic()
        fields.pop("force", None)
        cur = json.loads(self.store.q1("SELECT progress FROM jobs WHERE id=?", (self.id,))[0] or "{}")
        cur.update(fields)
        cur.pop("waiting", None)
        update(self.store, self.id, progress=cur, heartbeat=now())

    def should_stop(self) -> bool:
        return self.cancelled

    def resume_dir(self) -> Path | None:
        p = self.params.get("resume_from")
        return Path(p) if p and Path(p).exists() else None


JOB_KINDS: dict = {}


def job_kind(name: str):
    def deco(fn):
        JOB_KINDS[name] = fn
        return fn
    return deco


@job_kind("sleep")
def _sleep_job(ctx: JobContext) -> dict:
    """Diagnostic job used by the tests: counts steps, checkpoints each one, resumes exactly."""
    total = int(ctx.params.get("steps", 5))
    state_file = ctx.dir / "state.json"
    step = 0
    prev = ctx.resume_dir()
    if prev and (prev / "state.json").exists():
        step = json.loads((prev / "state.json").read_text())["step"]
        ctx.log(f"resumed from step {step} of job {prev.name}")
    while step < total:
        if ctx.should_stop():
            return {"cancelled": True, "step": step}
        time.sleep(float(ctx.params.get("step_s", 0.2)))
        step += 1
        tmp = state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"step": step}))
        tmp.replace(state_file)
        ctx.progress(step=step, total=total, force=True)
        ctx.log(f"step {step}/{total}")
    return {"step": step}


def run_job(job_id: str) -> int:
    store = Store(db_path())
    j = get(store, job_id)
    if j is None:
        print(f"no job {job_id}", flush=True)
        return 2
    for mod in ("eval_jobs", "bank_jobs"):          # register job kinds; bank_jobs needs the model stack
        try:
            __import__(f"mpnstudio.train.{mod}")
        except Exception as e:                       # the queue itself must keep working without it
            if j["kind"] not in JOB_KINDS:
                print(f"{mod} unavailable: {type(e).__name__}: {e}", flush=True)
    fn = JOB_KINDS.get(j["kind"])
    update(store, job_id, pid=os.getpid(), heartbeat=now())
    if fn is None:
        update(store, job_id, state="failed", error=f"unknown job kind {j['kind']!r}", finished=now())
        return 2
    ctx = JobContext(store, j)
    ctx.log(f"job {job_id} kind={j['kind']} attempt={j['attempt']} pid={os.getpid()} params={json.dumps(j['params'])}")
    try:
        result = fn(ctx) or {}
        state = "cancelled" if result.get("cancelled") else "succeeded"
        update(store, job_id, state=state, result=result, finished=now(), heartbeat=now())
        ctx.log(f"job {job_id} {state}")
        return 0
    except BaseException as e:
        import traceback
        traceback.print_exc()
        update(store, job_id, state="failed", error=f"{type(e).__name__}: {e}"[:500], finished=now())
        return 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--once", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("job_id")
    args = ap.parse_args()
    if args.cmd == "worker":
        worker_loop(once=args.once)
    else:
        sys.exit(run_job(args.job_id))


if __name__ == "__main__":
    # Run through the canonical module, not this __main__ copy: job kinds defined elsewhere (bank_jobs, eval_jobs)
    # register themselves in mpnstudio.train.jobs.JOB_KINDS, which is a different dict from this copy's.
    from mpnstudio.train import jobs as _canonical
    _canonical.main()
