import os
import signal
import time

import pytest

from mpnstudio.store.db import Store
from mpnstudio.train import jobs


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MPN_DATA", str(tmp_path))
    monkeypatch.setenv("MPN_DB", str(tmp_path / "m.sqlite"))
    monkeypatch.setenv("MPN_NO_SYSTEMD", "1")
    return Store(tmp_path / "m.sqlite")


def wait(store, jid, states, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = jobs.get(store, jid)
        if j["state"] in states:
            return j
        jobs.reconcile(store)
        time.sleep(0.1)
    raise AssertionError(f"job {jid} stuck in {jobs.get(store, jid)['state']}")


def test_job_runs_to_completion_with_log_and_progress(env):
    jid = jobs.submit(env, "sleep", {"steps": 3, "step_s": 0.05, "gpu": False, "memory_estimate_gib": 0})
    jobs.worker_loop(once=True)
    j = wait(env, jid, ("succeeded",))
    assert j["result"]["step"] == 3 and j["progress"]["step"] == 3
    assert "step 3/3" in open(j["log_path"]).read()


def test_cancel_saves_state_and_resume_continues_from_it(env):
    jid = jobs.submit(env, "sleep", {"steps": 40, "step_s": 0.1, "gpu": False, "memory_estimate_gib": 0})
    jobs.worker_loop(once=True)
    wait(env, jid, ("running",))
    t0 = time.time()      # wait for real progress: on a loaded machine the worker process can take seconds to start
    while ((jobs.get(env, jid).get("progress") or {}).get("step") or 0) < 3 and time.time() - t0 < 20:
        time.sleep(0.1)
    jobs.cancel(env, jid)
    j = wait(env, jid, ("cancelled",))
    done = j["result"]["step"]
    assert 0 < done < 40
    new = jobs.resume(env, jid)
    jobs.update(env, new, params=dict(jobs.get(env, new)["params"], steps=done + 2))
    jobs.worker_loop(once=True)
    j2 = wait(env, new, ("succeeded",))
    assert j2["resume_of"] == jid and j2["attempt"] == 2 and j2["result"]["step"] == done + 2
    assert f"resumed from step {done}" in open(j2["log_path"]).read()


def test_killed_job_is_marked_interrupted_and_requeued(env):
    jid = jobs.submit(env, "sleep", {"steps": 200, "step_s": 0.1, "gpu": False, "memory_estimate_gib": 0})
    jobs.worker_loop(once=True)
    j = wait(env, jid, ("running",))
    time.sleep(0.6)
    j = jobs.get(env, jid)
    os.kill(j["pid"], signal.SIGKILL)
    time.sleep(0.3)
    # the "started" guard in reconcile is 8 s; move the start time back so the check applies now
    env.conn().execute("UPDATE jobs SET started='2020-01-01T00:00:00Z' WHERE id=?", (jid,))
    jobs.worker_loop(once=True)
    assert jobs.get(env, jid)["state"] == "interrupted"
    resumed = [x for x in jobs.list_jobs(env) if x["resume_of"] == jid]
    assert len(resumed) == 1 and resumed[0]["attempt"] == 2
    jobs.cancel(env, resumed[0]["id"]) if resumed[0]["state"] != "queued" else jobs.update(env, resumed[0]["id"], state="cancelled")
    for x in jobs.list_jobs(env):
        if x["state"] == "running":
            jobs.cancel(env, x["id"])
            wait(env, x["id"], ("cancelled",))


def test_admission_refuses_a_job_that_does_not_fit(env):
    jid = jobs.submit(env, "sleep", {"steps": 1, "memory_estimate_gib": 100000})
    ok, why = jobs.admit(env, jobs.get(env, jid))
    assert not ok and "GiB" in why
    jobs.worker_loop(once=True)
    j = jobs.get(env, jid)
    assert j["state"] == "queued" and "GiB" in j["progress"]["waiting"]


def test_gpu_jobs_run_one_at_a_time(env):
    a = jobs.submit(env, "sleep", {"steps": 30, "step_s": 0.1, "memory_estimate_gib": 0})
    b = jobs.submit(env, "sleep", {"steps": 1, "step_s": 0.01, "memory_estimate_gib": 0})
    jobs.worker_loop(once=True)
    wait(env, a, ("running",))
    jobs.worker_loop(once=True)
    assert jobs.get(env, b)["state"] == "queued" and "waiting for job" in jobs.get(env, b)["progress"]["waiting"]
    jobs.cancel(env, a)
    wait(env, a, ("cancelled",))
    jobs.worker_loop(once=True)
    wait(env, b, ("succeeded",))


def test_job_runner_started_as_a_module_sees_kinds_registered_elsewhere(env, tmp_path):
    """`python -m mpnstudio.train.jobs run <id>` is how the worker starts every job; kinds from eval_jobs must resolve."""
    import subprocess
    import sys
    jid = jobs.submit(env, "evaluate", {"suite": str(tmp_path / "missing.json")})
    r = subprocess.run([sys.executable, "-m", "mpnstudio.train.jobs", "run", jid], capture_output=True, text=True, timeout=120,
                       # nothing listens on port 9: on the Spark a real runtime answers on 8841 and the job would run a
                       # full evaluation against it (that was the 120 s timeout there)
                       env={**os.environ, "MPN_DB": str(env.path), "MPN_NO_SYSTEMD": "1", "MPN_RUNTIME_URL": "http://127.0.0.1:9"})
    j = jobs.get(env, jid)
    assert "unknown job kind" not in (j.get("error") or ""), (j.get("error"), r.stdout[-500:], r.stderr[-500:])
