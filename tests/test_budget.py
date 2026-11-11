"""Token budget of one answer (mpnstudio/runtime/budget.py) and the chat endpoint's handling of it.

The packing tests use the real Qwen3-4B-Instruct-2507 tokenizer and chat template (no weights). Set MPN_TEST_TOKENIZER
to a directory with its tokenizer files; on the Spark the model directory is found by default."""
import json
import os
from pathlib import Path

import pytest

from mpnstudio.runtime import budget as tb

TOK_DIR = Path(os.environ.get("MPN_TEST_TOKENIZER", Path.home() / "mpn-studio" / "models" / "Qwen3-4B-Instruct-2507"))
DEFAULTS = {"context_tokens": 8192, "output_tokens": 1024}
CORE_MAX = 32768


# ---------------------------------------------------------------- limits (no tokenizer)
def test_client_max_tokens_is_a_ceiling_not_a_reservation():
    lim = tb.resolve(DEFAULTS, CORE_MAX, requested_output=4096)       # a chat client's quota default
    assert lim["context"] == 8192 and lim["output"] == 1024
    assert any("requested 4096" in n for n in lim["notes"])
    assert tb.resolve(DEFAULTS, CORE_MAX, requested_output=200)["output"] == 200
    assert tb.resolve(DEFAULTS, CORE_MAX)["output"] == 1024


def test_reported_4096_4096_pair_cannot_reach_generation():
    bad = {"context_tokens": 4096, "output_tokens": 4096}
    assert tb.check_budget_tokens(4096, 4096) is not None             # refused when saving
    fixed, note = tb.repair_budget(bad)                                # repaired when a legacy saved budget is read
    assert note and fixed["context_tokens"] == 4096 and fixed["output_tokens"] == 1024
    lim = tb.resolve(bad, CORE_MAX, requested_output=4096)            # and never usable even if it slipped through
    assert lim["output"] + tb.MIN_INPUT + lim["margin"] <= lim["context"]


def test_context_is_capped_by_the_core():
    assert tb.resolve({"context_tokens": 65536, "output_tokens": 1024}, CORE_MAX)["context"] == CORE_MAX


@pytest.mark.parametrize("v", [float("nan"), float("inf"), -5, 0, "x", 10.5])
def test_nonsense_limits_are_rejected(v):
    with pytest.raises(ValueError):
        tb.resolve({"context_tokens": 8192, "output_tokens": 1024}, CORE_MAX, requested_output=v)
    try:
        assert tb.check_budget_tokens(v, 1024) is not None      # refused with a reason
    except ValueError:
        pass                                                     # or refused as not a number


def test_valid_deliberate_choices_are_kept():
    assert tb.check_budget_tokens(2048, 256) is None
    b, note = tb.repair_budget({"context_tokens": 2048, "output_tokens": 256})
    assert note is None and b["output_tokens"] == 256


def test_explicit_zero_adapters_survives_defaults_storage_and_json(monkeypatch):
    from mpnstudio.api import app as appmod

    class S:
        def __init__(self, v):
            self.v = v

        def get_setting(self, k):
            return self.v if k == "budget" else None
    assert appmod.DEFAULT_BUDGET["max_active_adapters"] == 0
    assert appmod.get_budget(S(None))["max_active_adapters"] == 0
    assert appmod.get_budget(S(json.loads(json.dumps({"max_active_adapters": 0}))))["max_active_adapters"] == 0
    assert appmod.get_budget(S({"max_active_adapters": 1}))["max_active_adapters"] == 1   # deliberate choice kept
    legacy = appmod.get_budget(S({"context_tokens": 4096, "output_tokens": 4096}))
    assert legacy["output_tokens"] == 1024 and legacy["repair_note"]
    assert appmod.PRESETS["specialists (experimental)"]["max_active_adapters"] == 1
    assert all(p["max_active_adapters"] == 0 for n, p in appmod.PRESETS.items() if n != "specialists (experimental)")


# ---------------------------------------------------------------- packing with the real tokenizer
@pytest.fixture(scope="module")
def core():
    if not (TOK_DIR / "tokenizer.json").exists():
        pytest.skip(f"no tokenizer at {TOK_DIR}")
    transformers = pytest.importorskip("transformers")
    from mpnstudio.model.core import Core
    c = Core.__new__(Core)                       # the real chat_text() and ids(), without loading weights
    c.tok = transformers.AutoTokenizer.from_pretrained(str(TOK_DIR))
    return c


def node(i, words):
    return {"node": {"title": f"Node {i}", "text": " ".join(["fact"] * words)}}


def render(tn):
    return "\n".join(f"[{i + 1}] {n['node']['title']}: {n['node']['text']}" for i, n in enumerate(tn))


SYS = "You are Yaob, a model by Fylume. Answer from the knowledge nodes you are given."


def words_for_tokens(core, n):
    """A question of about n tokens, measured, not assumed."""
    q = "word " * n
    while len(core.ids(q)) > n:
        q = q[:-5]
    return q.strip()


def test_tiny_and_75_token_questions_fit_defaults(core):
    lim = tb.resolve(DEFAULTS, CORE_MAX, requested_output=4096)
    for q in ["hi", words_for_tokens(core, 75)]:
        p = tb.pack(core, SYS, [], [node(1, 50), node(2, 50), node(3, 50)], "", q, lim, render)
        assert "error" not in p and p["output"] == 1024 and len(p["text_nodes"]) == 3 and not p["reductions"]
        assert p["n_prompt"] == len(p["ids"]) == p["breakdown"]["total"]
        assert p["n_prompt"] + p["output"] + lim["margin"] <= lim["context"]


def test_2461_token_question_fits_defaults(core):
    q = words_for_tokens(core, 2461)
    assert 2400 <= len(core.ids(q)) <= 2461
    p = tb.pack(core, SYS, [], [node(1, 200)], "", q, tb.resolve(DEFAULTS, CORE_MAX, requested_output=4096), render)
    assert "error" not in p and p["output"] == 1024 and len(p["text_nodes"]) == 1


def test_exact_fit_and_one_token_over(core):
    q = "What is the capital of Australia?"
    probe = tb.pack(core, SYS, [], [], "", q, tb.resolve(DEFAULTS, CORE_MAX), render)
    n = probe["n_prompt"]
    out = tb.MIN_OUTPUT                           # at the floor, so nothing optional is left to shrink
    exact = {"context_tokens": n + out + tb.SAFETY_MARGIN, "output_tokens": out}
    lim = tb.resolve(exact, 10**6)
    lim["output"] = out                            # resolve() would cap a tiny context's answer; test the rule itself
    p = tb.pack(core, SYS, [], [], "", q, lim, render)
    assert "error" not in p and p["n_prompt"] == n and p["output"] == out
    lim_over = dict(lim, context=lim["context"] - 1)
    p = tb.pack(core, SYS, [], [], "", q, lim_over, render)
    assert p["error"]["status"] == 413
    msg = p["error"]["error"]
    assert f"({n} tokens) exceeds the available context size ({lim_over['context']} tokens)" in msg   # the form OpenAI-style clients parse
    assert "question alone" not in msg and p["breakdown"]["total"] == n


def test_long_material_is_reduced_in_order_with_a_true_breakdown(core):
    lim = tb.resolve({"context_tokens": 2048, "output_tokens": 512}, CORE_MAX)
    hist = [{"role": "user", "content": "earlier " * 600}, {"role": "assistant", "content": "reply " * 600}]
    nodes = [node(i, 500) for i in range(1, 4)]           # about 500 tokens each: even one node + 1,200 history + 512 does not fit 2,048
    p = tb.pack(core, SYS, hist, nodes, "", "When was it founded?", lim, render)
    assert "error" not in p
    r = p["reductions"]
    assert r[0].startswith("Dropped the text of “Node 3”") and r[1].startswith("Dropped the text of “Node 2”")
    assert any("earlier turns" in x for x in r)
    assert [n["node"]["title"] for n in p["text_nodes"]] == ["Node 1"]      # the best node's text is kept
    bd = p["breakdown"]
    assert bd["total"] == p["n_prompt"] == len(p["ids"]) and bd["template"] >= 0 and bd["node_text"] > 0 and bd["history"] == 0
    assert p["n_prompt"] + p["output"] + lim["margin"] <= lim["context"]


def test_answer_allowance_shrinks_before_the_last_node_is_dropped(core):
    lim = tb.resolve({"context_tokens": 2048, "output_tokens": 1024}, CORE_MAX)
    p = tb.pack(core, SYS, [], [node(1, 1300)], "", "When was it founded?", lim, render)   # ~1,360 input + 1,024 > 2,048
    assert "error" not in p and len(p["text_nodes"]) == 1
    assert tb.MIN_OUTPUT <= p["output"] < 1024 and p["reductions"][0].startswith("Answer limited to")


def test_irreducible_input_is_refused_with_counts_and_question_untouched(core):
    lim = tb.resolve({"context_tokens": 2048, "output_tokens": 512}, CORE_MAX)
    q = words_for_tokens(core, 3000)
    p = tb.pack(core, SYS + " Be helpful." * 50, [], [node(1, 50)], "", q, lim, render)
    e = p["error"]
    assert e["status"] == 413 and e["breakdown"]["question"] >= 2900 and e["breakdown"]["system"] > 100
    assert e["limits"] == {"context": 2048, "output": 512, "margin": tb.SAFETY_MARGIN}


# ---------------------------------------------------------------- chat endpoint
class FakeEngine:
    """Scripted engine events; records the budget each request was given."""
    script: list = []
    seen: list = []

    def __init__(self, *a, **k):
        pass

    def status(self):
        return {"core": {"loaded": True, "name": "Qwen3-4B-Instruct-2507", "params_total": 4022468096, "max_context": 32768},
                "checkpoint": {"id": "ck-test"}, "limits": {"context_tokens": 8192, "output_tokens": 1024, "max_active_adapters": 0}}

    def run(self, messages, mode, rev, request_id, budget, options):
        FakeEngine.seen.append(dict(budget or {}))
        yield from FakeEngine.script.pop(0)

    def cancel(self, rid):
        return True

    def __getattr__(self, name):
        return lambda *a, **k: {}


@pytest.fixture
def chat(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    monkeypatch.setenv("MPN_RUNTIME_AUTOLOAD", "0")
    monkeypatch.setenv("MPN_DATA", str(tmp_path))
    monkeypatch.setenv("MPN_DB", str(tmp_path / "x.sqlite"))
    from fastapi.testclient import TestClient

    from mpnstudio.runtime import server
    monkeypatch.setattr(server, "Engine", FakeEngine)
    FakeEngine.script, FakeEngine.seen = [], []
    return TestClient(server.create_app())


def ok_events(text="Canberra [1]"):
    return [("start", {"request_id": "r1", "rev": 1}), ("route", {"nodes": [{"id": "n1", "read_as": ["text"]}], "adapters": []}),
            ("token", {"text": text}), ("validation", {}),
            ("done", {"text": text, "usage": {"prompt_tokens": 90, "completion_tokens": 4}, "timing": {}, "memory": {}})]


def test_models_report_the_real_context(chat):
    m = chat.get("/v1/models").json()["data"][0]
    assert m["meta"]["n_ctx"] == 8192 and m["meta"]["max_output_tokens"] == 1024 and m["meta"]["max_active_adapters"] == 0


def test_max_tokens_is_passed_as_a_request_not_as_the_cap(chat):
    FakeEngine.script = [ok_events()]
    r = chat.post("/v1/chat/completions", json={"model": "yaob-n10k", "stream": True, "max_tokens": 4096,
                                                 "messages": [{"role": "user", "content": "Capital of Australia?"}]})
    assert r.status_code == 200 and "Canberra" in r.text and r.text.rstrip().endswith("data: [DONE]")
    assert FakeEngine.seen[0] == {"requested_output_tokens": 4096}


def test_refusal_during_budgeting_is_an_http_error_with_the_breakdown(chat):
    bd = {"question": 7000, "system": 40, "history": 0, "node_text": 0, "tool_text": 0, "template": 9, "total": 7049}
    FakeEngine.script = [[("start", {"request_id": "r1", "rev": 1}),
                          ("error", {"error": "The request (7049 tokens) exceeds the available context size (8192 tokens).", "status": 413,
                                     "breakdown": bd, "limits": {"context": 8192, "output": 1024, "margin": 16}})]]
    r = chat.post("/v1/chat/completions", json={"model": "yaob-n10k", "stream": True, "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 413 and r.json()["error"]["breakdown"]["total"] == 7049


def test_failure_after_text_started_is_visible_and_the_next_request_works(chat):
    FakeEngine.script = [ok_events("Partial")[:3] + [("error", {"error": "out of memory", "status": 507})], ok_events()]
    r = chat.post("/v1/chat/completions", json={"model": "yaob-n10k", "stream": True, "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200 and "[Yaob stopped: out of memory]" in r.text and '"finish_reason": "stop"' in r.text
    r = chat.post("/v1/chat/completions", json={"model": "yaob-n10k", "stream": True, "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200 and "Canberra" in r.text
