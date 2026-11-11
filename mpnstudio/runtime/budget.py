"""The token budget of one generation call.

One rule for every call the runtime makes to the core:

    serialized input tokens + max new tokens + SAFETY_MARGIN <= effective context

* effective context = min(the budget's `context_tokens`, the core's configured `max_context` from CORE.json).
  Generation runs in-process with transformers; the KV cache grows with the sequence, so this is the real limit.
* `output_tokens` in the budget is the most the runtime will generate for one answer (a cap). A client's
  `max_tokens` (OpenAI semantics: an upper bound on generated tokens) is honoured up to that cap and never
  reserves more than the cap.
* The input count is exact: the token ids of the chat-template text that is sent to the core, special tokens included.
  Per-part counts in the breakdown are measured separately and are approximate (tokenisation across part boundaries
  is not additive); the remainder is reported as template overhead. The total is authoritative.

When the input does not fit, optional material is reduced in this order and each step is reported:
  1. node text beyond the best node, 2. the oldest conversation turns, 3. the answer allowance, down to
  MIN_OUTPUT (or the requested size if smaller), 4. the last node's text.
The question and the system instructions are never cut. If they alone do not fit, the request is refused with the
measured counts.
"""
from __future__ import annotations

import math

SAFETY_MARGIN = 16      # tokens kept free beyond input + output; the input count is exact, so this is small
MIN_OUTPUT = 256        # the answer allowance is not shrunk below this to make room for optional material
MIN_INPUT = 512         # a budget must leave at least this much room for input after the output cap


def _int(v, name: str) -> int:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, not {v!r}.")
    if not math.isfinite(f) or f != int(f):
        raise ValueError(f"{name} must be a whole number, not {v!r}.")
    return int(f)


def check_budget_tokens(context_tokens, output_tokens) -> str | None:
    """Why a saved budget's context/output pair cannot work, or None. Used when saving a budget."""
    ctx, out = _int(context_tokens, "context_tokens"), _int(output_tokens, "output_tokens")
    if ctx <= 0 or out <= 0:
        return "context_tokens and output_tokens must be positive."
    if out + MIN_INPUT + SAFETY_MARGIN > ctx:
        return (f"output_tokens ({out}) leaves no room for the question: with a {ctx}-token context the answer cap can be at most "
                f"{ctx - MIN_INPUT - SAFETY_MARGIN} ({MIN_INPUT} tokens are kept for input and {SAFETY_MARGIN} as margin).")
    return None


def repair_budget(b: dict) -> tuple[dict, str | None]:
    """A budget saved before the context/output check existed may hold an impossible pair (e.g. 4096/4096). Repair it for
    use, without writing it back: the output cap becomes a quarter of the context. Returns (budget, note or None)."""
    try:
        why = check_budget_tokens(b.get("context_tokens"), b.get("output_tokens"))
    except ValueError as e:
        why = str(e)
    if why is None:
        return b, None
    b = dict(b)
    try:
        ctx = max(_int(b.get("context_tokens"), "context_tokens"), 2048)
    except ValueError:
        ctx = 8192
    b["context_tokens"], b["output_tokens"] = ctx, max(ctx // 4, 16)
    return b, f"The saved budget was not usable ({why}) Using context {ctx} and answer cap {b['output_tokens']} until it is saved again."


def resolve(budget: dict, core_max_context: int, requested_output: int | None = None) -> dict:
    """The limits that apply to one answer."""
    ctx_budget = _int(budget["context_tokens"], "context_tokens")
    cap = _int(budget["output_tokens"], "output_tokens")
    context = min(ctx_budget, int(core_max_context))
    notes = []
    if requested_output is not None:
        requested_output = _int(requested_output, "max_tokens")
        if requested_output <= 0:
            raise ValueError("max_tokens must be positive.")
    out = cap if requested_output is None else min(requested_output, cap)
    if requested_output is not None and requested_output > cap:
        notes.append(f"Answer limited to {cap} tokens (requested {requested_output}; the runtime's answer cap is {cap}).")
    if out + MIN_INPUT + SAFETY_MARGIN > context:     # an impossible pair must never reach generation
        new = max(context // 4, 16)
        notes.append(f"Answer limited to {new} tokens: {out} would leave no room for input in the {context}-token context.")
        out = new
    return {"context": context, "context_budget": ctx_budget, "core_max_context": int(core_max_context), "output_cap": cap,
            "output_requested": requested_output, "output": out, "margin": SAFETY_MARGIN, "notes": notes}


def fits(n_input: int, out: int, limits: dict) -> bool:
    return n_input + out + limits["margin"] <= limits["context"]


def too_long_message(bd: dict, limits: dict, out: int) -> str:
    """The refusal when the question and instructions alone do not fit. It also carries the phrase llama.cpp uses, which
    OpenAI-style chat clients parse to learn the context size and trim older turns."""
    parts = ", ".join(f"{k} {v}" for k, v in (("question", bd.get("question")), ("instructions", bd.get("system")),
                                                 ("conversation", bd.get("history")), ("node text", bd.get("node_text")),
                                                 ("tool results", bd.get("tool_text")), ("chat template", bd.get("template")))
                      if v)
    room = limits["context"] - out - limits["margin"]
    return (f"The request ({bd['total']} tokens) exceeds the available context size ({limits['context']} tokens). "
            f"Input {bd['total']} tokens ({parts}) + answer allowance {out} + margin {limits['margin']} is more than the "
            f"{limits['context']}-token context, which leaves {room} tokens for input. Shorten the message or raise context_tokens.")


def pack(core, system: str, history: list[dict], text_nodes: list, tool_text: str, question: str, limits: dict, render,
         specialist: bool = False) -> dict:
    """Assemble the prompt for one answer and fit it to `limits`, reducing optional material in the documented order.
    `core` provides chat_text() and ids() (the loaded core's tokenizer and chat template). Returns ids, n_prompt,
    output, the text nodes actually supplied, a token breakdown and the reductions made; or an `error` (413)."""
    from ..model.prompts import QUESTION_PREFIX
    history, text_nodes = list(history), list(text_nodes)
    ctx, out = limits["context"], limits["output"]
    floor_out = min(out, MIN_OUTPUT)
    reductions: list[str] = []

    def ntok(t: str) -> int:
        return len(core.ids(t)) if t else 0

    def breakdown(n_total: int) -> dict:
        bd = {"question": ntok(QUESTION_PREFIX + question), "system": ntok(system), "history": sum(ntok(m["content"]) for m in history),
              "node_text": ntok(render(text_nodes)) if text_nodes else 0, "tool_text": ntok(tool_text.strip()), "total": n_total}
        bd["template"] = max(n_total - sum(v for k, v in bd.items() if k != "total"), 0)
        return bd

    while True:
        body = ""
        if text_nodes:
            body += "Knowledge nodes:\n" + render(text_nodes) + "\n\n"
        if tool_text:
            body += tool_text.strip() + "\n\n"
        body += QUESTION_PREFIX + question
        msgs = [{"role": "system", "content": system}] + history + [{"role": "user", "content": body}]
        ids = core.ids(core.chat_text(msgs))
        n = len(ids)
        if fits(n, out, limits):
            return {"ids": ids, "n_prompt": n, "output": out, "text_nodes": text_nodes, "history": history,
                    "breakdown": breakdown(n), "reductions": reductions}
        room = ctx - n - limits["margin"]
        if len(text_nodes) > 1:
            dropped = text_nodes.pop()
            reductions.append(f"Dropped the text of “{dropped['node']['title']}” to fit the {ctx}-token context.")
        elif history:
            history = history[2:]
            reductions.append("Dropped earlier turns of the conversation to fit the context.")
        elif floor_out <= room < out:
            reductions.append(f"Answer limited to {room} tokens (from {out}) so the question and node text fit the {ctx}-token context.")
            out = room
        elif text_nodes:
            dropped = text_nodes.pop()
            reductions.append(f"Dropped the text of “{dropped['node']['title']}” to fit the {ctx}-token context; the answer relies on routing"
                              + (" and the specialist" if specialist else "") + " only.")
        else:
            bd = breakdown(n)
            return {"reductions": reductions, "breakdown": bd,
                    "error": {"error": too_long_message(bd, limits, out), "status": 413, "breakdown": bd,
                              "limits": {"context": ctx, "output": out, "margin": limits["margin"]}}}
