"""Prompt text shared by training and the runtime. Specialists are trained against exactly these
prefixes and this node rendering, so changing them means retraining the adapters."""

SYS_YAOB = ("You are Yaob, a model by Fylume. Answer from the knowledge nodes you are given or have learned. Cite text nodes as [1], [2]. "
            "If a tool result is given, use its value exactly. If nothing covers the question, say so. Be concise.")
SYS_RETRIEVAL = SYS_YAOB
SYS_CORE = "You are Yaob, a model by Fylume. Answer the question concisely. If you do not know, say so."
QUESTION_PREFIX = "Question: "


def fact_period(f: dict) -> str:
    """When a fact holds, if the source says: " (as of 2021)", " (1621 to 1638)", " (from 1729)", " (until 1858)"."""
    if f.get("as_of"):
        return f" (as of {f['as_of']})"
    a, b = f.get("from"), f.get("until")
    if a and b:
        return f" ({a} to {b})"
    if a or b:
        return f" (from {a})" if a else f" (until {b})"
    return ""


def render_node(n: dict, i: int, max_statements: int = 6, max_facts: int = 14, focus: set | None = None) -> str:
    """One node as text evidence: numbered title, statements, facts, and the constraints an answer must respect.
    `focus`: fact properties the question names; those facts are listed first so the 14-fact limit cannot cut them
    (multihop-dev-001: 6 of 60 second-hop facts were cut). Without it the order is the stored order, as in training."""
    k = n["cells"].get("knowledge", {})
    lines = [s["text"] for s in k.get("statements", [])[:max_statements] if s.get("text")]
    all_facts = k.get("facts", [])
    if focus:
        all_facts = [f for f in all_facts if str(f.get("property", "")).lower() in focus] + \
                    [f for f in all_facts if str(f.get("property", "")).lower() not in focus]
    facts = [f"{f['property']}: {f['value']}" + fact_period(f) for f in all_facts[:max_facts] if f.get("value")]
    cons = [x["text"] for x in n["cells"].get("reasoning", {}).get("constraints", [])[:3] if x.get("text")]
    return f"[{i}] {n['title']}\n" + " ".join(lines) + ("\nFacts: " + "; ".join(facts) if facts else "") + ("\nNote: " + " ".join(cons) if cons else "")


def render_nodes(nodes: list[dict], focus: set | None = None) -> str:
    return "\n\n".join(render_node(n, i, focus=focus) for i, n in enumerate(nodes, start=1))
