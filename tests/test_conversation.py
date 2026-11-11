"""Follow-up routing, small talk and citation checking (mpnstudio/runtime/conversation.py)."""
from mpnstudio.runtime import conversation as conv


def msgs(*turns):
    out = []
    for i, t in enumerate(turns):
        out.append({"role": "user" if i % 2 == 0 else "assistant", "content": t})
    return out


def test_follow_up_is_routed_with_the_previous_question():
    m = msgs("When was the Ottoman Empire dissolved?", "On 17 November 1922 [1].", "And who was its last sultan?")
    q, used = conv.routing_query(m[-1]["content"], m)
    assert used and "Ottoman Empire" in q and q.endswith("And who was its last sultan?")


def test_topic_change_is_routed_alone():
    m = msgs("When was the Ottoman Empire dissolved?", "1922 [1].", "Who wrote Hamlet?")
    assert conv.routing_query(m[-1]["content"], m) == ("Who wrote Hamlet?", False)
    m = msgs("When was the Ottoman Empire dissolved?", "1922 [1].", "What is its capital, Paris or Rome?")   # names of its own, no lead
    assert conv.routing_query(m[-1]["content"], m)[1] is False


def test_what_about_keeps_the_earlier_predicate():
    m = msgs("What is the capital of Peru?", "Lima [1].", "And what about Chile?")
    q, used = conv.routing_query(m[-1]["content"], m)
    assert used and "capital of Peru" in q and "Chile" in q


def test_first_message_and_history_bound():
    assert conv.routing_query("And its population?", [{"role": "user", "content": "And its population?"}]) == ("And its population?", False)
    long = "word " * 500
    m = msgs(long, "ok", "What about it?")
    q, used = conv.routing_query(m[-1]["content"], m)
    assert used and len(q) <= 400 + 1 + len("What about it?")


def test_small_talk():
    for s in ["hi", "Hello!", "hi, who are you?", "thanks", "Thank you very much", "who are you", "good morning"]:
        assert conv.is_small_talk(s), s
    for s in ["hi, what is the capital of Peru?", "Who are you, Ottoman Empire?x", "When was Hindi standardised?"]:
        assert not conv.is_small_talk(s), s


def run(nodes, pieces):
    f = conv.CitationFilter(nodes)
    out = "".join(f.feed(p) for p in pieces) + f.finish()
    return out, f.removed


def test_unsupported_and_out_of_range_citations_are_removed_while_streaming():
    nodes = ["Ethanol. Facts: boiling point: 78.37 degree Celsius; chemical formula: C2H6O"]
    out, removed = run(nodes, ["Hello! How can I", " assist you today?", " [", "1]"])
    assert out == "Hello! How can I assist you today?" and removed[0]["n"] == 1
    out, removed = run(nodes, ["The boiling point of ethanol is 78.37 degree Celsius ", "[1]. It is ", "flammable [2]."])
    assert out == "The boiling point of ethanol is 78.37 degree Celsius [1]. It is flammable." and [r["n"] for r in removed] == [2]


def test_brackets_that_are_not_citations_pass_through():
    out, removed = run(["x"], ["The array a[i] and the list [a, b] ", "stay."])
    assert out == "The array a[i] and the list [a, b] stay." and not removed


def test_supports():
    assert conv.supports("Its first capital city was Melbourne", "Melbourne. Melbourne was the capital of Australia from 1901 to 1927.")
    assert not conv.supports("I am Yaob, a model by Fylume", "Hindi. Hindi is an Indo-Aryan language spoken in India.")


def test_relation_and_property_labels():
    assert "born_in" in conv.relations_named("What is the continent of the place of birth of Otto Hahn?")
    assert "continent" in conv.properties_named("What is the continent of the place of birth of Otto Hahn?")
    assert conv.relations_named("hi") == set()
