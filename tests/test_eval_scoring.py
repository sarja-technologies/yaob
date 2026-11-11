"""Deterministic scoring of the held-out evaluation."""
from mpnstudio.evals.heldout import gold_in_question, score_task


def test_an_abstaining_reply_is_wrong_even_if_it_names_the_value():
    t = {"scoring": "values", "gold": ["Paris"]}
    assert score_task(t, "The capital of France is Paris.") == (True, True)
    assert score_task(t, "I have no information about Paris.") == (False, True)


def test_unicode_subscripts_and_thousands_separators_match():
    assert score_task({"scoring": "values", "gold": ["C₁₃H₁₈O₂"]}, "Its formula is C13H18O2.")[0]
    assert score_task({"scoring": "result", "gold": 1022}, "A /22 network has 1,022 usable hosts.")[0]


def test_unsupported_questions_want_an_abstention():
    t = {"scoring": "abstain", "gold": None}
    assert score_task(t, "I don't know of any Wexqua Canal.")[0]
    assert not score_task(t, "The Wexqua Canal is 42 km long.")[0]


def test_tasks_whose_answer_is_in_the_question_are_detected():
    assert gold_in_question("What is the country of the capital of France?", ["France"])
    assert not gold_in_question("What is the capital of France?", ["Paris"])


def test_refusals_seen_in_eval_001_count_as_abstaining():
    from mpnstudio.evals.heldout import abstains
    for reply in ["None of the provided knowledge nodes mention the Bexsarnyun Conservatory or its founding year. "
                  "Therefore, based on the given information, the founding year cannot be determined.",
                  "The Wexrin Canal is not mentioned in any of the provided knowledge nodes. Therefore, its length cannot be determined.",
                  "The knowledge nodes provided do not specify who invented it."]:
        assert abstains(reply)
    for reply in ["April [1]", "Anders Celsius [2]", "It was founded in 1824 and is still determined to grow [1]."]:
        assert not abstains(reply)


def test_hedged_assertions_about_an_invented_entity_get_no_refusal_credit():
    from mpnstudio.evals.heldout import score_task
    t = {"scoring": "abstain", "gold": None, "invented": "Garlocht"}
    for reply in ["No node mentions it, but the Garlocht Regatta is held in April [1].",
                  "The Garlocht Regatta is held in April. However, the provided nodes do not mention it.",
                  "None of the provided knowledge nodes mention this regatta; it takes place in 1987."]:
        assert score_task(t, reply) == (False, False), reply
    for reply in ["None of the provided knowledge nodes mention the Garlocht Regatta. Therefore the month cannot be determined.",
                  "I do not know about the Garlocht Regatta. It may be a fictional event.",
                  # eval-001, retrieval, unsupported-014: a real refusal that cites a year of a different observatory
                  "There is no information in the provided knowledge nodes about the founding of a \"Minrin Observatory.\"\n\n"
                  "The only observatory mentioned is the Uppsala Astronomical Observatory, which was founded by Anders Celsius in 1741 [3].\n\n"
                  "Therefore, based on the given knowledge, the question cannot be answered."]:
        assert score_task(dict(t, invented="Minrin" if "Minrin" in reply else "Garlocht"), reply) == (True, True), reply


def test_refusal_on_an_answerable_question_is_never_credited():
    from mpnstudio.evals.heldout import score_task
    t = {"scoring": "values", "gold": ["Canberra"]}
    assert score_task(t, "The nodes do not mention it, but it may be Canberra.")[0] is False
    assert score_task(t, "The capital of Australia is Canberra [1].")[0] is True
