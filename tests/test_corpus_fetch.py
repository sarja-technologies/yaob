"""Wikidata reads: a response the API truncated is split and asked again, never silently shortened."""
from mpnstudio.corpus import fetch


class FakeClient:
    def __init__(self, limit):
        self.limit, self.calls = limit, []

    def get(self, host, **params):
        ids = params["ids"].split("|")
        self.calls.append(len(ids))
        if len(ids) > self.limit:                       # like the 12 MB cut: some entities, plus a warning
            return {"warnings": {"result": {"warnings": "This result was truncated because it would otherwise be larger than the limit"}},
                    "entities": {q: {"id": q} for q in ids[: self.limit]}}
        return {"entities": {q: {"id": q} for q in ids}}


def test_truncated_batches_are_split_until_complete():
    c = FakeClient(limit=12)
    qids = [f"Q{i}" for i in range(50)]
    got = fetch.fetch_entities(c, qids, "claims", "entities")
    assert sorted(got) == sorted(qids)
    assert c.calls[0] == 50 and max(c.calls[1:]) <= 25
