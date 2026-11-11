"""Conversation handling around one Yaob request: what the router reads, small talk, which facts a node shows first,
and citations that must point at supplied, supporting node text.

* Routing query. The router reads the newest message. When that message depends on an earlier one ("And who was its
  last sultan?": an anaphor or a leading "and/also/so", and no name of its own), the previous user turn (at most two,
  at most 400 characters) is prepended for routing only. A message with its own name ("Who wrote Hamlet?") is a topic
  change and is routed alone. The prompt to the core always carries the bounded history as before.
* Small talk (greetings, thanks, "who are you") reads no nodes, so the reply cannot cite any.
* Citations. While the answer streams, each "[k]" is checked when it completes: it is kept only if k names a node whose
  text was supplied and the sentence it closes shares a number or enough content words with that node's text.
  Otherwise it is removed before the client sees it, and the removal is reported.
"""
from __future__ import annotations

import re

ANAPHOR = re.compile(r"\b(?:it|its|it's|they|them|their|theirs|he|him|his|she|her|hers|this|that|these|those|there|then|the same|"
                     r"the former|the latter)\b", re.I)
LEAD = re.compile(r"^\s*(?:and|also|so|but|then|plus|what about|how about|and what|and who|and when|and where)\b", re.I)
NAME = re.compile(r"(?<![.?!]\s)(?<!^)\b(?:[A-Z][a-z]+|[A-Z]{2,})\b")
SMALL_TALK = re.compile(r"^\s*(?:hi+|hello+|hey+|yo|hiya|howdy|good (?:morning|afternoon|evening|night)|thanks?(?: you)?(?: (?:very|so) much)?|"
                        r"thank you|cheers|ok(?:ay)?|cool|great|nice|bye|goodbye|see you|how are you(?: doing)?|"
                        r"who are you|what are you|what is your name|what's your name|who made you|who built you|"
                        r"(?:hi|hello|hey)[, ]+(?:there|yaob)?[, ]*(?:who are you|how are you)?)\s*[!?.,]*\s*$", re.I)
STOP = set("""a an the and or but of to in on at by for with from as is are was were be been being this that these those it its
which who whom whose what when where why how than then there their they them he she his her we you your our i me my
not no yes do does did done has have had can could would should may might will shall about into over under after before
also very more most such only other some any each all both few many much one two three first last same own so just
answer question node nodes knowledge provided given information based according""".split())


def routing_query(question: str, messages: list[dict]) -> tuple[str, bool]:
    """(text the router should read, whether earlier turns were used)."""
    q = question or ""
    own_name = [w for w in NAME.findall(q) if w.lower() not in STOP]
    dependent = bool(LEAD.search(q)) or (bool(ANAPHOR.search(q)) and not own_name)
    if not dependent or own_name and not LEAD.search(q):
        return q, False
    prev = [m["content"] for m in messages if m.get("role") == "user" and (m.get("content") or "").strip()][:-1][-2:]
    if not prev:
        return q, False
    ctx = " ".join(p.strip() for p in prev)[-400:]
    return ctx + " " + q, True


def is_small_talk(question: str) -> bool:
    return bool(SMALL_TALK.match(question or ""))


_LABELS: dict | None = None


def _labels() -> dict:
    global _LABELS
    if _LABELS is None:
        from ..corpus.fetch import PROPS
        _LABELS = {"relations": {t: label.lower() for label, t in PROPS.values() if t and label},
                   "properties": {label.lower() for label, _t in PROPS.values() if label}}
    return _LABELS


def _norm(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", (text or "").lower()) + " "


def relations_named(question: str) -> set[str]:
    """Edge types whose Wikidata property label (corpus/fetch.py, e.g. P19 "place of birth" -> born_in) or own name the question contains."""
    q = _norm(question)
    return {t for t, label in _labels()["relations"].items() if f" {label} " in q or f" {t.replace('_', ' ')} " in q}


def properties_named(question: str) -> set[str]:
    """Fact property labels the question contains ("population", "date of birth", ...)."""
    q = _norm(question)
    return {p for p in _labels()["properties"] if f" {p} " in q}


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z\-']{3,}", (text or "").lower()) if w not in STOP}


def supports(sentence: str, node_text: str) -> bool:
    s = re.sub(r"\[\d+\]", " ", sentence)
    nums = {n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", s)}
    nt = node_text.lower().replace(",", "")
    if nums and any(n in nt for n in nums):
        return True
    words = content_words(s)
    if not words:
        return False
    hit = sum(1 for w in words if w in nt)
    return hit >= 2 and hit / len(words) >= 0.25


class CitationFilter:
    """Streams text through, holding back only a citation until it is complete, then keeping or dropping it."""
    CITE = re.compile(r"\s*\[(\d+)\]")

    def __init__(self, node_texts: list[str]):
        self.node_texts = node_texts
        self.out = ""          # text released so far
        self.buf = ""          # held back: a possible citation in progress
        self.removed: list[dict] = []

    def _sentence(self) -> str:
        cut = max(self.out.rfind(". "), self.out.rfind("! "), self.out.rfind("? "), self.out.rfind("\n"))
        return self.out[cut + 1:]

    def _decide(self, k: int) -> bool:
        if not 1 <= k <= len(self.node_texts):
            self.removed.append({"n": k, "reason": "no node with this number was supplied"})
            return False
        if not supports(self._sentence(), self.node_texts[k - 1]):
            self.removed.append({"n": k, "reason": "the sentence it closes is not supported by that node's text"})
            return False
        return True

    def feed(self, piece: str) -> str:
        self.buf += piece
        emit = ""
        while self.buf:
            i = self.buf.find("[")
            if i < 0:
                emit += self.buf
                self.out += self.buf
                self.buf = ""
                break
            # release everything before a possible citation (keep trailing spaces with the citation)
            j = i
            while j > 0 and self.buf[j - 1] == " ":
                j -= 1
            head = self.buf[:j]
            emit += head
            self.out += head
            self.buf = self.buf[j:]
            m = self.CITE.match(self.buf)
            if m:
                keep = self._decide(int(m.group(1)))
                cit = m.group(0)
                if keep:
                    emit += cit
                    self.out += cit
                self.buf = self.buf[m.end():]
                continue
            if re.fullmatch(r"\s*\[\d{0,3}", self.buf):
                break              # incomplete: wait for more text
            # a bracket that is not a citation
            emit += self.buf[0]
            self.out += self.buf[0]
            self.buf = self.buf[1:]
        return emit

    def finish(self) -> str:
        rest, self.buf = self.buf, ""
        self.out += rest
        return rest
