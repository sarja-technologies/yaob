"""Stage 1: read Wikipedia's Vital Articles lists into a topic table.

The Vital Articles project is a human-curated, hierarchical list of the most
important encyclopedia topics. Level 4 holds about 10,000 topics; level 5 about
50,000. We read the list pages (with their revision ids) and keep, for every
listed article, the path of section headings above it. That path becomes the
node's place in the MPN hierarchy.

Usage: python -m mpnstudio.corpus.vital --out data/corpus
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .wiki import WikiClient

WP = "en.wikipedia.org"
HEADING = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$")
# A list entry: bullets or numbering, optional icons/templates, then the first wikilink.
ENTRY = re.compile(r"^[*#]+\s*(?:'''?)?(?:\{\{[^}]*\}\}\s*)*(?:'''?)?\s*\[\[([^\]|#]+)(?:\|[^\]]*)?\]\]")
COUNT_SUFFIX = re.compile(r"\s*\(\s*[\d,]+\s*(?:articles?|/[\d,]+)?[^)]*\)\s*$")


def clean_heading(h: str) -> str:
    h = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", h)
    h = re.sub(r"\{\{[^}]*\}\}", "", h)
    h = re.sub(r"<[^>]+>", "", h)
    h = h.replace("'''", "").replace("''", "")
    h = COUNT_SUFFIX.sub("", h)
    return h.strip()


def list_subpages(client: WikiClient, level: int) -> list[str]:
    """All list pages of a level: the root page plus its real (non-redirect) subpages."""
    prefix = f"Vital articles/Level {level}"
    titles = []
    cont = {}
    while True:
        d = client.get(WP, action="query", list="allpages", apnamespace=4, apprefix=prefix,
                       apfilterredir="nonredirects", aplimit=500, **cont)
        titles += [p["title"] for p in d["query"]["allpages"]]
        if "continue" not in d:
            break
        cont = {"apcontinue": d["continue"]["apcontinue"]}
    skip = ("Article alerts", "Removed", "Draft", "draft", "Candidates", "Archive", "Header", "Sandbox", "/doc", "Talk")
    return sorted(t for t in titles if not any(s in t for s in skip))


def parse_list_page(wikitext: str) -> list[dict]:
    """Return [{title, path}] where path is the heading trail above the entry."""
    out = []
    trail: list[tuple[int, str]] = []
    for line in wikitext.splitlines():
        m = HEADING.match(line.strip())
        if m:
            depth = len(m.group(1))
            name = clean_heading(m.group(2))
            trail = [t for t in trail if t[0] < depth]
            if name:
                trail.append((depth, name))
            continue
        m = ENTRY.match(line.strip())
        if m:
            title = m.group(1).strip().replace("_", " ")
            if ":" in title and title.split(":", 1)[0] in ("Wikipedia", "File", "Category", "Template", "Help", "Portal", "WP"):
                continue
            if title:
                title = title[0].upper() + title[1:]
                out.append({"title": title, "path": [t[1] for t in trail]})
    return out


def fetch_level(client: WikiClient, level: int) -> dict:
    pages = []
    for page in list_subpages(client, level):
        d = client.get(WP, action="parse", page=page, prop="wikitext|revid", redirects=1)
        p = d["parse"]
        entries = parse_list_page(p["wikitext"])
        retrieved = client.retrieved_at(WP, action="parse", page=page, prop="wikitext|revid", redirects=1)
        pages.append({"page": p["title"], "revid": p["revid"], "retrieved_at": retrieved, "entries": entries})
        print(f"  {p['title']}: rev {p['revid']}, {len(entries)} entries")
    return {"level": level, "pages": pages}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus")
    ap.add_argument("--levels", default="4,5")
    args = ap.parse_args()
    out = Path(args.out)
    client = WikiClient(out / "cache")
    for level in [int(x) for x in args.levels.split(",")]:
        print(f"level {level}")
        data = fetch_level(client, level)
        n = sum(len(p["entries"]) for p in data["pages"])
        uniq = len({e["title"] for p in data["pages"] for e in p["entries"]})
        print(f"level {level}: {len(data['pages'])} pages, {n} entries, {uniq} unique titles")
        (out / f"vital_level{level}.json").write_text(json.dumps(data, ensure_ascii=False))
    print(f"network calls {client.network_calls}, cache hits {client.cache_hits}")


if __name__ == "__main__":
    main()
