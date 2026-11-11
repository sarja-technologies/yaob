# Yaob N10K corpus

Built 11 November 2026 by `scripts/corpus_chain.sh` (network stages) and `scripts/corpus_finish.sh` (build, load, canary, layout). Every stage is cached on disk (`data/corpus/cache`, one file per API request with its retrieval time), so the build is reproducible from the cache without the network. Report: `data/corpus/build_report.json`.

## What a node is here

**10,000 canonical live nodes**: 9,808 knowledge nodes and 192 procedure nodes. Edges, cells, memberships, sources and the 160 diagnostic canary nodes are not counted. The store reports the canaries separately (`diagnostic_nodes`), keeps their MPN disabled, and enables it only at a pinned revision during evaluation.

- **Knowledge node**: one Wikipedia article chosen from the Vital articles lists (levels 4 and 5). Its Knowledge cell holds up to six lead sentences as statements, each citing the article at a fixed revision. It also holds the item's Wikidata claims as facts (a curated property set, `corpus/fetch.py` `PROPS`), each citing the Wikidata item. The Thinking cell holds entity class, aliases and cues; the routing key is trained into it. The Reasoning cell holds validity notes (dissolved, died, "as of" values), "not to be confused with" links and typed referrals to related nodes.
- **Procedure node**: one function in `runtime/procedures.py`, with its formula, parameters and a worked example that the test suite re-runs. Procedures are code, not learned.

## Sources and licences

| Source | Used for | Licence |
|---|---|---|
| English Wikipedia, lead sections at a recorded revision (`action=query&prop=extracts`) | statements, links, hatnotes | CC BY-SA 4.0. Attribution per node: article title, revision link, "Wikipedia contributors" |
| Wikidata (`wbgetentities`) | facts, relations, aliases, descriptions | CC0 1.0 |
| Wikipedia "Vital articles" lists, level 4 and 5, at a recorded revision | topic selection and sub-domains | CC BY-SA 4.0 |
| This repository | procedures | project source |

19,649 source records. Requests were anonymous (no contact address is sent unless `MPN_CORPUS_CONTACT` is set), one at a time, at least 6.4 s apart, and honoured `Retry-After`. They sent `maxlag=5`. While Wikidata reported sustained replication lag, reads went out without it for ten minutes at a time: `maxlag` exists to slow writers, and these were single reads at the minimum interval. Text from these sources is stored as data and never executed or followed as instructions.

## Per MPN

| MPN | Target | Knowledge | Procedures | Of the knowledge, from level 5 |
|---|---|---|---|---|
| Language and linguistics | 380 | 372 | 8 | 171 |
| Mathematics and statistics | 560 | 518 | 42 | 176 |
| Computing and AI | 520 | 507 | 13 | 412 |
| Physics | 520 | 489 | 31 | 75 |
| Chemistry | 460 | 452 | 8 | 163 |
| Biology | 760 | 753 | 7 | 0 |
| Earth, environment and climate | 460 | 455 | 5 | 191 |
| Astronomy and space | 400 | 392 | 8 | 163 |
| Medicine and health | 560 | 552 | 8 | 237 |
| Engineering and technology | 680 | 670 | 10 | 52 |
| History | 980 | 977 | 3 | 0 |
| Geography and geopolitics | 900 | 896 | 4 | 0 |
| Economics and business | 420 | 406 | 14 | 248 |
| Civics, law and government | 440 | 436 | 4 | 251 |
| Philosophy, logic and religion | 520 | 517 | 3 | 0 |
| Psychology and society | 460 | 459 | 1 | 31 |
| Arts, literature and music | 640 | 634 | 6 | 0 |
| Everyday life | 340 | 323 | 17 | 0 |

Every MPN meets its target exactly. No topic came from the reserve list.

## Links and memberships

105,857 edges. By type: mentions 79,113, subclass_of 2,339, country 2,314, instance_of 2,179, citizen_of 1,705, borders 1,509, has_part 1,392, field_of_work 1,364, part_of 1,165, religion 1,122, different_from 1,118, influenced_by 928, official_language 779, named_after 629, … "mentions" edges come from links in the lead; all others come from Wikidata relations between two nodes that are both in the network.

2,000 nodes belong to more than one MPN. A node's second (and third) membership is added when a procedure declares it, or when at least 4 of the node's links, and at least 30 % of them, go into that MPN. The node keeps one identity and its membership evidence is recorded. Most frequent pairs (primary → shared):

| Primary | Shared | Nodes |
|---|---|---|
| history | geography-geopolitics | 269 |
| language-linguistics | geography-geopolitics | 136 |
| geography-geopolitics | history | 83 |
| arts-literature-music | geography-geopolitics | 65 |
| psychology-society | geography-geopolitics | 58 |
| medicine-health | biology | 51 |
| engineering-technology | physics | 51 |
| chemistry | physics | 50 |

111 nodes have no edge to any other node in the network.

## What was dropped, and why

- **Lead sentences** the plaintext extract had mangled, counted by reason: 73 with a flattened exponent ("2×1030 kg" for 2×10^30 kg, which would store a wrong number as a sourced statement), 146 flattened formulas, 222 lead-ins whose formula or list was removed, 38 fragments after a removed formula, and 322 under 25 characters. Some symbol-heavy sentences remain (for example "The nth root of x is written as xn using the radical symbol x.").
- **Facts**: claims with no value, deprecated claims, entity values whose label was not looked up: only values that recur (at least twice across the corpus), and at most the 6,000 most frequent, were looked up, at about 6 s per 50 labels at the anonymous rate, and repeated claims. Two claims with the same value but different validity periods are both kept, and the period is shown and given to the core ("London (1621 to 1638)").
- **Topics**: none in the final build. The first build lost 9 major topics (Kazakhstan, Nigeria, Thailand, Bangladesh, Serbia, Iceland, Romania, South Korea, Moscow) to a silently truncated Wikidata response. That fetch defect is fixed and tested (`tests/test_corpus_fetch.py`).

## Known limitations

- Statements are lead sentences, not the whole article. A question about a detail outside the lead has no node text to retrieve.
- Wikidata and the Wikipedia lead sometimes disagree. For example, Gavrilo Princip's date of birth is 13 July 1894 in Wikidata and 25 July 1894 in the lead (calendar difference). Both are kept, each with its own source, and nothing reconciles them.
- Facts reflect Wikidata on 11 November 2026. Values with an "as of" date are marked as possibly outdated.
- The selection follows Wikipedia's Vital articles, with their coverage biases.
