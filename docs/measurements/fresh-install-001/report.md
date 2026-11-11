# Held-out evaluation

Suite version 1, seed 20261111, 236 tasks. Evaluated at revision 40.

Accuracy by task family (correct / n). Scoring is deterministic.

| Family | n | deployed |
|---|---|---|
| fact_paraphrase | 64 | 62 (97%) |
| canary_paraphrase | 40 | 40 (100%) |
| canary_statement | 20 | 20 (100%) |
| two_hop | 30 | 25 (83%) |
| procedure | 38 | 21 (55%) |
| unsupported | 20 | 20 (100%) |
| adversarial | 24 | 23 (96%) |
| **all** | 236 | **211 (89%)** [85%–93%] |

Routing, cost and speed.

| | deployed |
|---|---|
| Replies that named the value but said they did not know (scored wrong) | 4 |
| Expected node among those read | 0.9861 |
| Expected MPN consulted | 0.9769 |
| Share of consulted MPNs that were expected | 0.8302 |
| Distractor MPN consulted (adversarial tasks) | 0.8333 |
| MPNs consulted per request | 1.5 |
| Nodes read per request | 7.13 |
| Prompt tokens, median | 822.0 |
| Specialist adapters applied, median | 0.0 |
| of which text evidence | 725.5 |
| Warm time to first token, median ms | 211.9 |
| Warm total, median ms | 1210.7 |
| Routing time, median ms | 48.79 |
| First request after evicting all shards, total ms | 5145.3 |
| Tokens per second, median | 20.7 |
| Peak torch memory, MiB | 9262.9 |
| Peak process RSS, MiB | 2537.8 |
| Peak resident MPN state, MiB | 3.58 |
