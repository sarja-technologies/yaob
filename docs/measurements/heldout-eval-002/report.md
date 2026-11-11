# Held-out evaluation

Suite version 1, seed 20261111, 236 tasks. Evaluated at revision 7.

Accuracy by task family (correct / n). Scoring is deterministic.

| Family | n | deployed | abl_no_adapters |
|---|---|---|---|
| fact_paraphrase | 64 | 62 (97%) | 62 (97%) |
| canary_paraphrase | 40 | 40 (100%) | 40 (100%) |
| canary_statement | 20 | 20 (100%) | 20 (100%) |
| two_hop | 30 | 25 (83%) | 13 (43%) |
| procedure | 38 | 21 (55%) | 21 (55%) |
| unsupported | 20 | 20 (100%) | 20 (100%) |
| adversarial | 24 | 23 (96%) | 24 (100%) |
| **all** | 236 | **211 (89%)** [85%–93%] | **200 (85%)** [80%–89%] |

Routing, cost and speed.

| | deployed | abl_no_adapters |
|---|---|---|
| Replies that named the value but said they did not know (scored wrong) | 4 | 2 |
| Expected node among those read | 0.9861 | 0.9352 |
| Expected MPN consulted | 0.9769 | 0.9769 |
| Share of consulted MPNs that were expected | 0.8302 | 0.8302 |
| Distractor MPN consulted (adversarial tasks) | 0.8333 | 0.8333 |
| MPNs consulted per request | 1.5 | 1.5 |
| Nodes read per request | 7.13 | 7.01 |
| Prompt tokens, median | 822.0 | 813.0 |
| Specialist adapters applied, median | 0.0 | 0.0 |
| of which text evidence | 725.5 | 718.0 |
| Warm time to first token, median ms | 214.2 | 210.6 |
| Warm total, median ms | 1230.9 | 1199.6 |
| Routing time, median ms | 46.37 | 36.86 |
| First request after evicting all shards, total ms | 5268.6 | 746.3 |
| Tokens per second, median | 20.2 | 20.2 |
| Peak torch memory, MiB | 9262.9 | 9179.9 |
| Peak process RSS, MiB | 2511.1 | 2520.4 |
| Peak resident MPN state, MiB | 3.58 | 3.58 |
