# Held-out evaluation

Suite version 1, seed 20261111, 236 tasks. Evaluated at revision 3.

Accuracy by task family (correct / n). Scoring is deterministic.

| Family | n | core | retrieval | yaob | yaob_closed_book | abl_no_adapters | abl_router_baseline |
|---|---|---|---|---|---|---|---|
| fact_paraphrase | 64 | 45 (70%) | 62 (97%) | 63 (98%) | 41 (64%) | 62 (97%) | 63 (98%) |
| canary_paraphrase | 40 | 0 (0%) | 40 (100%) | 40 (100%) | 6 (15%) | 40 (100%) | 40 (100%) |
| canary_statement | 20 | 0 (0%) | 20 (100%) | 20 (100%) | 1 (5%) | 20 (100%) | 20 (100%) |
| two_hop | 30 | 7 (23%) | 13 (43%) | 12 (40%) | 6 (20%) | 14 (47%) | 12 (40%) |
| procedure | 38 | 19 (50%) | 17 (45%) | 22 (58%) | 28 (74%) | 21 (55%) | 19 (50%) |
| unsupported | 20 | 20 (100%) | 9 (45%) | 7 (35%) | 9 (45%) | 10 (50%) | 2 (10%) |
| adversarial | 24 | 13 (54%) | 23 (96%) | 24 (100%) | 11 (46%) | 24 (100%) | 23 (96%) |
| **all** | 236 | **104 (44%)** [38%–50%] | **184 (78%)** [72%–83%] | **188 (80%)** [74%–84%] | **102 (43%)** [37%–50%] | **191 (81%)** [75%–85%] | **179 (76%)** [70%–81%] |

Routing, cost and speed.

| | core | retrieval | yaob | yaob_closed_book | abl_no_adapters | abl_router_baseline |
|---|---|---|---|---|---|---|
| Replies that named the value but said they did not know (scored wrong) | 3 | 0 | 1 | 1 | 1 | 0 |
| Expected node among those read | 0.0 | 0.7222 | 0.9352 | 0.9352 | 0.9352 | 0.7315 |
| Expected MPN consulted | – | 0.0 | 0.9769 | 0.9769 | 0.9769 | 0.9722 |
| Share of consulted MPNs that were expected | – | – | 0.8302 | 0.8302 | 0.8302 | 0.5988 |
| Distractor MPN consulted (adversarial tasks) | 0.0 | 0.0 | 0.8333 | 0.8333 | 0.8333 | 0.7917 |
| MPNs consulted per request | 0.0 | 0.0 | 1.5 | 1.5 | 1.5 | 2.22 |
| Nodes read per request | 0.0 | 5.0 | 7.01 | 7.01 | 7.01 | 8.0 |
| Prompt tokens, median | 55.0 | 818.5 | 813.0 | 86.5 | 813.0 | 818.5 |
| Specialist adapters applied, median | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| of which text evidence | 0.0 | 720.0 | 718.0 | 0.0 | 718.0 | 720.0 |
| Warm time to first token, median ms | 123.7 | 234.9 | 276.8 | 176.9 | 240.5 | 296.9 |
| Warm total, median ms | 1672.5 | 1241.7 | 1077.5 | 1000.8 | 1233.6 | 1061.1 |
| Routing time, median ms | 0.0 | 35.64 | 44.66 | 43.59 | 38.06 | 35.97 |
| First request after evicting all shards, total ms | 2192.7 | 830.7 | 773.8 | 2966.6 | 771.8 | 772.8 |
| Tokens per second, median | 20.2 | 20.1 | 20.1 | 20.5 | 20.2 | 20.2 |
| Peak torch memory, MiB | 9127.0 | 9427.6 | 9532.6 | 9354.8 | 9365.0 | 9405.5 |
| Peak process RSS, MiB | 2466.7 | 2488.4 | 2501.9 | 2499.7 | 2497.2 | 2499.6 |
| Peak resident MPN state, MiB | 126.6 | 0.0 | 381.79 | 381.79 | 129.01 | 255.35 |
