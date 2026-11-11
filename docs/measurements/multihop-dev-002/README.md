# Development comparison of referral settings (12 Nov 2026)

These are development sets only. Questions come from the held-out generator under seed 777001. Any task that shares a node with the locked held-out suite was dropped. The setting chosen here was then measured once on the held-out suite (eval-002).

Every request was routed with node text and no specialist adapter, with 128 output tokens. Each folder holds `report.json` (rates and the first failing stage per task) and `rows.jsonl` (every question, route and reply).

## Two-step questions (60)

Each question has the form "What is the \<property\> of the \<relation\> of \<A\>?".

| Variant | Setting | Second node selected | Both nodes given as text | Answered correctly |
|---|---|---|---|---|
| `plain` | the referral rule used until 12 Nov 2026 | 41 | 17 | 25 (42 %) |
| `focus` | plain, and a node's facts that the question names are shown first | 41 | 17 | 26 (43 %) |
| `relation` | focus, and links whose relation the question names are followed first, may pass the selection cap, and their targets are given as text (up to 2) | 42 | 36 | 37 (62 %) |
| `relation-enabled` | relation, and those asked-for links may also enter an enabled MPN that the router did not activate | 59 | 53 | **47 (78 %)** |

First failing stage for `relation-enabled`: one routing miss, four cases where packing left a text out, one case where rendering cut the fact, and seven cases where the core did not combine the two texts.

## Single-fact questions (80), `../dev-facts-002`

| Variant | Correct |
|---|---|
| plain | 75 |
| relation-enabled (the code of `relation-enabled` above) | 75 |

The two variants never disagreed.

## Decision

`referral_mode: "relation"` with `fact_focus: true` became the default. In the runtime, relation mode includes the enabled-MPN referrals. `"plain"` remains selectable in the budget.
