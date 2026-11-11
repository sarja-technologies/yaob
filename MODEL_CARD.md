# Model card: Yaob N10K 0.1.0

| | |
|---|---|
| Developer | Fylume (https://fylu.me) |
| Release | 0.1.0, 12 November 2026. Research release |
| Type | Modular retrieval-and-routing system around a frozen language model |
| Core | Qwen3-4B-Instruct-2507, revision `cdbee75f`, frozen and unmodified (Apache-2.0) |
| Encoder | Qwen3-Embedding-0.6B, revision `97b0c614`, frozen and unmodified (Apache-2.0) |
| Learned by Fylume | router (about 3.1 M parameters). Four experimental LoRA specialist adapters (33 M each: astronomy, chemistry, history, diagnostic canary), off by default |
| Network | 10,000 nodes (9,808 knowledge, 192 procedures), 105,857 typed relations, 18 domain MPNs, plus 160 invented diagnostic canary facts in a disabled MPN |
| Language | English |
| Terms | code and weights: Fylume source-available terms (no resale), not open source. Node data: CC BY-SA 4.0 |

## How it answers

1. The frozen encoder embeds the question. For a follow-up such as "And who was its last sultan?", up to two earlier user turns are also embedded.
2. The learned router selects up to three enabled MPNs, then up to eight nodes inside them.
3. Links the question asks about are followed one step. For example, for "the place of birth of X", the `born_in` link is followed, and its target may come from another enabled MPN.
4. The best nodes are placed in the prompt as numbered text. Procedures are run as code.
5. The frozen core writes the answer. A citation is removed while the answer streams if it points to a node that was not supplied, or to one that does not support the sentence.
6. Every reply carries an evidence record that lists the MPNs consulted and the nodes read. It records what was accessed. It does not explain the model's reasoning.

**Most of the accuracy is retrieval.** Node content reaches the core as text. The router decides which text, and that is the main learned part. Per node, the learned state is a 256-number routing key and a gate. It does not hold the node's facts.

## Results on the held-out suite

The suite has 236 tasks, seed 20261111, and was built before evaluation. It is published in `docs/measurements/heldout-eval-001/suite.json` and has not been used for tuning. Settings were chosen on separate development sets (`docs/measurements/multihop-dev-002`), then measured once here. Scoring is deterministic. Intervals are Wilson 95 %. The runs used a DGX Spark, a 128-token answer cap and an 8,192-token context.

| Configuration | All 236 | Facts (64) | Invented canary facts (60) | Two-step (30) | Procedures (38) | Unanswerable (20) | Distractor prefix (24) |
|---|---|---|---|---|---|---|---|
| **Default 0.1.0** (eval-002): routing, node text, relation referrals, no specialist | **89 %** (85–93) | 97 % | 100 % | 83 % | 55 % | 100 % | 96 % |
| eval-001 default: the same with the earlier referral rule | 85 % (80–89) | 97 % | 100 % | 47 % | 55 % | 100 % | 100 % |
| One specialist adapter and node text (eval-001) | 81 % (75–85) | 98 % | 100 % | 40 % | 58 % | 50 % | 100 % |
| Specialist adapter, no node text (eval-001) | 43 % (37–50) | 64 % | 12 % | 20 % | 74 % | 45 % | 46 % |
| Plain retrieval, same core (eval-001) | 82 % (76–86) | 94 % | 100 % | 43 % | 45 % | 100 % | 96 % |
| Core model alone (eval-001) | 44 % (38–50) | 69 % | 0 % | 23 % | 50 % | 100 % | 54 % |

The eval-001 default was re-run under the 0.1.0 code in eval-002 and scored 200 against its original 201 (two-step 13 against 14). Every row used the same released checkpoint and node data.

The canary facts are invented, so the core cannot know them. They show that routing and text supply work. They do not show that facts are stored in weights. Closed-book recall of canaries was 7 of 60 (12 %).

## Cost and speed (eval-002, default)

| | |
|---|---|
| GPU memory, peak | 9.3 GiB (core 7.6 GiB in bfloat16, encoder, cache) |
| Process memory, peak | 2.5 GiB |
| Prompt, median | 822 tokens, of which 726 are node text |
| Time to first token, median | 214 ms |
| Generation | about 20 tokens per second |
| Routing, median | 46 ms |

The machine was shared with other GPU work during all runs.

## Specialist adapters (experimental)

Each adapter is a rank-16 LoRA on every layer of the core, trained on its MPN's node text. In a closed-book check (no node text), the specialist and the core alone scored:

| MPN | With specialist | Core alone |
|---|---|---|
| Astronomy | 39.6 % | 24.0 % |
| Chemistry | 55.2 % | 41.7 % |
| History | 58.3 % | 46.9 % |

With an adapter active, Yaob declined only 50 % of questions about invented entities. Without one, it declined 100 %. Adapters are therefore off by default. They are in the `-specialists-experimental` bundle, and the `specialists (experimental)` budget preset turns one on.

## Limitations

- **Coverage.** The network holds 10,000 topics. When a question needs a fact that no node holds, the core may answer from a nearby node instead of declining. For example, it named Abdul Hamid II as the last Ottoman sultan; no node covers Mehmed VI. Treat answers without a matching citation as unsupported.
- **Two-step questions** still fail when routing misses the second node or the core does not combine the two texts: 5 of 30 in eval-002.
- **Procedures** (unit conversion, statistics, voting methods, …) were answered correctly 55 % of the time.
- **Distractor prefixes.** In 83 % of the distractor tasks, an MPN named only in the distractor was also consulted. The answer was still right in 96 %.
- **Small suite.** 236 tasks. Differences of a few points between configurations are within the intervals.
- **Removing an MPN is not unlearning.** Disabling or uninstalling an MPN removes its nodes and adapter from requests. The frozen core keeps whatever it learned in pretraining.
- **Source text.** Node statements are excerpts from English Wikipedia lead sections and Wikidata facts as of November 2026, with their errors and gaps.

## Intended use

Research into modular, inspectable knowledge for small language models. It is suited to local question answering over the included topics, where every claim can be traced to a node. Do not use it for medical, legal or financial decisions, or as a source of record.

## Data

- Node text comes from English Wikipedia (CC BY-SA 4.0). Each statement cites its article and revision.
- Facts and relations come from Wikidata (CC0).
- Procedures and the 160 canary facts were written by Fylume.

Selection and processing are described in `docs/corpus.md`.

## Reproduce

```bash
python -m mpnstudio.evals.heldout run --suite docs/measurements/heldout-eval-001/suite.json --out runs/eval --arms deployed,abl_no_adapters,core,retrieval
```

The rows, replies and routes of every run are in `docs/measurements/`.
