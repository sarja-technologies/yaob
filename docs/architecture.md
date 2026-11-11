# Yaob architecture: what is learned, what is retrieved, what is computed

Status: 11 November 2026. This describes the code in `mpnstudio/` as it is, and the two experiments that shaped it.

## Names

- **Fylume** is the lab, **Yaob** the model family, **MPN Studio** the application.
- **MPN**, Modular Parameter Network: one installable module of nodes (History, Physics, …) with its learned state.
- **3CN**, Three Cell Node: a node with Knowledge, Thinking and Reasoning cells. **5CN** adds Provenance and Interface cells. Cell roles are versioned in `store/schemas.py`; relabelling a cell never changes its `runtime_role`.
- **CNP**, cell-node parameters: the learned tensors attached to one node.
- **Yaob N10K**: 10,000 canonical live nodes (knowledge plus procedures). Cells, edges, memberships, diagnostic canaries and deleted records are not counted as nodes.

## One request

```
question ──► frozen encoder (Qwen3-Embedding-0.6B) ──► query vector [1024]
                │
                ▼
  learned router, stage 1: MPN prototypes  ─► which MPNs (only enabled ones; budgeted count)
                │
                ▼  load only those MPNs: routing shard (+ specialist adapter), pinned for this request
  learned router, stage 2: node keys and gates inside the selected MPNs
     + text-derived keys for new/edited nodes (overlay)  + text-search matches inside the selected MPNs
                │
                ▼  resolve against the pinned revision; bounded referral hops (links the question asks about may reach any enabled MPN)
  selected nodes ──► procedures run as code (if a procedure node is selected and its arguments are present)
                │
                ▼
  frozen core (Qwen3-4B-Instruct-2507) with exactly the selected MPNs' LoRA specialists active
     prompt = system + [best nodes as numbered text] + [tool results] + question
                │
                ▼
  answer ──► deterministic checks (numbers supported, tool result stated, citations) ──► evidence record
```

The evidence record (`route`, `validation`, `done` events) lists what was loaded and read. It is a record of access, not of reasoning.

## Components and their parameters

| Component | Learned here? | Shape (N10K) | Parameters | Stored in |
|---|---|---|---|---|
| Core, Qwen3-4B-Instruct-2507 rev `cdbee75f` | no: pretrained, frozen, never modified | 36 layers, hidden 2560 | 4.02 B | `models/` (referenced, not shipped) |
| Encoder, Qwen3-Embedding-0.6B | no: frozen | 1024-d output | 0.6 B | `models/` |
| Router: per-node routing key residual `key_delta` (Thinking cell) | yes | [N, 256] | 256 per node | `router.safetensors`, `shards/mpn_<id>` (final keys) |
| Router: per-node gate bias `gate_bias` (Reasoning cell) | yes | [N] | 1 per node | same |
| Router: `enc_to_key`, `query_to_key` | yes | Linear 1024→256, twice | 524,800 | `shards/shared.safetensors` |
| Router: MPN prototypes `mpn_proto` | yes | [M, 256] | 256 per MPN | same |
| Router: `logit_scale`, `gate_scale`, `gate_center` | yes | scalars | 3 | same |
| Specialist adapter per MPN (LoRA on q,k,v,o,gate,up,down of every layer) | yes, one per MPN | rank 16 | 33,030,144 per MPN (measured in pilot 2; 132 MB float32 on disk) | `adapters/<mpn id>/` |
| Node text (statements, facts, constraints) | no: records | — | — | the store |
| Procedures (192) | no: code | — | — | `runtime/procedures.py` |

With N = 10,000 nodes and M = 18 MPNs the router has about 3.1 M learned parameters, 2.57 M of them per node. Each trained specialist adds about 33 M. Per node, the only learned state of its own is its routing key and gate (257 numbers). **The content of a node's Knowledge cell is not stored in per-node learned state.** It reaches the core in two ways: as text, which is retrieval and is labelled as such, and through its MPN's specialist adapter, which is shared by all of that MPN's nodes.

## What each mode uses

| | core | retrieval | yaob |
|---|---|---|---|
| Learned router | — | — | yes (or the baseline retriever, as an ablation) |
| Specialist adapters | off | off | at most `max_active_adapters` (default 0 since eval-001; the "specialists (experimental)" preset sets 1): the top-routed MPN's, unless a selected node of it was edited since training |
| Node text in the prompt | — | yes (ordinary retrieval) | the best selected nodes (switchable) |
| Procedures | — | — | yes |
| Validation checks | yes | yes | yes |

Ablations used in evaluation (`evals/heldout.py` `ARMS`): `yaob_closed_book` (routing + specialists, no node text), `abl_no_adapters` (routing + text), `abl_router_baseline` (ordinary retrieval chooses nodes, specialists still applied).

## Why this design: the two pilots

### Pilot 1: per-node learned memory tokens (failed)

One learned vector per statement, a shared bridge into the frozen core, and the router. 96 invented canary nodes the core cannot know, 1,200 steps, 200 exposures per node.

| Arm | Trained questions | Held-out phrasings |
|---|---|---|
| Core alone | 0 % | 0 % |
| Node text in the prompt | 100 % | 100 % |
| Learned memory tokens | 5 % | 1.7 % |
| Router recall@4 | 100 % | 100 % |

Gradients reached the memory, the parameters changed and reload reproduced the numbers, so the path was real; it did not carry facts. With far more capacity (20 free soft tokens per node, no bridge, 100 exposures per node, 51,200 parameters per node) it reached 21 % on trained questions, 19 % on paraphrases and 1 % on facts stated only in text. Per-node soft memory through a frozen 4B core is too slow to train for 10,000 nodes. The router was kept.

### Pilot 2: one LoRA specialist per MPN (adopted, with two limits)

Same kind of invented canary network (64 nodes in four MPNs: institute, river, mineral, festival), one rank-16 LoRA adapter per MPN on all seven projections of every layer (33,030,144 parameters, 132 MB on disk in float32), trained only on its MPN's nodes: 280 steps of 16, 20 epochs, about 16 minutes per adapter, torch peak 54 GiB, 590 training tokens/s while sharing the GPU. Questions were scored by whether the reply contains the stored value; 96 questions per set. Full report: `docs/measurements/pilot-adapters-001.json`.

| Arm (closed book unless stated) | Trained phrasing | Held-out phrasing | Fact only in a statement, never a training answer |
|---|---|---|---|
| Core alone | 0 % | 0 % | 0 % |
| Node text in the prompt, no adapter | 100 % | 100 % | 100 % |
| Own MPN's adapter | 100 % | 100 % | 82 % |
| Another MPN's adapter | 2 % | 1 % | 0 % |
| All four adapters at once | 19 % | 24 % | 14 % |
| Own adapter + node text | 100 % | 100 % | 97 % |
| Own adapter after save, delete and reload from disk | 100 % | 100 % | 82 % (identical) |
| Core alone after all training (core unchanged) | — | 0 % | — |

Edit test: one institute's founding year was changed from 1800 to 1937. The stale adapter answered 1800 closed book **and still answered 1800 with the corrected text in the prompt**. Retraining that MPN's adapter on the edited node with replay took 119 s; afterwards it answered 1937 and kept 97 % on the MPN's other nodes.

What follows from it, and is now built into the runtime:

1. **Specialists do carry knowledge**, MPN-specifically, including facts they only saw as statement text. This is the learned path the project keeps.
2. **Specialists must not be stacked.** Four active at once lost most of what each knew. The runtime applies at most `max_active_adapters` specialists per request (the top-routed MPN's; default 0 since the held-out evaluation below); other selected MPNs contribute routed text only.
3. **A stale specialist overrides corrected text.** When a selected node in an MPN has been edited since that MPN's specialist was trained, the runtime leaves the specialist off for the request and says so; the node's MPN is marked for retraining.
4. With node text present, the specialist adds nothing on these canaries (text alone is already 100 %) and slightly lowers the statement-only facts (97 % vs 100 %). Its value is closed-book recall when retrieval misses or the context budget is small. Whether that shows up on real questions is for the held-out evaluation.

### Held-out evaluation eval-001 (12 November 2026): specialists off by default

236 held-out questions per arm, checkpoint `ck-261111-200127-6f01ff` (router + astronomy, chemistry, history and canary specialists), context 4,096, answer 128 tokens; re-scored once after a refusal-detection fix (`docs/measurements/heldout-eval-001/`). Learned routing + node text without a specialist: 85 % [80–89]; with one specialist: 81 % [75–85]; plain retrieval: 82 % [76–86]; core alone 44 %. With a specialist the model declined 50 % of unanswerable questions, without one 100 %. Closed book, specialists did not beat the core on held-out real-knowledge questions (95 vs 103 of 176). The router put the expected node in front of the model 93.5 % of the time vs 72.2 % for plain retrieval. So the runtime's default is routing + node text with `max_active_adapters = 0`; specialists stay available as an experimental preset. These numbers describe that configuration and protocol, not the deployed system after later changes.

Not measured validly: the pilot's two-node "composition" questions. Their scorer counts a reply as correct if it names the right entity, so the core alone scored 100 % while replying that it had no information about either. Those numbers are excluded.

## Edits, revisions and learned state

- Every edit is a revision. A request pins one revision; a job pins one revision for its whole run.
- Router rows (`bank_rows`) are append-only: a node keeps its row for life, deletion keeps the row, and a job never sees rows move under it.
- Each node's learned state is valid only for the content hash it was trained on (`nodes.json` in the checkpoint). An edited node is routed with a key derived from its new text (overlay) until retrained, and its MPN's specialist still reflects the old content until that MPN is retrained. The inspector says so; the board never claims the weights already changed.
- Moving a node on the board changes geometry only: no index, key or adapter is affected.
- Disabling or uninstalling an MPN removes its routing shard and adapter from the runtime's reach. It does not remove what the core learned in pretraining. Module removal is not machine unlearning.

## What is and is not new here

Routing with learned keys over a frozen encoder, LoRA adapters and retrieval-augmented prompting are all established techniques. What MPN Studio adds is their arrangement as an editable, versioned, budgeted network of nodes and modules: per-module adapters loaded only when routed to, shared nodes with one identity and several memberships, and a record of access per request. Whether this arrangement answers better than ordinary retrieval is a question for the held-out evaluation, not a claim made here.
