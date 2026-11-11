# Choosing the shared cognitive core

Decision date: 11 November 2026. Sources retrieved the same day.

**Default core: `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`, bf16 safetensors, Apache-2.0.**

It is a pretrained model used as is. Nothing in this project trains a 4B model from scratch, and the core keeps whatever its pretraining gave it: it is not a knowledge-free reasoning engine.

## Why this one

Three candidates were run locally. The decision rests on one hard requirement and two measurements.

1. **The learned modular path needs a core that accepts injected embeddings and passes gradients back to them while frozen.** Qwen3-4B-Instruct-2507 and Qwen3.5-4B do. Gemma 4 E4B, through the stock `transformers` interface, does not: it derives per-layer embeddings from token ids and rejects arbitrary `inputs_embeds` (error quoted below). A workaround through `per_layer_inputs` may exist; it was not tested, so Gemma is not claimed to be unusable, only unproven for this architecture.
2. Of the two candidates that meet that requirement, Qwen3-4B-Instruct-2507 scored higher on the local suite (84 vs 80 of 94) and was better at the task closest to routing (9 vs 6 of 12).
3. It needs about half the memory of Gemma 4 E4B (7.5 vs 14.8 GiB resident) and loads in a third of the time.

Gemma 4 E4B had the best quality score on the suite (90 of 94). If its injection path is proven later, it is the first candidate to re-evaluate. With only one suite of 94 tasks, a six-task gap is suggestive, not decisive.

## Candidates considered

| Model | Revision | Licence | Access | Learned weights | Notes |
|---|---|---|---|---|---|
| Qwen/Qwen3-4B-Instruct-2507 | `cdbee75f` | Apache-2.0 | open | 4.02 B | text only, non-thinking; the known 4B baseline |
| Qwen/Qwen3.5-4B | `851bf6e8` | Apache-2.0 | open | 4.54 B loaded (4.21 B text stack) | vision-language; thinking on by default |
| google/gemma-4-E4B-it | `4ecc2ba6` | Apache-2.0 | open | 7.94 B loaded (7.46 B text stack; "4.5 B effective") | text, image, audio; per-layer embeddings |
| Qwen/Qwen3.5-9B | `c2022362` | Apache-2.0 | open, already on the Spark | 9.65 B | above the 4–8B range; not benchmarked |
| google/gemma-3-4b-it | `093f9f38` | Gemma terms | gated (manual approval) | 4.30 B | set aside: gated and superseded by Gemma 4 |
| mistralai/Ministral-8B-Instruct-2410 | `2f494a19` | Mistral Research Licence | open | 8.02 B | set aside: research-only licence |
| microsoft/Phi-4-mini-instruct | `cfbefacb` | MIT | open | 3.84 B | not benchmarked; older than the three run |
| ibm-granite/granite-4.0-h-tiny | `791e0d3d` | Apache-2.0 | open | 6.94 B | not benchmarked |

Licence and access come from each repository's metadata on huggingface.co (API fields `cardData.license`, `gated`). Weight counts for the three benchmarked models are counted from the loaded model; the others are the Hub's safetensors totals.

## Local benchmark

Same machine, same 94 prompts, same system prompt, greedy decoding, 256 new tokens at most, bf16, `transformers 5.22.2`, `torch 2.15.0+cu132`, thinking disabled wherever the chat template offers the switch. Every task is scored by a deterministic checker (`mpnstudio/evals/core_suite.py`); no model judges another. Raw replies are in `runs/core-bench/*.json` on the Spark.

| | Qwen3-4B-Instruct-2507 | Qwen3.5-4B | Gemma 4 E4B |
|---|---|---|---|
| **Total (94)** | **84** | **80** | **90** |
| Instruction following (10) | 8 | 8 | 10 |
| Structured output (10) | 10 | 10 | 10 |
| Tool calls (8) | 8 | 8 | 8 |
| Routing compliance (12) | 9 | 6 | 10 |
| Mathematics (12) | 11 | 10 | 10 |
| Reasoning (10) | 7 | 8 | 10 |
| Coding (8) | 8 | 7 | 8 |
| Cross-domain synthesis (8) | 8 | 7 | 8 |
| Grounding (8) | 7 | 8 | 8 |
| Abstention (8) | 8 | 8 | 8 |
| JSON wrapped in a code fence despite "JSON only" | 0 | 0 | 5 |
| Cold load | 53 s | 66 s | 154 s |
| Warm time to first token, median / p95 | 0.16 / 0.37 s | 0.31 / 0.87 s | 0.25 / 1.02 s |
| Tokens per second, median | 19.8 | 10.7 | 16.9 |
| Resident after load (torch / NVML) | 7.49 GiB / 7.8 GiB | 8.46 / 8.9 | 14.79 / 15.3 |
| Peak during suite (torch) | 7.60 GiB | 8.67 | 14.87 |
| 5,600-token prompt: extra memory, time to first token | 1.19 GiB, 1.2 s | 1.49 GiB, 4.8 s | 1.03 GiB, 2.2 s |
| Accepts injected embeddings, gradient reaches them | yes | yes | **no (stock interface)** |

Reading the table:

- **Speed rows are not a clean ranking.** The Spark was shared. A 28 GiB media job ran at 94–96 % GPU utilisation during the whole Qwen3.5-4B run and the start of the Gemma run, and began partway through the Qwen3-4B run. The quality and memory rows do not depend on that load; the speed rows do.
- Tokens per second here is the Python `generate()` loop in bf16, which is the backend the modular path needs. It is slower than the llama.cpp GGUF path the same machine uses for plain chat and should not be compared with it.
- Two checker corrections were made after the first run and applied to the stored replies of all three models without re-running them: Qwen3.5's XML tool-call syntax is accepted, and JSON inside a Markdown fence is accepted (and counted separately, since the prompt asked for JSON only).
- Qwen3.5-4B's published results use its thinking mode. Here thinking was off for every model so that decoding rules matched, which likely understates it.

Gemma 4 E4B's injection error, verbatim:

> RuntimeError: It seems like you tried to call `forward` from `inputs_embeds` without providing `input_ids`, and that the `inputs_embeds` you provided do not exactly match the embedding weights. Since Gemma4 needs to reverse the embedding to compute another embedding, make sure you provide exact `inputs_embeds`

## Published results (not comparable with the table above, or with each other)

Different suites, dates, precision and decoding. Listed so the local numbers can be placed in context, not ranked together.

- Qwen3-4B-Instruct-2507 (model card): MMLU-Pro 69.6, GPQA 62.0, IFEval 83.4, LiveCodeBench v6 35.1, BFCL-v3 61.9, ZebraLogic 80.2.
- Qwen3.5-4B (model card, thinking mode): MMLU-Pro 79.1, GPQA Diamond 76.2, IFEval 89.8, LiveCodeBench v6 55.8, BFCL-V4 50.3.
- Gemma 4 E4B (Google model card): MMLU Pro 69.4 %, GPQA Diamond 58.6 %, LiveCodeBench v6 52.0 %, AIME 2026 42.5 %.

## Capability contract: how to replace the core

A core is described by `CoreContract` (`mpnstudio/model/core.py`) and recorded in every checkpoint and bundle manifest: repository, revision, licence, architecture, hidden size, weight count, precision, context limit, and two capabilities: accepts injected embeddings, and passes gradients to them while frozen.

To adopt another core:

1. Download it at a pinned revision into `~/mpn-studio/models/<name>` and add a `CORE.json` with repository, revision and licence.
2. Run `scripts/core_bench.py` on it. The `injection` block must report `grad_reaches_soft_prompt: true` and `generate_from_embeds: true`.
3. Retrain the specialist adapters. They are tied to the core's layer shapes and weights: a checkpoint trained for one core is refused by the runtime for another (the manifest's core revision must match). The router depends only on the frozen encoder and carries over.

The router, node records, schemas, MPN packages' knowledge content and the board do not depend on the core.

## Sources

- https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 (card and API metadata)
- https://huggingface.co/Qwen/Qwen3.5-4B (card and API metadata)
- https://huggingface.co/google/gemma-4-E4B-it (API metadata), https://ai.google.dev/gemma/docs/core/model_card_4 (model card)
- https://huggingface.co/api/models/<id> for licence, gating, revision and weight counts of every model in the candidate table
- https://docs.nvidia.com/dgx/dgx-spark/system-overview.html (machine)

## Core provenance file

The runtime and training jobs read `CORE.json` next to the core's weights (`~/mpn-studio/models/Qwen3-4B-Instruct-2507/CORE.json` on the Spark) for the repository, revision and licence they record in every checkpoint and adapter. Without it those fields are empty. On the Spark it was written on 11 Nov 2026 from the download record (`.cache/huggingface/download/config.json.metadata`): repository `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`, licence Apache-2.0. The encoder, Qwen3-Embedding-0.6B, is revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` (Apache-2.0).
