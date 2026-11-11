# Portable formats

Everything MPN Studio exports is a directory of JSON, JSON Lines and safetensors files with a `manifest.json` that lists a SHA-256 and size for every other file. Import verifies those before reading anything else. Importing never runs code from a package.

## Identity

A node's portable identity is its `uid`, a string:

| Form | Meaning |
|---|---|
| `yaob:wd:Q48314` | reference node for the Wikidata item Q48314 |
| `yaob:proc:math.gcd` | procedure node bound to the built-in implementation `math.gcd` |
| `yaob:canary:<seed>:<n>` | diagnostic canary node |
| `yaob:u:<n>` | node created in a store by a user |

Inside one store nodes also have a 64-bit integer id. Ids are local to a store, are always sent as decimal strings over the API, and are not portable; `uid` is.

Importing a package matches nodes by `uid`. A node that is already present gains the membership and is not created a second time.

## MPN package: `yaob-mpn-package/1`

```
<slug>-<version>/
  manifest.json
  nodes.jsonl      one canonical node per line (nodes whose primary membership is this MPN)
  members.jsonl    shared memberships this MPN holds on nodes that live in another MPN
  edges.jsonl      typed relations from this MPN's nodes
  sources.jsonl    every source cited, with licence and revision
  domains.jsonl    the sub-domain tree
  weights/shard.safetensors   optional: this MPN's trained shard
```

`manifest.json` fields: `format`, `slug`, `name`, `kind` (`domain`, `diagnostic`, `user`), `description`, `version`, `class`, `exported_at_rev`, `counts` (`nodes`, `shared_members`, `edges`, `sources`, `domains`, `procedures`), `dependencies` (other MPNs its edges point into; optional to install), `compatibility` (`schemas`, `procedure_implementations`, `weights`: checkpoint, core name and revision), `licences` (source count per licence), `resources` (`record_bytes`, `weight_bytes`, `estimate_resident_bytes`), `evaluation`, `installs_code` (always `false`), `files`.

`class` says what the package contains: `knowledge-only` (no learned state), `knowledge+routing` (routing shard), `knowledge+routing+specialist` (routing shard and specialist adapter), and `+deterministic-tools` when it has procedure nodes.

A node line:

```json
{"uid": "yaob:wd:Q48314", "type": "knowledge", "title": "Battle of Waterloo", "summary": "…", "aliases": ["…"],
 "schema": {"id": "3cn", "version": 1}, "status": "live", "validation": "source_backed", "origin": "reference",
 "cells": {"knowledge": {"statements": [{"text": "…", "source": "12"}], "facts": [{"pid": "P585", "property": "point in time", "value": "18 June 1815", "source": "13"}]},
           "thinking": {"entity_class": ["battle"], "cues": ["…"], "answers": ["point in time", "…"]},
           "reasoning": {"constraints": [{"kind": "…", "text": "…"}], "checks": [{"kind": "fact_consistency", "properties": ["…"]}], "referrals": [{"relation": "part_of", "node": "yaob:wd:Q…", "title": "…"}]}},
 "content_hash": "…", "domain": "modern--europe", "evidence": {"basis": "list section", "path": ["Modern", "Europe"]},
 "sources": [{"source": "12", "locator": "lead section"}]}
```

`source` values refer to `id` in the package's own `sources.jsonl`; they are remapped on import.

**Weights in a package.** `weights/shard.safetensors` holds the MPN's routing keys and gate biases (final, normalised keys, float16); `weights/adapter/` holds its LoRA specialist when it has one. The shard was trained against one checkpoint's shared router; the adapter against one core revision. Installing a package into another model makes its knowledge usable at once (as text, and through text-derived node state after indexing). Its trained state contributes only after a training job in the receiving model. The manifest says which checkpoint and core the shard belongs to.

**Procedures.** A procedure node names an implementation key. Import refuses a package that names a key this build does not ship. There is no mechanism for a package to add code.

## Model bundle: `yaob-bundle/1`

```
yaob-n10k/
  manifest.json
  MODEL_CARD.md
  mpns/<slug>/…          one MPN package per MPN
  schemas.json           cell schemas (3CN, 5CN, custom versions)
  layout.jsonl           {uid, x, y, pinned}
  regions.jsonl          MPN and domain regions of the board
  views.json             saved views
  checkpoint/
    memory.safetensors   all learned modular tensors of the active checkpoint
    nodes.json           uid -> {row, hash, trained}
    mpn_index.json       MPN slug -> index of its prototype and adapter
    manifest.json        training run record (objective, optimiser, data hashes, metrics, parameter counts)
```

`manifest.json` adds: `revision`, exact `counts`, `packages`, `core` (repository, revision, licence, and the SHA-256 of each weight file), `encoder`, `checkpoint`, `requires`.

**Base weights are not in the bundle.** The core and encoder are referenced by repository and revision with file checksums. Fetch them once:

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 --local-dir ~/mpn-studio/models/Qwen3-4B-Instruct-2507
hf download Qwen/Qwen3-Embedding-0.6B --local-dir ~/mpn-studio/models/Qwen3-Embedding-0.6B
```

After that the bundle runs with no network access (`HF_HUB_OFFLINE=1` is set in the service environment).

**A bundle needs MPN Studio's runtime.** The modular network is the store, the learned tensors and the runtime that routes and injects them. A GGUF or safetensors file of the core is only the core.

## Commands

```bash
mpn bundle export yaob-n10k-rev3                 # through the API; written under data/bundles/
mpn bundle export-mpn history                    # one MPN package, under data/packages/
mpn bundle packages                              # packages available to install
mpn bundle install history-0.1.0-rev3
python -m mpnstudio.store.bundle verify  <dir>   # check every file against the manifest
python -m mpnstudio.store.bundle import  <bundle_dir> <new_store.sqlite>
```

## The store itself

`yaob-n10k.sqlite` is the authoritative state. Semantic tables hold immutable versions with `[rev_from, rev_to)` validity; a row with `rev_to` NULL is current. `mpnstudio/store/schema.py` documents every table. It is an implementation, not an interchange format: exchange bundles, not database files.
