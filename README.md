# Yaob N10K and MPN Studio

Yaob N10K is a modular knowledge network by [Fylume](https://fylu.me). It has 10,000 canonical nodes (9,808 knowledge nodes and 192 procedures) in 18 installable modules called MPNs (History, Chemistry, Physics, …). A learned router chooses which MPNs and nodes a question needs, and a frozen language model, Qwen3-4B-Instruct-2507, writes the answer from them. MPN Studio is the editor and runtime: a whiteboard for the whole network, a node inspector, schema editor, training and evaluation jobs, and an OpenAI-compatible chat endpoint.

This is research software. Read [MODEL_CARD.md](MODEL_CARD.md) before relying on it.

## What is learned and what is retrieved

Most of Yaob's accuracy comes from retrieval. In the default setup, the router picks nodes and their text is placed in the prompt. The router is learned, and node text is supplied as text, not stored in weights.

| Part | Learned by Fylume | Size |
|---|---|---|
| Router: a 256-number routing key and a gate per node, MPN prototypes, two projections | yes | about 3.1 M parameters |
| Specialist LoRA adapters, one per MPN (astronomy, chemistry, history and the diagnostic canary MPN) | yes, experimental, off by default | 33 M parameters each |
| Node text: statements, facts and relations from Wikipedia and Wikidata | no, records | 10,000 nodes, 105,857 relations |
| Procedures (unit conversion, arithmetic, dates, …) | no, code | 192 |
| Core model Qwen3-4B-Instruct-2507 and encoder Qwen3-Embedding-0.6B | no, frozen and unmodified | 4.0 B and 0.6 B |

Disabling or uninstalling an MPN stops the router from reading it and removes its text and adapter from requests. This is not machine unlearning. The frozen core still knows whatever it learned in pretraining.

## Measured results

The held-out suite has 236 tasks (seed 20261111). It was built before evaluation and has not been used for tuning. Full reports are in [docs/measurements](docs/measurements).

| Configuration | Correct | 95 % interval |
|---|---|---|
| Core model alone | 44 % | 38–50 % |
| Plain retrieval (encoder similarity and text search, node text) | 82 % | 76–86 % |
| **Default 0.1.0**: learned routing, node text, relation-aware referrals, no specialist (eval-002) | **89 %** | 85–93 % |
| The same with the earlier referral rule (eval-001) | 85 % | 80–89 % |
| The same with one specialist adapter | 81 % | 75–85 % |
| Specialist adapter, no node text ("closed book") | 43 % | 37–50 % |

Known weaknesses:

- **Gaps in coverage.** When no node holds the fact a question needs, the core may answer from a nearby node instead of declining. Check that a citation supports the answer.
- **Two-step questions.** An example is "What is the population of the capital of Peru?". The default answered 25 of 30, against 13 to 14 with the earlier referral rule.
- **Specialist adapters make refusals worse.** With an adapter active, Yaob declined only half of the questions about invented entities. Without one, it declined all of them. This is why specialists are off by default.
- **Closed-book recall is weak.** Without node text, a specialist adapter adds 11 to 16 points over the core on its own MPN's questions (for example chemistry 55 % versus 42 %). That is far below retrieval.

## Downloads

Release assets are on the [releases page](https://github.com/sarja-technologies/yaob/releases). `scripts/fetch_release.sh` downloads them, checks every SHA-256 and reassembles split files.

| Asset | Contents | Terms |
|---|---|---|
| `yaob-n10k-<version>.tar.gz` | node data, layout and the learned router (the default setup) | data CC BY-SA 4.0, router weights Fylume source-available |
| `yaob-n10k-<version>-specialists-experimental.tar.gz` | the same, plus the four experimental specialist adapters | as above |
| `mpn-studio-web-<version>.tar.gz` | the built web interface | Fylume source-available, bundled MIT code |
| `qwen3-4b-instruct-2507-<rev>.tar.part-*` | the core model, unchanged, split into parts | Apache-2.0 (Qwen) |
| `qwen3-embedding-0.6b-<rev>.tar` | the encoder, unchanged | Apache-2.0 (Qwen) |
| `SHA256SUMS` | checksums of every asset | — |

The base models are also available from Hugging Face at the revisions listed in [THIRD-PARTY.md](THIRD-PARTY.md).

## Quick start

You need Linux, Python 3.12 or newer, and an NVIDIA GPU with CUDA and at least 12 GB of free memory. Yaob uses about 9.5 GB at its peak. It was developed and measured on a DGX Spark (aarch64). See [docs/install.md](docs/install.md) for the full steps, the CPU-only editor and training.

```bash
git clone https://github.com/sarja-technologies/yaob && cd yaob
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[model]'
scripts/fetch_release.sh ~/yaob                       # about 9.5 GB, checksummed
python -m mpnstudio.store.bundle import ~/yaob/bundle/yaob-n10k ~/yaob/data/yaob-n10k.sqlite
export MPN_DATA=~/yaob/data MPN_MODELS=~/yaob/models MPN_WEB=~/yaob/web-dist
python -m mpnstudio.runtime.server &                  # runtime and chat endpoint on 127.0.0.1:8841
python -m mpnstudio.api.app                           # MPN Studio on http://127.0.0.1:8840
```

Any OpenAI-compatible chat client can use `http://127.0.0.1:8841/v1` with the model `yaob-n10k`.

## Documentation

- [docs/install.md](docs/install.md): install, run, train and evaluate
- [docs/architecture.md](docs/architecture.md): what each component does and what is learned
- [docs/formats.md](docs/formats.md): bundle and MPN package formats
- [docs/api.md](docs/api.md): HTTP API
- [docs/corpus.md](docs/corpus.md): how the 10,000 nodes were selected and built
- [docs/core-selection.md](docs/core-selection.md): why Qwen3-4B-Instruct-2507
- [docs/scale.md](docs/scale.md): measurements at one million nodes

## Licence

Yaob and MPN Studio are **source-available, not open source**. You may download, use, study, modify and give away the code and weights free of charge. This includes use inside a company, and paid consulting or support. You may not sell them or a modified version, or offer them as a paid hosted service. You must keep Fylume's and the upstream notices, and you must not claim authorship. The node data is CC BY-SA 4.0 and carries no resale restriction.

- Code: [LICENSE](LICENSE)
- Weights trained by Fylume: [LICENSE-WEIGHTS](LICENSE-WEIGHTS)
- Data: [DATA-LICENSE.md](DATA-LICENSE.md)
- Third parties: [THIRD-PARTY.md](THIRD-PARTY.md)
- Plain-language summary: [LICENSING.md](LICENSING.md)

The Fylume and Yaob names and logos are not licensed for other uses.
