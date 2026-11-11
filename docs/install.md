# Install, run, train and evaluate

## Requirements

| To | You need |
|---|---|
| Edit the network in MPN Studio, search it, use the CLI, export and import bundles | Linux or macOS, Python 3.12 or newer |
| Ask Yaob questions (the runtime) | Linux, an NVIDIA GPU with CUDA and 12 GB of free memory, about 12 GB of disk |
| Train the router or specialists | the same GPU with about 40 GB of free memory |

Measured on a DGX Spark (aarch64, unified memory): a request used at most 9.5 GiB of GPU memory and 2.5 GiB of process memory. Generation ran at about 20 tokens per second.

## 1. Code

```bash
git clone https://github.com/sarja-technologies/yaob && cd yaob
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[model,dev]'        # leave out "model" for the editor without the runtime
```

Install PyTorch first if your platform needs a specific build (see https://pytorch.org/get-started/locally/).

## 2. Release files

```bash
scripts/fetch_release.sh ~/yaob                      # latest release, default setup
scripts/fetch_release.sh ~/yaob --specialists        # also the experimental specialist adapters
scripts/fetch_release.sh ~/yaob --tag v0.1.0 --no-base  # skip the base models (for example, to fetch them from Hugging Face)
```

The script downloads into `~/yaob/downloads` and checks each file against `SHA256SUMS`. It reassembles the split core model and unpacks everything:

```
~/yaob/models/Qwen3-4B-Instruct-2507/    core model, unchanged, with CORE.json (repository, revision, licence)
~/yaob/models/Qwen3-Embedding-0.6B/      encoder, unchanged
~/yaob/bundle/yaob-n10k/                 node data, layout, router checkpoint (and adapters, if chosen)
~/yaob/web-dist/                         built web interface
```

To fetch the base models from Hugging Face instead, download these revisions into the same folders:

- `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`
- `Qwen/Qwen3-Embedding-0.6B`, revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`

Then copy `CORE.json` from the release, or write it with the repository, revision and licence. The runtime records those fields in its evidence.

## 3. Create the store

```bash
python -m mpnstudio.store.bundle verify ~/yaob/bundle/yaob-n10k
python -m mpnstudio.store.bundle import ~/yaob/bundle/yaob-n10k ~/yaob/data/yaob-n10k.sqlite
```

Import refuses to overwrite an existing store. It checks every file's SHA-256 and rebuilds the router checkpoint for this store's node ids. It prints the node, relation and membership counts and whether they match the bundle. For Yaob N10K that is 10,000 nodes, 105,857 relations and 12,254 memberships.

## 4. Run

```bash
cp deploy/mpn-studio.env.example deploy/mpn-studio.env     # then edit the paths
set -a; . deploy/mpn-studio.env; set +a
python -m mpnstudio.runtime.server &        # loads the core and the active checkpoint, serves 127.0.0.1:8841
python -m mpnstudio.api.app &               # MPN Studio at http://127.0.0.1:8840
python -m mpnstudio.train.jobs worker &     # runs training and evaluation jobs
```

The `deploy/*.service` files are systemd user units for the same three processes.

The API and the web interface accept requests only from the networks in `MPN_ALLOWED_NETS`, which defaults to this machine. There is no password, so add only networks you trust.

Ask from the command line:

```bash
mpn status
mpn run "What is the boiling point of ethanol?" --evidence
mpn run "Who wrote Hamlet?" --mode retrieval
```

Or use any OpenAI-compatible client with base URL `http://127.0.0.1:8841/v1`:

| Model id | What answers |
|---|---|
| `yaob-n10k` | learned routing, node text, procedures; specialist adapters only if the budget allows them |
| `yaob-n10k-retrieval` | ordinary retrieval over the same nodes with the same core, for comparison |

Every reply carries an evidence record that lists the MPNs consulted, the nodes read and how each was read. It is a record of what was accessed, not an explanation of the model's reasoning.

## Budget

The budget sets how much each request may read and write. Defaults: an 8,192-token context, 1,024 output tokens, no specialist adapters, relation-aware referrals. Change it in MPN Studio's Budget panel or with `mpn budget show`, `mpn budget set key=value` and `mpn budget preset <name>`. The preset `specialists (experimental)` allows one adapter per request. With an adapter, closed-book recall improves, but the model declines fewer questions about things it does not know (see MODEL_CARD.md).

A request whose input cannot fit is refused with HTTP 413. The message gives the needed and available token counts, so clients can trim older turns.

## Train

Training runs as a job. You can start it from MPN Studio's Jobs panel or with the CLI. The job pauses when GPU memory is short, saves a checkpoint after each stage and can be resumed:

```bash
mpn job start train_bank                                   # router only
mpn job start train_bank adapters='["chemistry"]'          # router and one specialist
mpn job log <job id> --follow
mpn checkpoint list
mpn checkpoint activate <checkpoint id>
```

After you edit nodes, `mpn job start retrain_nodes only_nodes='[<ids>]'` updates just those nodes' routing keys and their MPNs' specialists. Until then, edited nodes are routed by keys derived from their text. A specialist is not applied to a request when a selected node of its MPN changed after training.

The released checkpoint's training settings are in `checkpoint/manifest.json` inside the bundle.

## Evaluate

Build a held-out suite once, before training on anything it covers, and do not tune on it:

```bash
python -m mpnstudio.evals.heldout build --out suite.json --seed 20261111
python -m mpnstudio.evals.heldout run --suite suite.json --out runs/eval --arms core,retrieval,deployed
```

`run` writes `rows.jsonl` and `report.md`/`report.json` with accuracy per task family and Wilson 95 % intervals. It also reports routing recall, prompt size, latency and memory. To compare settings, use the development-set tools `mpnstudio.evals.dev_compare` and `mpnstudio.evals.multihop_diag`. They draw questions under a different seed and drop any that share a node with the held-out suite.

The suite used for the published numbers is `docs/measurements/heldout-eval-001/suite.json`.

## Edit the network

MPN Studio shows all 10,000 nodes on one whiteboard. From there you can inspect and edit a node's cells and relations, move nodes between MPNs, and create, enable, disable, install or uninstall MPNs. Before you uninstall an MPN, Studio shows its impact. Every change is a revision and can be undone. `mpn bundle export` and `mpn bundle export-mpn <slug>` write portable packages (see formats.md).
