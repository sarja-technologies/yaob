# Fresh install of the public release (12 Nov 2026)

On the DGX Spark, in a folder separate from the development install:

1. `git clone https://github.com/sarja-technologies/yaob` (main at `a34810f`).
2. A new virtual environment: `pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132`, then `pip install -e '.[model,dev]'`.
3. `scripts/fetch_release.sh ~/yaob-verify/r --tag v0.1.0`: all nine assets downloaded from GitHub and matched `SHA256SUMS`, and the core was reassembled from its five parts.
4. `python -m mpnstudio.store.bundle verify` and `import`: the counts matched the bundle (10,000 nodes, 105,857 relations, 12,254 memberships), and all 10,160 keyed nodes were restored as trained.
5. Tests: 71 passed and 19 were skipped.
6. The runtime started from that install on 127.0.0.1:18841 and answered "What is the continent of the place of birth of Otto Hahn?" with "Europe" (via Frankfurt).
7. The held-out suite (`docs/measurements/heldout-eval-001/suite.json`), arm `deployed`, ran against that runtime and scored **211/236 (89 %)**. Every family matched eval-002 (`report.md` here).

The first import, at `877ae1b`, restored every node as untrained: cells cite sources by the exporting store's ids, so the recomputed content hash differed. The fix is `a34810f`.
