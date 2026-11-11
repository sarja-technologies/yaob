#!/usr/bin/env bash
# Runs the network-bound corpus stages in order. Every stage is cached, so re-running resumes.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-data/corpus}
while pgrep -f "mpnstudio.corpus.[p]ages" >/dev/null; do sleep 20; done
[ -f "$OUT/pageinfo.json" ] || python3 -m mpnstudio.corpus.pages --out "$OUT"
python3 -m mpnstudio.corpus.select --out "$OUT"
python3 -m mpnstudio.corpus.fetch --out "$OUT" --stage leads
python3 -m mpnstudio.corpus.fetch --out "$OUT" --stage wikidata
python3 -m mpnstudio.corpus.fetch --out "$OUT" --stage labels
echo CORPUS_FETCH_DONE
