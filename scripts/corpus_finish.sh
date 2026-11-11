#!/usr/bin/env bash
# After scripts/corpus_chain.sh: build the canonical records, load them as one import revision, add the diagnostic
# canary MPN (disabled), and lay out the board. Refuses to overwrite an existing store.
#   bash scripts/corpus_finish.sh [corpus dir] [store path]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-data/corpus}
DB=${2:-data/yaob-n10k.sqlite}
PY=${PY:-.venv/bin/python}
[ -e "$DB" ] && { echo "$DB exists; refusing to overwrite"; exit 1; }
$PY -m mpnstudio.corpus.build --out "$OUT" > "$OUT/build.out"
$PY -c "import json,sys; r=json.load(open('$OUT/build_report.json')); print('exact_10000', r['exact_10000'], r['totals']['nodes']); sys.exit(0 if r['exact_10000'] else 1)"
$PY -m mpnstudio.corpus.load --corpus "$OUT" --db "$DB"
$PY -m mpnstudio.corpus.canary --db "$DB"
$PY - "$DB" <<'PY'
import json, sys
from mpnstudio.store.db import Store
from mpnstudio.store.layout import compute_layout
s = Store(sys.argv[1])
print(json.dumps(compute_layout(s)))
c = s.counts()
print(json.dumps(c))
assert c["canonical_live_knowledge_procedure_nodes"] == 10000, c
PY
echo CORPUS_FINISH_DONE
