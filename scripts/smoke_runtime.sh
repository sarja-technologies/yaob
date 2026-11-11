#!/usr/bin/env bash
# End-to-end smoke test of the learned path on the Spark, in an isolated data directory.
#   store:   the synthetic dev fixture plus the diagnostic canary MPN (invented facts the core cannot know), canary enabled
#   train:   train_bank: router, then the canary MPN's specialist adapter (small budget)
#   runtime: a private runtime on loopback port 8849; activate the checkpoint; ask canary questions in the three modes
# Nothing here touches ~/mpn-studio/data or the deployed units. Run on the Spark from ~/mpn-studio/app:
#   bash scripts/smoke_runtime.sh [steps]
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:-120}
export MPN_DATA=$HOME/mpn-studio/data-smoke MPN_DB=$HOME/mpn-studio/data-smoke/smoke.sqlite MPN_MODELS=$HOME/mpn-studio/models
export MPN_NO_SYSTEMD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 MPN_RUNTIME_PORT=8849
PY=../venv/bin/python
mkdir -p "$MPN_DATA"
if [ ! -f "$MPN_DB" ]; then
  cp data/dev-fixture.sqlite "$MPN_DB"
  $PY - <<'EOF'
import os
from mpnstudio.store.db import Store
from mpnstudio.corpus import canary
s = Store(os.environ["MPN_DB"])
print(canary.install(s, 8))
m = s.mpn_by_slug("diagnostic-canary")
with s.edit(actor="smoke", kind="mpn", message="enable canary for the smoke test") as tx:
    tx.set_mpn_state(int(m["id"]), enabled=True)
EOF
fi
JOB=$($PY - <<EOF
import os
from mpnstudio.store.db import Store
from mpnstudio.train import jobs
s = Store(os.environ["MPN_DB"])
print(jobs.submit(s, "train_bank", {"router": True, "router_epochs": 3, "adapters": ["diagnostic-canary"], "steps": $STEPS, "batch": 8,
                                    "rank": 16, "eval_n": 200, "adapter_eval_n": 48, "base": "none"}))
EOF
)
echo "job $JOB"
$PY -m mpnstudio.train.jobs run "$JOB"
CK=$($PY -c "
import os,json; from mpnstudio.store.db import Store; from mpnstudio.train import jobs
j = jobs.get(Store(os.environ['MPN_DB']), '$JOB'); print(j['state'], json.dumps(j['result'])[:600], file=__import__('sys').stderr); print((j['result'] or {}).get('checkpoint',''))")
echo "checkpoint $CK"
[ -n "$CK" ] || exit 1
MPN_RUNTIME_AUTOLOAD=1 setsid nohup $PY -m mpnstudio.runtime.server > "$MPN_DATA/runtime.log" 2>&1 < /dev/null &
RT=$!
trap 'kill $RT 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
  if curl -s -m 3 localhost:8849/status | grep -q '"loaded": true'; then break; fi
  sleep 5
done
curl -s -m 600 -X POST localhost:8849/activate -H 'content-type: application/json' -d "{\"checkpoint\": \"$CK\"}"; echo
$PY scripts/smoke_ask.py "$MPN_DB" http://127.0.0.1:8849
