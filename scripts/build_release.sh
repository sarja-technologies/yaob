#!/usr/bin/env bash
# Build the release assets that scripts/fetch_release.sh downloads.
#
#   scripts/build_release.sh VERSION DB MODELS WEB_DIST OUT
#
# VERSION   e.g. 0.1.0
# DB        the store to export (its active checkpoint is included)
# MODELS    folder holding Qwen3-4B-Instruct-2507/ (with CORE.json) and Qwen3-Embedding-0.6B/
# WEB_DIST  the built web interface (pnpm --dir web build writes web/dist)
# OUT       an empty or new folder for the assets
#
# The base models are packed unchanged. Hugging Face cache folders are left out. Files above 1,900 MiB are split into
# <name>.part-aa, -ab, ... so that each asset stays below GitHub's 2 GiB limit. SHA256SUMS lists every asset.
set -euo pipefail
[ $# -eq 5 ] || { sed -n 2,13p "$0"; exit 2; }
V="$1" DB="$2" MODELS="$3" WEB="$4" OUT="$5"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
CORE=Qwen3-4B-Instruct-2507 ENC=Qwen3-Embedding-0.6B
[ -f "$MODELS/$CORE/CORE.json" ] || { echo "$MODELS/$CORE/CORE.json is missing" >&2; exit 1; }
mkdir -p "$OUT"
[ -z "$(ls -A "$OUT")" ] || { echo "$OUT is not empty" >&2; exit 1; }
W="$OUT/.work"; mkdir -p "$W/default" "$W/specialists" "$W/licence/$ENC"

echo "bundle, default (router, no specialist adapters)"
"$PY" -m mpnstudio.store.bundle export "$DB" "$W/default/yaob-n10k" --name yaob-n10k --core-json "$MODELS/$CORE/CORE.json" --card "$HERE/MODEL_CARD.md" --no-adapters
"$PY" -m mpnstudio.store.bundle verify "$W/default/yaob-n10k"
tar -czf "$OUT/yaob-n10k-$V.tar.gz" -C "$W/default" yaob-n10k

echo "bundle, with the experimental specialist adapters"
"$PY" -m mpnstudio.store.bundle export "$DB" "$W/specialists/yaob-n10k" --name yaob-n10k --core-json "$MODELS/$CORE/CORE.json" --card "$HERE/MODEL_CARD.md"
"$PY" -m mpnstudio.store.bundle verify "$W/specialists/yaob-n10k"
tar -czf "$OUT/yaob-n10k-$V-specialists-experimental.tar.gz" -C "$W/specialists" yaob-n10k

echo "web interface"
tar -czf "$OUT/mpn-studio-web-$V.tar.gz" -C "$(dirname "$WEB")" --transform "s|^$(basename "$WEB")|web-dist|" "$(basename "$WEB")"

rev() { "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"][:8])' "$1"; }
CREV=$(rev "$MODELS/$CORE/CORE.json")
echo "core $CORE ($CREV)"
tar -cf - --exclude=.cache -C "$MODELS" "$CORE" | split -b 1900M -a 2 - "$OUT/qwen3-4b-instruct-2507-$CREV.tar.part-"

# The encoder's repository has no LICENSE file; its model card declares Apache-2.0. The Apache License 2.0 text from the
# core's repository (same licensor) is added beside it as LICENSE. The model files themselves are unchanged.
cp "$MODELS/$CORE/LICENSE" "$W/licence/$ENC/LICENSE"
ENC_REV="${ENC_REV:-97b0c614}"
echo "encoder $ENC ($ENC_REV)"
tar -cf "$OUT/qwen3-embedding-0.6b-$ENC_REV.tar" --exclude=.cache -C "$MODELS" "$ENC" -C "$W/licence" "$ENC/LICENSE"

rm -rf "$W"
(cd "$OUT" && sha256sum -- * > SHA256SUMS)
ls -l "$OUT"
