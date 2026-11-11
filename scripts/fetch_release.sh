#!/usr/bin/env bash
# Download a Yaob release, check every file against SHA256SUMS, reassemble split files and unpack.
#
#   scripts/fetch_release.sh DEST [--tag vX.Y.Z] [--specialists] [--no-base] [--no-web]
#
# DEST/downloads  the release files as published (kept, so a second run only fetches what is missing or wrong)
# DEST/models     Qwen3-4B-Instruct-2507 and Qwen3-Embedding-0.6B, unchanged
# DEST/bundle     yaob-n10k: node data, layout and router checkpoint (and specialist adapters with --specialists)
# DEST/web-dist   the built MPN Studio web interface
#
# Needs bash, curl, sha256sum (or shasum), tar and gzip. Set YAOB_REPO to download from a fork.
set -euo pipefail

REPO="${YAOB_REPO:-sarja-technologies/yaob}"
DEST="" TAG="" SPECIALISTS=0 BASE=1 WEB=1
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --specialists) SPECIALISTS=1; shift ;;
    --no-base) BASE=0; shift ;;
    --no-web) WEB=0; shift ;;
    -h|--help) sed -n 2,11p "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) DEST="$1"; shift ;;
  esac
done
[ -n "$DEST" ] || { sed -n 2,11p "$0"; exit 2; }

if command -v sha256sum >/dev/null; then SHA="sha256sum"; else SHA="shasum -a 256"; fi
if [ -z "$TAG" ]; then
  TAG=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)
  [ -n "$TAG" ] || { echo "could not find the latest release of $REPO; pass --tag" >&2; exit 1; }
fi
BASE_URL="https://github.com/$REPO/releases/download/$TAG"
DL="$DEST/downloads/$TAG"
mkdir -p "$DL" "$DEST/models" "$DEST/bundle" "$DEST/data"
echo "Yaob release $TAG from $REPO into $DEST"

curl -fsSL -o "$DL/SHA256SUMS" "$BASE_URL/SHA256SUMS"

want() {   # names from SHA256SUMS matching a pattern
  awk '{print $2}' "$DL/SHA256SUMS" | sed 's/^\*//' | grep -E "$1" || true
}
fetch() {  # download one file unless it is already present and correct
  local f="$1" sum
  sum=$(awk -v f="$f" '{n=$2; sub(/^\*/, "", n)} n == f {print $1}' "$DL/SHA256SUMS")
  [ -n "$sum" ] || { echo "$f is not listed in SHA256SUMS" >&2; exit 1; }
  if [ -f "$DL/$f" ] && [ "$($SHA "$DL/$f" | cut -d' ' -f1)" = "$sum" ]; then
    echo "  ok      $f"; return
  fi
  echo "  fetch   $f"
  curl -fL --retry 5 --retry-delay 5 -C - -o "$DL/$f.partial" "$BASE_URL/$f" || curl -fL --retry 5 -o "$DL/$f.partial" "$BASE_URL/$f"
  [ "$($SHA "$DL/$f.partial" | cut -d' ' -f1)" = "$sum" ] || { echo "checksum mismatch in $f; delete $DL/$f.partial and retry" >&2; exit 1; }
  mv "$DL/$f.partial" "$DL/$f"
}

if [ "$SPECIALISTS" = 1 ]; then BUNDLE=$(want '^yaob-n10k-.*-specialists-experimental\.tar\.gz$'); else BUNDLE=$(want '^yaob-n10k-[0-9.]+\.tar\.gz$'); fi
[ -n "$BUNDLE" ] || { echo "no bundle in this release" >&2; exit 1; }
fetch "$BUNDLE"
rm -rf "$DEST/bundle/yaob-n10k"
tar -xzf "$DL/$BUNDLE" -C "$DEST/bundle"
echo "  unpacked $DEST/bundle/yaob-n10k"

if [ "$WEB" = 1 ]; then
  W=$(want '^mpn-studio-web-.*\.tar\.gz$')
  if [ -n "$W" ]; then
    fetch "$W"; rm -rf "$DEST/web-dist"; tar -xzf "$DL/$W" -C "$DEST"; echo "  unpacked $DEST/web-dist"
  fi
fi

if [ "$BASE" = 1 ]; then
  # the core is split into parts below GitHub's 2 GiB limit: <name>.tar.part-aa, -ab, ...
  for stem in $(want '\.tar(\.part-[a-z]+)?$' | sed 's/\.part-[a-z]*$//' | sort -u); do
    parts=$(want "^${stem//./\\.}(\.part-[a-z]+)?$" | sort)
    for p in $parts; do fetch "$p"; done
    echo "  unpack  $stem"
    # shellcheck disable=SC2086
    (cd "$DL" && cat $parts) | tar -xf - -C "$DEST/models"
  done
  echo "  unpacked $DEST/models"
fi

echo
echo "Next:"
echo "  python -m mpnstudio.store.bundle import $DEST/bundle/yaob-n10k $DEST/data/yaob-n10k.sqlite"
echo "  export MPN_DATA=$DEST/data MPN_MODELS=$DEST/models MPN_WEB=$DEST/web-dist"
