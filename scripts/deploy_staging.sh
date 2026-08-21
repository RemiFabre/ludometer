#!/usr/bin/env bash
# Publish web/player/ AS IT STANDS to the PRIVATE staging Space, so changes can
# be played for real before touching production. The staging hostname contains
# "staging", which js/upload.js and js/analytics.js treat like localhost: no
# game is ever sent to the collector and no hit touches the tally from there.
#
#   ./scripts/deploy_staging.sh           # checks + push
#   ./scripts/deploy_staging.sh --fast    # skip the JS engine/parity checks
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPACE="RemiFabre/faience-staging"
URL="https://remifabre-faience-staging.static.hf.space/"

: "${HF_TOKEN:?HF_TOKEN must be set}"
say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
cd "$REPO_ROOT"

if [ "${1:-}" != "--fast" ]; then
  say "Checking the JS engine and the runtime"
  nice -n 15 node web/player/test/engine.test.mjs
  nice -n 15 node web/player/test/margin.test.mjs
fi

say "Staging"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
rsync -a --exclude 'test/' --exclude 'space/' --exclude '.DS_Store' web/player/ "$STAGE/site/"
git clone --quiet --depth 1 "https://RemiFabre:$HF_TOKEN@huggingface.co/spaces/$SPACE" "$STAGE/space"
rsync -a --delete --exclude '.git' --exclude '.gitattributes' --exclude 'README.md' \
  "$STAGE/site/" "$STAGE/space/"
grep -q '^\*\.png ' "$STAGE/space/.gitattributes" \
  || echo '*.png filter=lfs diff=lfs merge=lfs -text' >> "$STAGE/space/.gitattributes"
cat > "$STAGE/space/README.md" <<'EOF'
---
title: "Faïence staging"
emoji: "🚧"
colorFrom: gray
colorTo: yellow
sdk: static
pinned: false
---
The private test copy of Faïence. Nothing played here is recorded or counted.
EOF

cd "$STAGE/space"
git add -A
if git diff --cached --quiet; then
  echo "staging is already current"
else
  git -c user.name="deploy_staging.sh" -c user.email="remi.fabre@pollen-robotics.com" \
    commit --quiet -m "Staging deploy — $(git -C "$REPO_ROOT" rev-parse --short HEAD) + working tree"
  git push --quiet origin main
fi

say "Waiting for $URL"
for _ in $(seq 1 40); do
  CODE=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $HF_TOKEN" "${URL}index.html" || true)
  if [ "$CODE" = "200" ]; then echo "live: $URL"; exit 0; fi
  printf '  %s ... waiting 10s\n' "$CODE"
  sleep 10
done
echo "staging not live yet — check https://huggingface.co/spaces/$SPACE" >&2
exit 1
