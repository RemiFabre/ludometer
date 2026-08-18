#!/usr/bin/env bash
# Test and publish the game collector to the Hugging Face Space
# RemiFabre/faience-ingest (see web/ingest/ and docs/HUGGINGFACE.md).
#
#   ./scripts/deploy_ingest.sh            # test + push
#   ./scripts/deploy_ingest.sh --dry-run  # test + build the stage, push nothing
#
# The Space is self-contained: web/ingest/engine.js (a dev-time shim that
# re-exports the player's engine) is REPLACED in the stage by a real copy of
# web/player/js/engine.js, so the collector validates games with exactly the
# rules the page plays by. Auth comes from HF_TOKEN in the environment.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPACE="RemiFabre/faience-ingest"
SPACE_URL="https://remifabre-faience-ingest.hf.space"

DO_PUSH=1
[ "${1:-}" = "--dry-run" ] && DO_PUSH=0

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

: "${HF_TOKEN:?HF_TOKEN must be set}"

say "Running the ingest tests"
node "$REPO_ROOT/web/ingest/test/ingest.test.mjs"

say "Staging the Space"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
git clone --quiet --depth 1 "https://RemiFabre:$HF_TOKEN@huggingface.co/spaces/$SPACE" "$STAGE/space"
cp "$REPO_ROOT/web/ingest/server.js" \
   "$REPO_ROOT/web/ingest/verify.js" \
   "$REPO_ROOT/web/ingest/package.json" \
   "$REPO_ROOT/web/ingest/Dockerfile" \
   "$REPO_ROOT/web/ingest/README.md" "$STAGE/space/"
cp "$REPO_ROOT/web/player/js/engine.js" "$STAGE/space/engine.js"

cd "$STAGE/space"
git add -A
if git diff --cached --quiet; then
  echo "nothing changed; the Space is already current"
  exit 0
fi
git -c user.name="deploy_ingest.sh" -c user.email="remi.fabre@pollen-robotics.com" \
  commit --quiet -m "Deploy ingest — $(git -C "$REPO_ROOT" rev-parse --short HEAD)"

if [ "$DO_PUSH" = 0 ]; then
  say "Dry run — not pushing. Stage: $STAGE/space"
  trap - EXIT
  exit 0
fi

say "Pushing $SPACE"
git push --quiet origin main

say "Waiting for $SPACE_URL/health"
for _ in $(seq 1 40); do
  CODE=$(curl -s -o /dev/null -w '%{http_code}' "$SPACE_URL/health" || true)
  if [ "$CODE" = "200" ]; then
    echo "live: $SPACE_URL"
    curl -s "$SPACE_URL/stats"
    echo
    exit 0
  fi
  printf '  %s ... waiting 15s\n' "$CODE"
  sleep 15
done
echo "still not 200 after ~10 minutes — check https://huggingface.co/spaces/$SPACE" >&2
exit 1
