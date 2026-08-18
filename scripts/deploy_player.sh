#!/usr/bin/env bash
# Re-export the best net, re-check the player, and publish it everywhere it
# lives: the Hugging Face Space (the canonical site) and GitHub Pages (a
# "we moved" stub for every link already in the wild, plus a still-playable
# copy at classic/ as a fallback). See docs/HUGGINGFACE.md.
#
#   ./scripts/deploy_player.sh              # export + test + push HF + gh-pages
#   ./scripts/deploy_player.sh --no-export  # publish web/player/ as it stands
#   ./scripts/deploy_player.sh --dry-run    # build and test, push nothing
#
# The site is web/player/ minus its test and space directories. Pages is
# published to the `gh-pages` branch **through a temporary git index**, never
# by checking that branch out: a training run is usually writing into runs/
# while this runs, and nothing here may disturb the main working tree. The HF
# push needs HF_TOKEN in the environment and git-lfs installed (the model and
# the wasm runtimes ride LFS there).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE_SRC="$REPO_ROOT/web/player"
BRANCH="gh-pages"
REMOTE="origin"
SPACE="RemiFabre/faience"
PLAY_URL="https://remifabre-faience.static.hf.space/"
PAGES_URL="https://remifabre.github.io/ludometer/"

DO_EXPORT=1
DO_PUSH=1
for arg in "$@"; do
  case "$arg" in
    --no-export) DO_EXPORT=0 ;;
    --dry-run) DO_PUSH=0 ;;
    -h|--help) sed -n '2,9p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

cd "$REPO_ROOT"

# 1. the model ------------------------------------------------------------------
if [ "$DO_EXPORT" = 1 ]; then
  say "Exporting the best rated checkpoint to ONNX"
  # `--group export` pulls in onnx + onnxruntime, which only this path needs.
  # nice + one thread: a training run usually owns the CPU while this is running.
  OMP_NUM_THREADS=1 nice -n 15 uv run --group export \
    python -m ludometer.export.onnx_export
else
  say "Skipping the export (--no-export)"
fi

# 2. the checks -----------------------------------------------------------------
say "Checking the JS engine against the Python fixtures"
nice -n 15 node web/player/test/engine.test.mjs

say "Checking onnxruntime-web against the torch reference"
nice -n 15 node web/player/test/parity.test.mjs

# The GPU path only some visitors take, held to the same reference. Skips itself
# (and passes) on a machine with no WebGPU adapter.
say "Checking the WebGPU backend against the same reference"
nice -n 15 node web/player/test/webgpu.test.mjs

say "Checking output detection, batched read-out and virtual-loss bookkeeping"
nice -n 15 node web/player/test/margin.test.mjs

# 3. stage the site -------------------------------------------------------------
say "Staging the site"
STAGE="$(mktemp -d)"
INDEX="$STAGE.index"  # must not exist yet: git refuses to read an empty index file
HF_CLONE="$STAGE.hf"
PAGES="$STAGE.pages"
trap 'rm -rf "$STAGE" "$INDEX" "$HF_CLONE" "$PAGES"' EXIT

# everything a visitor needs, and nothing else: the fixtures, the node tests
# and the Space card are development apparatus and would triple the download
# for no one's benefit.
rsync -a --exclude 'test/' --exclude 'space/' --exclude '.DS_Store' "$SITE_SRC/" "$STAGE/"

if [ ! -f "$STAGE/model/model.onnx" ]; then
  echo "no model at $STAGE/model/model.onnx — run without --no-export" >&2
  exit 1
fi
SIZE=$(du -sk "$STAGE" | cut -f1)
echo "site payload: $((SIZE / 1024)) MB across $(find "$STAGE" -type f | wc -l | tr -d ' ') files"
# The site carries two onnxruntime builds but every visitor downloads exactly
# one of them, so the total on disk is not what anybody waits for. Pages serves
# these gzipped, which is the number that matters.
gz() { gzip -c "$1" 2>/dev/null | wc -c | tr -d ' '; }
V="$STAGE/vendor/onnxruntime-web"
if [ -f "$V/ort-wasm-simd-threaded.wasm" ]; then
  MODEL_GZ=$(gz "$STAGE/model/model.onnx")
  CPU_GZ=$(gz "$V/ort-wasm-simd-threaded.wasm")
  GPU_GZ=$([ -f "$V/ort-wasm-simd-threaded.jspi.wasm" ] && gz "$V/ort-wasm-simd-threaded.jspi.wasm" || echo 0)
  printf 'over the wire (gzipped), per visitor: %s MB on the CPU path' \
    "$(echo "$MODEL_GZ $CPU_GZ" | awk '{printf "%.1f", ($1+$2)/1e6}')"
  [ "$GPU_GZ" != 0 ] && printf ', %s MB on the WebGPU path' \
    "$(echo "$MODEL_GZ $GPU_GZ" | awk '{printf "%.1f", ($1+$2)/1e6}')"
  printf '\n'
fi

# 4. the canonical site: the Hugging Face Space ---------------------------------
say "Building the $SPACE commit"
: "${HF_TOKEN:?HF_TOKEN must be set (the Space push needs it)}"
CKPT=$(python3 -c "import json,sys; m=json.load(open('$STAGE/model/model_meta.json')); print(f\"{m['run']}/{m['checkpoint']} elo {m['elo']}\")")
MESSAGE="Deploy browser player — $CKPT"
git clone --quiet --depth 1 "https://RemiFabre:$HF_TOKEN@huggingface.co/spaces/$SPACE" "$HF_CLONE"
# the Space's own files (.gitattributes routes the model and wasm through LFS,
# README.md is the card) survive the sync; everything else mirrors the stage
rsync -a --delete --exclude '.git' --exclude '.gitattributes' --exclude 'README.md' \
  "$STAGE/" "$HF_CLONE/"
cp "$SITE_SRC/space/README.md" "$HF_CLONE/README.md"
# the Hub refuses binary files outside LFS; the default .gitattributes already
# routes *.onnx and *.wasm, but the social card is a png
grep -q '^\*\.png ' "$HF_CLONE/.gitattributes" \
  || echo '*.png filter=lfs diff=lfs merge=lfs -text' >> "$HF_CLONE/.gitattributes"
git -C "$HF_CLONE" add -A
if git -C "$HF_CLONE" diff --cached --quiet; then
  echo "the Space is already current"
else
  git -C "$HF_CLONE" -c user.name="deploy_player.sh" -c user.email="remi.fabre@pollen-robotics.com" \
    commit --quiet -m "$MESSAGE"
fi

# 5. the old address: a moved-notice stub, and the same site at classic/ ---------
say "Building the $BRANCH commit (stub + classic/)"
mkdir -p "$PAGES/classic"
cp "$REPO_ROOT/web/pages/index.html" "$PAGES/index.html"
cp "$STAGE/social.png" "$PAGES/social.png"  # the circulating cards point here
rsync -a "$STAGE/" "$PAGES/classic/"
touch "$PAGES/.nojekyll"  # keep Jekyll's hands off vendor/ and _-prefixed names
export GIT_INDEX_FILE="$INDEX"
git --git-dir="$REPO_ROOT/.git" --work-tree="$PAGES" add -A .
TREE=$(git --git-dir="$REPO_ROOT/.git" write-tree)
PARENT=$(git --git-dir="$REPO_ROOT/.git" rev-parse --verify --quiet "refs/heads/$BRANCH" || true)
if [ -n "$PARENT" ]; then
  COMMIT=$(git --git-dir="$REPO_ROOT/.git" commit-tree "$TREE" -p "$PARENT" -m "$MESSAGE")
else
  COMMIT=$(git --git-dir="$REPO_ROOT/.git" commit-tree "$TREE" -m "$MESSAGE")
fi
unset GIT_INDEX_FILE
git update-ref "refs/heads/$BRANCH" "$COMMIT"
echo "$BRANCH -> $COMMIT ($MESSAGE)"

if [ "$DO_PUSH" = 0 ]; then
  say "Dry run — not pushing. Inspect with: git show --stat $COMMIT ; ls $HF_CLONE"
  trap - EXIT
  echo "stages kept for inspection: $HF_CLONE and $PAGES (rm them yourself)"
  rm -rf "$STAGE" "$INDEX"
  exit 0
fi

# 6. push both, and wait until they answer ---------------------------------------
say "Pushing $SPACE"
git -C "$HF_CLONE" push --quiet origin main

say "Pushing $BRANCH"
git push "$REMOTE" "$BRANCH"

say "Making sure GitHub Pages is serving $BRANCH"
# Both calls answer with an empty body on success, which makes gh complain about
# JSON it never got — the exit status is what matters, so the noise is dropped.
if gh api "repos/RemiFabre/ludometer/pages" >/dev/null 2>&1; then
  gh api "repos/RemiFabre/ludometer/pages" -X PUT \
    -f "source[branch]=$BRANCH" -f "source[path]=/" >/dev/null 2>&1 \
    && echo "Pages source confirmed: $BRANCH /"
else
  gh api "repos/RemiFabre/ludometer/pages" -X POST \
    -f "source[branch]=$BRANCH" -f "source[path]=/" >/dev/null 2>&1 \
    && echo "Pages enabled on $BRANCH /"
fi

say "Waiting for $PLAY_URL to answer"
for _ in $(seq 1 40); do
  # the Space 302s / to /index.html, so ask for the page itself
  CODE=$(curl -s -o /dev/null -w '%{http_code}' "${PLAY_URL}index.html" || true)
  if [ "$CODE" = "200" ]; then
    echo "live: $PLAY_URL"
    curl -s -o /dev/null -w 'model/model_meta.json -> %{http_code}\n' "${PLAY_URL}model/model_meta.json"
    break
  fi
  printf '  %s ... waiting 15s\n' "$CODE"
  sleep 15
done
[ "${CODE:-}" = "200" ] || { echo "the Space is still not 200 — check https://huggingface.co/spaces/$SPACE" >&2; exit 1; }

say "Waiting for $PAGES_URL to answer with the stub"
for _ in $(seq 1 40); do
  BODY=$(curl -s "$PAGES_URL" || true)
  if printf '%s' "$BODY" | grep -q "has moved"; then
    echo "live: $PAGES_URL (stub)"
    curl -s -o /dev/null -w 'classic/model/model_meta.json -> %{http_code}\n' "${PAGES_URL}classic/model/model_meta.json"
    exit 0
  fi
  printf '  stub not up yet ... waiting 15s\n'
  sleep 15
done
echo "the stub is still not live after ~10 minutes — check https://github.com/RemiFabre/ludometer/deployments" >&2
exit 1
