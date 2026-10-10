#!/usr/bin/env bash
# Build the two leaderboard candidates (D' family) with the exact flags of the submitted notebooks.
#
#   scripts/build_candidates.sh OUT_DIR [DRAFT_KERNEL DRAFT_MANIFEST SLUG_TAG DRAFT_NAME]
#
# With OUT_DIR alone it rebuilds the current candidates byte for byte (tests/test_build_candidates.py pins them):
#   OUT_DIR/exp083  scottmahony/arc3-dprime-r14a05-arcmap-draft-full            (base + ARC map + draft)
#   OUT_DIR/exp084  scottmahony/arc3-dprime-r14a05-harness4-percept-draft-full  (+ the harness bundle)
# both serving MTP session A v2's draft (kernel scottmahony/arc3-mtp-session-a; the manifest copy in
# kaggle/franzen/drafts/ pins its version). For another draft, pass its kernel, its pulled arc3-draft-manifest.json,
# a SLUG_TAG that replaces "draft" in both slugs (e.g. draft2, so the submitted notebooks stay untouched) and a
# DRAFT_NAME for the notes. Push each folder with scripts/push_eval.py: the push is the full-length public-25 save
# run (~2.3 GPU-h), and that version is what gets submitted (docs/SUBMITTING.md).
set -euo pipefail
OUT=${1:?usage: scripts/build_candidates.sh OUT_DIR [DRAFT_KERNEL DRAFT_MANIFEST SLUG_TAG DRAFT_NAME]}
DRAFT=${2:-scottmahony/arc3-mtp-session-a}
MANIFEST=${3:-kaggle/franzen/drafts/arc3-mtp-session-a.manifest.json}
TAG=${4:-draft}
NAME=${5:-MTP session A v2\'s fine-tuned draft}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY=${PYTHON:-.venv/bin/python}
P=kaggle/franzen/patches
COMMON=(--base dprime --full25 121 --input-fallback --wait-inputs 120)
SERVE=(--cfg MAXREQ=14 --cfg CUDAGRAPH_MAXBS=14 --cfg MAMBA_CACHE=84 --cfg SPEC_ACCEPT_SINGLE=0.5 --cfg SPEC_ACCEPT_ACC=0.5)

"$PY" scripts/build_franzen_nb.py "${COMMON[@]}" \
  --env ARC3_MAX_ACTIVE_STREAMS=14 --env ARC3_HTTP_RETRY_INITIAL_SECONDS=2400 "${SERVE[@]}" \
  --reap-kept kaggle/franzen/reap448_kept_experts.json \
  --patch "$P/ours-sandbox-timeout-keeps-work.patch" \
  --hot-tokens kaggle/franzen/hot_tokens_64k_arc.pt \
  --draft "$DRAFT" --draft-manifest "$MANIFEST" \
  --fail-fast --compact \
  --out "$OUT/exp083" --slug "arc3-dprime-r14a05-arcmap-$TAG-full" \
  --note "exp-083: exp-075 (the exp-074t candidate at full length) + ARC FR-Spec map + $NAME (relaxed acceptance 0.5/0.5 as the candidate) + first-request grace 2400 s"

"$PY" scripts/build_franzen_nb.py "${COMMON[@]}" \
  --env ARC3_MAX_ACTIVE_STREAMS=14 --env ARC3_HTTP_RETRY_INITIAL_SECONDS=2400 \
  --env-add OURS_BUDGET_METER=1 --env-add OURS_WIN_LEDGER=1 --env-add OURS_SEARCH_HELPER=1 \
  --env-add OURS_LEVEL_MEM=1 --env-add OURS_PERCEPTION=1 \
  "${SERVE[@]}" \
  --reap-kept kaggle/franzen/reap448_kept_experts.json --hot-tokens kaggle/franzen/hot_tokens_64k_arc.pt --fail-fast \
  --draft "$DRAFT" --draft-manifest "$MANIFEST" \
  --patch "$P/ours-sandbox-timeout-keeps-work.patch" --patch "$P/ours-02-budget-meter.patch" \
  --patch "$P/ours-04-search-helper.patch" --patch "$P/ours-03b-win-ledger-on-02-04.patch" \
  --patch "$P/ours-05-level-mem.patch" --patch "$P/ours-08b-perception-on-01-02-04-03b-05.patch" \
  --compact --out "$OUT/exp084" --slug "arc3-dprime-r14a05-harness4-percept-$TAG-full" \
  --note "exp-084: exp-081 (the bundle candidate: budget meter, search helper, win ledger, level mem, perception, ARC FR-Spec map, RESET not exposed) + $NAME + first-request grace 2400 s"
