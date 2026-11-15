#!/bin/bash
# Retry wrapper for `hf download` — the download environment on this node
# has shown two real failure modes: hf-xet's fast-transfer backend hangs
# silently with no error and no recovery (fixed by HF_HUB_DISABLE_XET=1,
# see conversation log 2026-08-12), and even with xet disabled, individual
# large-file transfers can die mid-stream (RemoteProtocolError, "peer
# closed connection"). `hf download` resumes from committed progress on
# each restart, so retrying is safe and just picks up where it left off.
set -eo pipefail

REPO="$1"
shift
MAX_RETRIES=20
DELAY=30

export HF_HOME=/scratch/gpfs/MM4/apjanco/.cache/huggingface
export HF_HUB_DISABLE_XET=1

# Resolve `hf` explicitly. A non-interactive shell (nohup, cron, a hook)
# does not source the conda profile, so a bare `hf` is "command not found"
# — and because that exits non-zero like a real transfer failure, the retry
# loop below cheerfully burns all 20 attempts on it, 30s apart, reporting
# a network problem that never happened. Fail loudly instead.
HF_BIN="${HF_BIN:-/home/aj7878/.conda/envs/penelope/bin/hf}"
if [ ! -x "$HF_BIN" ]; then
    HF_BIN="$(command -v hf || true)"
fi
if [ -z "$HF_BIN" ]; then
    echo "ERROR: no 'hf' executable found. Set HF_BIN or activate the penelope env." >&2
    exit 127
fi
echo "using hf: $HF_BIN"

for i in $(seq 1 $MAX_RETRIES); do
  echo "=== attempt $i/$MAX_RETRIES: $HF_BIN download $REPO $* ==="
  if "$HF_BIN" download "$REPO" "$@"; then
    echo "=== $REPO: download complete ==="
    exit 0
  fi
  echo "=== $REPO: attempt $i failed, retrying in ${DELAY}s ==="
  sleep $DELAY
done

echo "=== $REPO: FAILED after $MAX_RETRIES attempts ==="
exit 1
