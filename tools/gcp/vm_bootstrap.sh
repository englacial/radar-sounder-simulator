#!/usr/bin/env bash
# Runs ON the GPU VM: fetch bundle from GCS, set up env, re-simulate the
# requested runs, push results back. Usage: vm_bootstrap.sh [resim keys...]
#
# Hard-won notes (claude_notes/gpu_benchmark_findings.md):
# - `uv run` re-syncs to uv.lock and silently reverts a pip-installed GPU
#   jax; install the CUDA plugin AT the locked jax version and use
#   `uv run --no-sync` for every invocation.
# - GCS-copied runs/*.npz do not pass the tool's cache meta check on a
#   different machine: expect re-simulation of anything not listed here.
set -euo pipefail
BUCKET=gs://ice-infrastructure-soundersim
RESIM=${@:-"firn_N40"}

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

mkdir -p ~/soundersim && cd ~/soundersim
gcloud storage cp $BUCKET/bundle/repo.tar.gz /tmp/repo.tar.gz
tar xzf /tmp/repo.tar.gz
mkdir -p outputs/cache outputs/b26_comparison/runs
gcloud storage cp -r "$BUCKET/bundle/cache/*" outputs/cache/
gcloud storage cp $BUCKET/bundle/b26/run_config.json outputs/b26_comparison/
gcloud storage cp "$BUCKET/bundle/b26/runs/*.npz" outputs/b26_comparison/runs/ || true

uv sync -q
JAXV=$(uv run --no-sync python -c "import jax; print(jax.__version__)")
uv pip install -q "jax[cuda12]==$JAXV"
uv run --no-sync python -c "import jax; d=jax.devices(); print('devices:', d); assert d[0].platform=='gpu'"

for k in $RESIM; do rm -f outputs/b26_comparison/runs/$k.npz; done
/usr/bin/time -v uv run --no-sync python tools/run_b26_comparison.py --no-pilot \
  2>&1 | tee run.log

HOST=$(hostname)
gcloud storage cp run.log outputs/b26_comparison/budget_log.json \
  outputs/b26_comparison/metrics.json "$BUCKET/results/$HOST/" || true
gcloud storage cp "outputs/b26_comparison/runs/*.npz" \
  "$BUCKET/results/$HOST/runs/" || true
echo BOOTSTRAP_DONE
