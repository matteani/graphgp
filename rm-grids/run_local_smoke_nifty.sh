#!/usr/bin/env bash
set -euo pipefail

# Tiny end-to-end GraphGP + NIFTy.re reconstruction.
# Run this from the folder containing:
#   01_make_mock_regular_cartesian.py
#   02_reconstruct_rm_nifty_graphgp.py
#   03_plot_compare_reconstructions.py
#   rm_common.py

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-4}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}

mkdir -p logs outputs figures/local_smoke_regular_compare

python - <<'PY' | tee logs/local_smoke_env.log
import jax
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
PY

python 01_make_mock_regular_cartesian.py \
  --out outputs/local_smoke_mock_regular.npz \
  --truth-prior-kind divfree_rbf \
  --n 20 \
  --n-directions 32 \
  --n-shells 3 \
  --r-min 0.10 \
  --r-max 0.42 \
  --graph-n0 128 \
  --graph-k 10 \
  --ell 0.085 \
  --n-ray-steps 80 \
  --rm-noise-frac 0.055 \
  --rm-noise-floor 0.012 \
  --seed 101 | tee logs/local_smoke_01_mock.log

python 02_reconstruct_rm_nifty_graphgp.py \
  --data outputs/local_smoke_mock_regular.npz \
  --out outputs/local_smoke_recon_divfree.npz \
  --prior-kind divfree_rbf \
  --n-vi-iterations 1 \
  --n-vi-samples 2 \
  --seed 102 | tee logs/local_smoke_02_recon_divfree.log

python 02_reconstruct_rm_nifty_graphgp.py \
  --data outputs/local_smoke_mock_regular.npz \
  --out outputs/local_smoke_recon_standard.npz \
  --prior-kind standard_vector_rbf \
  --n-vi-iterations 1 \
  --n-vi-samples 2 \
  --seed 103 | tee logs/local_smoke_03_recon_standard.log

python 03_plot_compare_reconstructions.py \
  --divfree outputs/local_smoke_recon_divfree.npz \
  --standard outputs/local_smoke_recon_standard.npz \
  --outdir figures/local_smoke_regular_compare | tee logs/local_smoke_04_plot.log

echo "Done. Open: figures/local_smoke_regular_compare"
