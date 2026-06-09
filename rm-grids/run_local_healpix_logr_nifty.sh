#!/usr/bin/env bash
set -euo pipefail

# Local-GPU HEALPix x log-r GraphGP + NIFTy.re reconstruction.
# Override any parameter from the command line, e.g.
#   NSIDE=6 N_RADIAL=18 N_PULSAR_SHELLS=6 N_VI_ITER=2 N_VI_SAMPLES=3 bash run_local_healpix_logr_nifty.sh

export TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR:-cuda_malloc_async}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-2}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-2}

NSIDE=${NSIDE:-4}
N_RADIAL=${N_RADIAL:-16}
R_MIN=${R_MIN:-0.12}
R_MAX=${R_MAX:-0.90}
N_PULSAR_DIRECTIONS=${N_PULSAR_DIRECTIONS:-0}   # 0 = all HEALPix pixels
N_PULSAR_SHELLS=${N_PULSAR_SHELLS:-6}
GRAPH_N0=${GRAPH_N0:-128}
GRAPH_K=${GRAPH_K:-10}
ELL=${ELL:-0.20}
N_RAY_STEPS=${N_RAY_STEPS:-96}
N_VI_ITER=${N_VI_ITER:-1}
N_VI_SAMPLES=${N_VI_SAMPLES:-2}
SEED=${SEED:-123}

mkdir -p logs outputs figures/local_healpix_logr_compare

python - <<'PY' | tee logs/local_healpix_env.log
import jax
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
PY

python 01_make_mock_healpix_logr.py \
  --out outputs/local_healpix_logr_mock_divfree_truth.npz \
  --truth-prior-kind divfree_rbf \
  --nside "$NSIDE" \
  --n-radial "$N_RADIAL" \
  --r-min "$R_MIN" \
  --r-max "$R_MAX" \
  --n-pulsar-directions "$N_PULSAR_DIRECTIONS" \
  --n-pulsar-shells "$N_PULSAR_SHELLS" \
  --graph-n0 "$GRAPH_N0" \
  --graph-k "$GRAPH_K" \
  --ell "$ELL" \
  --n-ray-steps "$N_RAY_STEPS" \
  --rm-noise-frac 0.055 \
  --rm-noise-floor 0.010 \
  --seed "$SEED" | tee logs/local_healpix_01_mock.log

python 02_reconstruct_rm_nifty_graphgp.py \
  --data outputs/local_healpix_logr_mock_divfree_truth.npz \
  --out outputs/local_healpix_logr_recon_divfree_prior.npz \
  --prior-kind divfree_rbf \
  --n-vi-iterations "$N_VI_ITER" \
  --n-vi-samples "$N_VI_SAMPLES" \
  --seed 17 | tee logs/local_healpix_02_recon_divfree.log

python 02_reconstruct_rm_nifty_graphgp.py \
  --data outputs/local_healpix_logr_mock_divfree_truth.npz \
  --out outputs/local_healpix_logr_recon_standard_prior.npz \
  --prior-kind standard_vector_rbf \
  --n-vi-iterations "$N_VI_ITER" \
  --n-vi-samples "$N_VI_SAMPLES" \
  --seed 18 | tee logs/local_healpix_03_recon_standard.log

python 03_plot_compare_reconstructions.py \
  --divfree outputs/local_healpix_logr_recon_divfree_prior.npz \
  --standard outputs/local_healpix_logr_recon_standard_prior.npz \
  --outdir figures/local_healpix_logr_compare | tee logs/local_healpix_04_plot.log

echo "Done. Main outputs:"
echo "  figures/local_healpix_logr_compare/00_pulsar_geometry_and_sightlines.png"
echo "  figures/local_healpix_logr_compare/01_prior_comparison_metrics.png"
echo "  figures/local_healpix_logr_compare/02_endpoint_sign_scatter.png"
echo "  figures/local_healpix_logr_compare/03_rm_fit.png"
echo "  figures/local_healpix_logr_compare/04_healpix_outer_shell_sky.png"
