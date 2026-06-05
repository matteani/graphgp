#!/usr/bin/env bash
set -euo pipefail

# Local-GPU HEALPix x log-r shell-collocated GraphGP + NIFTy.re reconstruction.
# No interpolation is used in the RM response: pulsars sit on radial shell centres
# and RM sums existing shell points along the HEALPix radial column.
#
# Override parameters, e.g.
#   NSIDE=16 N_RADIAL=18 N_PULSAR_DIRECTIONS=384 N_PULSAR_SHELLS=6 N_VI_ITER=4 N_VI_SAMPLES=4 bash run_local_healpix_logr_collocated_nifty.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/..:${PYTHONPATH:-}"

find_script() {
  local name="$1"
  if [[ -f "${SCRIPT_DIR}/${name}" ]]; then
    printf '%s\n' "${SCRIPT_DIR}/${name}"
  elif [[ -f "${SCRIPT_DIR}/../${name}" ]]; then
    printf '%s\n' "${SCRIPT_DIR}/../${name}"
  else
    echo "Could not find ${name} in ${SCRIPT_DIR} or ${SCRIPT_DIR}/.." >&2
    exit 1
  fi
}

MAKE_MOCK="$(find_script 01_make_mock_healpix_logr_collocated.py)"
RECON="$(find_script 02_reconstruct_rm_nifty_graphgp.py)"
PLOT_COMPARE="$(find_script 03_plot_compare_reconstructions.py)"
PLOT_COMPONENTS="$(find_script 05_component_recovery_healpix_logr.py)"

export TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR:-cuda_malloc_async}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-2}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-2}

NSIDE=${NSIDE:-8}
N_RADIAL=${N_RADIAL:-32}
R_MIN=${R_MIN:-0.12}
R_MAX=${R_MAX:-0.90}
N_PULSAR_DIRECTIONS=${N_PULSAR_DIRECTIONS:-384}
N_PULSAR_SHELLS=${N_PULSAR_SHELLS:-6}
GRAPH_N0=${GRAPH_N0:-256}
GRAPH_K=${GRAPH_K:-14}
ELL=${ELL:-0.20}
N_VI_ITER=${N_VI_ITER:-2}
N_VI_SAMPLES=${N_VI_SAMPLES:-3}
SEED=${SEED:-123}
SHELL_INDEX=${SHELL_INDEX:--1}
PREFIX=${PREFIX:-local_healpix_logr_collocated}
OUTDIR=${OUTDIR:-figures/local_healpix_logr_collocated_compare}

mkdir -p logs outputs "$OUTDIR"

python - <<'PY' | tee logs/${PREFIX}_env.log
import jax
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
PY

python "$MAKE_MOCK" \
  --out outputs/${PREFIX}_mock_divfree_truth.npz \
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
  --rm-noise-frac 0.055 \
  --rm-noise-floor 0.010 \
  --seed "$SEED" | tee logs/${PREFIX}_01_mock.log

python "$RECON" \
  --data outputs/${PREFIX}_mock_divfree_truth.npz \
  --out outputs/${PREFIX}_recon_divfree_prior.npz \
  --prior-kind divfree_rbf \
  --n-vi-iterations "$N_VI_ITER" \
  --n-vi-samples "$N_VI_SAMPLES" \
  --seed 17 | tee logs/${PREFIX}_02_recon_divfree.log

python "$RECON" \
  --data outputs/${PREFIX}_mock_divfree_truth.npz \
  --out outputs/${PREFIX}_recon_standard_prior.npz \
  --prior-kind standard_vector_rbf \
  --n-vi-iterations "$N_VI_ITER" \
  --n-vi-samples "$N_VI_SAMPLES" \
  --seed 18 | tee logs/${PREFIX}_03_recon_standard.log

python "$PLOT_COMPARE" \
  --divfree outputs/${PREFIX}_recon_divfree_prior.npz \
  --standard outputs/${PREFIX}_recon_standard_prior.npz \
  --outdir "$OUTDIR" | tee logs/${PREFIX}_04_plot_compare.log

python "$PLOT_COMPONENTS" \
  --divfree outputs/${PREFIX}_recon_divfree_prior.npz \
  --standard outputs/${PREFIX}_recon_standard_prior.npz \
  --outdir "${OUTDIR}/component_recovery" \
  --shell-index "$SHELL_INDEX" | tee logs/${PREFIX}_05_component_recovery.log

echo "Done. Main outputs:"
echo "  ${OUTDIR}/prior_comparison_summary.txt"
echo "  ${OUTDIR}/04_healpix_outer_shell_sky.png"
echo "  ${OUTDIR}/component_recovery/05_healpix_component_shell_scatter_maps.png"
echo "  ${OUTDIR}/component_recovery/06_healpix_component_summary.png"
echo "  ${OUTDIR}/component_recovery/07_healpix_component_radial_profiles.png"
