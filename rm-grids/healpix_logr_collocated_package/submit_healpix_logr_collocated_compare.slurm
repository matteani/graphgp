#!/bin/bash -l
#SBATCH --job-name=rm-healpix-colloc
#SBATCH --partition=apu1
#SBATCH --constraint=apu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=110000
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

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

mkdir -p logs outputs figures/viper_healpix_logr_collocated_compare

module purge
module load python-waterboa/2025.06
module load rocm/7.0
module load jax/0.7.1

source ${GRAPHGP_VENV:-$HOME/venvs/graphgp-venv}/bin/activate

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-4}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}

# About 55k model points, comparable to N=30 Cartesian-plus-rays with 384 directions and 64 ray nodes.
NSIDE=${NSIDE:-16}
N_RADIAL=${N_RADIAL:-18}
R_MIN=${R_MIN:-0.12}
R_MAX=${R_MAX:-0.90}
N_PULSAR_DIRECTIONS=${N_PULSAR_DIRECTIONS:-384}
N_PULSAR_SHELLS=${N_PULSAR_SHELLS:-6}
GRAPH_N0=${GRAPH_N0:-512}
GRAPH_K=${GRAPH_K:-16}
ELL=${ELL:-0.20}
N_VI_ITER=${N_VI_ITER:-4}
N_VI_SAMPLES=${N_VI_SAMPLES:-4}
SEED=${SEED:-123}
SHELL_INDEX=${SHELL_INDEX:--1}
PREFIX=${PREFIX:-viper_healpix_logr_collocated}
OUTDIR=${OUTDIR:-figures/viper_healpix_logr_collocated_compare}

python - <<'PY' | tee logs/${PREFIX}_env_${SLURM_JOB_ID}.log
import jax
import graphgp
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("graphgp import OK")
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
  --seed "$SEED" | tee logs/${PREFIX}_01_mock_${SLURM_JOB_ID}.log

python "$RECON" \
  --data outputs/${PREFIX}_mock_divfree_truth.npz \
  --out outputs/${PREFIX}_recon_divfree_prior.npz \
  --prior-kind divfree_rbf \
  --n-vi-iterations "$N_VI_ITER" \
  --n-vi-samples "$N_VI_SAMPLES" \
  --seed 17 | tee logs/${PREFIX}_02_recon_divfree_${SLURM_JOB_ID}.log

python "$RECON" \
  --data outputs/${PREFIX}_mock_divfree_truth.npz \
  --out outputs/${PREFIX}_recon_standard_prior.npz \
  --prior-kind standard_vector_rbf \
  --n-vi-iterations "$N_VI_ITER" \
  --n-vi-samples "$N_VI_SAMPLES" \
  --seed 18 | tee logs/${PREFIX}_03_recon_standard_${SLURM_JOB_ID}.log

python "$PLOT_COMPARE" \
  --divfree outputs/${PREFIX}_recon_divfree_prior.npz \
  --standard outputs/${PREFIX}_recon_standard_prior.npz \
  --outdir "$OUTDIR" | tee logs/${PREFIX}_04_plot_compare_${SLURM_JOB_ID}.log

python "$PLOT_COMPONENTS" \
  --divfree outputs/${PREFIX}_recon_divfree_prior.npz \
  --standard outputs/${PREFIX}_recon_standard_prior.npz \
  --outdir "${OUTDIR}/component_recovery" \
  --shell-index "$SHELL_INDEX" | tee logs/${PREFIX}_05_component_recovery_${SLURM_JOB_ID}.log

echo "Done. Summary: ${OUTDIR}/prior_comparison_summary.txt"
echo "Component summary: ${OUTDIR}/component_recovery/component_recovery_healpix_summary.txt"
