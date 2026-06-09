#!/bin/bash -l
#SBATCH --job-name=rm-cart-plus-rays
#SBATCH --partition=apu1
#SBATCH --constraint=apu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=110000
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs outputs figures/viper_cartesian_plus_rays_compare

module purge
module load python-waterboa/2025.06
module load rocm/7.0
module load jax/0.7.1

# Adjust this if your venv path is different on Viper.
source ${GRAPHGP_VENV:-$HOME/venvs/graphgp-venv}/bin/activate

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-4}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}

N=${N:-30}
N_DIRECTIONS=${N_DIRECTIONS:-384}
N_SHELLS=${N_SHELLS:-6}
R_MIN=${R_MIN:-0.08}
R_MAX=${R_MAX:-0.46}
GRAPH_N0=${GRAPH_N0:-256}
GRAPH_K=${GRAPH_K:-14}
ELL=${ELL:-0.09}
N_RAY_STEPS=${N_RAY_STEPS:-160}
RAY_START=${RAY_START:-0.02}
N_VI_ITER=${N_VI_ITER:-4}
N_VI_SAMPLES=${N_VI_SAMPLES:-4}
SEED=${SEED:-42}
PREFIX=${PREFIX:-viper_cartesian_plus_rays}
OUTDIR=${OUTDIR:-figures/viper_cartesian_plus_rays_compare}

python - <<'PY' | tee logs/${PREFIX}_env_${SLURM_JOB_ID}.log
import jax
import graphgp
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("graphgp import OK")
PY

python 01_make_mock_cartesian_plus_rays.py \
  --out outputs/${PREFIX}_mock_divfree_truth.npz \
  --truth-prior-kind divfree_rbf \
  --n "$N" \
  --n-directions "$N_DIRECTIONS" \
  --n-shells "$N_SHELLS" \
  --r-min "$R_MIN" \
  --r-max "$R_MAX" \
  --graph-n0 "$GRAPH_N0" \
  --graph-k "$GRAPH_K" \
  --ell "$ELL" \
  --n-ray-steps "$N_RAY_STEPS" \
  --ray-start "$RAY_START" \
  --rm-noise-frac 0.055 \
  --rm-noise-floor 0.012 \
  --seed "$SEED" | tee logs/${PREFIX}_01_mock_${SLURM_JOB_ID}.log

python 02_reconstruct_rm_nifty_graphgp_embedded.py \
  --data outputs/${PREFIX}_mock_divfree_truth.npz \
  --out outputs/${PREFIX}_recon_divfree_prior.npz \
  --prior-kind divfree_rbf \
  --n-vi-iterations "$N_VI_ITER" \
  --n-vi-samples "$N_VI_SAMPLES" \
  --seed 7 | tee logs/${PREFIX}_02_recon_divfree_${SLURM_JOB_ID}.log

python 02_reconstruct_rm_nifty_graphgp_embedded.py \
  --data outputs/${PREFIX}_mock_divfree_truth.npz \
  --out outputs/${PREFIX}_recon_standard_prior.npz \
  --prior-kind standard_vector_rbf \
  --n-vi-iterations "$N_VI_ITER" \
  --n-vi-samples "$N_VI_SAMPLES" \
  --seed 8 | tee logs/${PREFIX}_03_recon_standard_${SLURM_JOB_ID}.log

python 03_plot_compare_reconstructions_plus_rays.py \
  --divfree outputs/${PREFIX}_recon_divfree_prior.npz \
  --standard outputs/${PREFIX}_recon_standard_prior.npz \
  --outdir "$OUTDIR" | tee logs/${PREFIX}_04_plot_${SLURM_JOB_ID}.log

echo "Done. Summary: $OUTDIR/prior_comparison_summary.txt"
