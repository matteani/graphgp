#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/..:${PYTHONPATH:-}"

DIVFREE=${DIVFREE:-outputs/local_cartesian_plus_rays_recon_divfree_prior.npz}
STANDARD=${STANDARD:-outputs/local_cartesian_plus_rays_recon_standard_prior.npz}
OUTDIR=${OUTDIR:-figures/local_cartesian_plus_rays_compare/ray_shells}
N_BINS=${N_BINS:-8}
NBOOT=${NBOOT:-800}
SEED=${SEED:-0}

python 09_ray_shell_recovery_plus_rays.py \
  --divfree "$DIVFREE" \
  --standard "$STANDARD" \
  --outdir "$OUTDIR" \
  --n-bins "$N_BINS" \
  --nboot "$NBOOT" \
  --seed "$SEED"
