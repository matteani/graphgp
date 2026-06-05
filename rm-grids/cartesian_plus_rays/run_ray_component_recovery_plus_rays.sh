#!/usr/bin/env bash
set -euo pipefail

DIVFREE=${DIVFREE:-outputs/local_cartesian_plus_rays_recon_divfree_prior.npz}
STANDARD=${STANDARD:-outputs/local_cartesian_plus_rays_recon_standard_prior.npz}
OUTDIR=${OUTDIR:-figures/local_cartesian_plus_rays_compare/ray_components}
N_EXAMPLES=${N_EXAMPLES:-4}
MAX_SCATTER_POINTS=${MAX_SCATTER_POINTS:-9000}
SEED=${SEED:-0}

python 06_ray_component_recovery_plus_rays.py \
  --divfree "$DIVFREE" \
  --standard "$STANDARD" \
  --outdir "$OUTDIR" \
  --n-examples "$N_EXAMPLES" \
  --max-scatter-points "$MAX_SCATTER_POINTS" \
  --seed "$SEED"
