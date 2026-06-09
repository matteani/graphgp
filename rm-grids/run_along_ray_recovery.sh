#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   GEOM=cartesian bash run_along_ray_recovery.sh
#   GEOM=healpix bash run_along_ray_recovery.sh
# Optional overrides:
#   N_RAY_SAMPLES=160 MAX_RAYS=1000 N_EXAMPLES=8 GEOM=cartesian bash run_along_ray_recovery.sh

GEOM=${GEOM:-cartesian}
N_RAY_SAMPLES=${N_RAY_SAMPLES:-128}
MAX_RAYS=${MAX_RAYS:-0}
N_EXAMPLES=${N_EXAMPLES:-6}
SEED=${SEED:-0}

if [[ "$GEOM" == "cartesian" ]]; then
  DIVFREE=${DIVFREE:-outputs/local_cartesian_recon_divfree_prior.npz}
  STANDARD=${STANDARD:-outputs/local_cartesian_recon_standard_prior.npz}
  OUTDIR=${OUTDIR:-figures/local_cartesian_compare/along_ray}
elif [[ "$GEOM" == "healpix" || "$GEOM" == "healpix_logr" ]]; then
  DIVFREE=${DIVFREE:-outputs/local_healpix_logr_recon_divfree_prior.npz}
  STANDARD=${STANDARD:-outputs/local_healpix_logr_recon_standard_prior.npz}
  OUTDIR=${OUTDIR:-figures/local_healpix_logr_compare/along_ray}
else
  echo "Unknown GEOM=$GEOM. Use GEOM=cartesian or GEOM=healpix." >&2
  exit 1
fi

python 04_along_ray_recovery.py \
  --divfree "$DIVFREE" \
  --standard "$STANDARD" \
  --outdir "$OUTDIR" \
  --n-ray-samples "$N_RAY_SAMPLES" \
  --max-rays "$MAX_RAYS" \
  --n-examples "$N_EXAMPLES" \
  --seed "$SEED"
