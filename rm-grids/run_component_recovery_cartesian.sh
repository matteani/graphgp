#!/usr/bin/env bash
set -euo pipefail

DIVFREE=${DIVFREE:-outputs/local_cartesian_recon_divfree_prior.npz}
STANDARD=${STANDARD:-outputs/local_cartesian_recon_standard_prior.npz}
OUTDIR=${OUTDIR:-figures/local_cartesian_compare/component_recovery}
SLICE_INDEX=${SLICE_INDEX:--1}
MAX_SCATTER_POINTS=${MAX_SCATTER_POINTS:-4000}

python 05_component_recovery_cartesian.py \
  --divfree "$DIVFREE" \
  --standard "$STANDARD" \
  --outdir "$OUTDIR" \
  --slice-index "$SLICE_INDEX" \
  --max-scatter-points "$MAX_SCATTER_POINTS"
