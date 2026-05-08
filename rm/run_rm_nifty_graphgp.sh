#!/usr/bin/env bash
set -euo pipefail

python 01_make_mock_rm_graphgp.py \
  --out rm_graphgp_mock.npz \
  --n 30 \
  --n-pulsars 192 \
  --graph-n0 512 \
  --graph-k 18 \
  --ell 0.085 \
  --n-ray-steps 120

python 02_reconstruct_rm_nifty_graphgp.py \
  --data rm_graphgp_mock.npz \
  --out rm_nifty_reconstruction.npz \
  --n-vi-iterations 1 \
  --n-vi-samples 4 \
  --verbosity 1

python 03_plot_rm_results.py \
  --input rm_nifty_reconstruction.npz \
  --outdir rm_figures
