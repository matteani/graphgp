# Updated local RM reconstruction scripts

This package adds presentation-ready diagnostics for the same mock RM data reconstructed with two priors:

- pulsar geometry and line-of-sight ray plot
- sign-accuracy bars with binomial intervals, without labelling the error-bar method on the figure
- paired endpoint sign outcome counts
- sign accuracy versus true line-of-sight field strength
- existing RM fit, endpoint scatter, and map/slice plots

## Copy into your working folder

```bash
cd ~/Documents/MPA/graphgp/rm-grids
cp /path/to/rm_nifty_local_update/* .
```

## Plot again using outputs you already have

```bash
python 03_plot_compare_reconstructions.py \
  --divfree outputs/local_healpix_logr_recon_divfree_prior.npz \
  --standard outputs/local_healpix_logr_recon_standard_prior.npz \
  --outdir figures/local_healpix_logr_compare
```

New/updated figures:

```text
00_pulsar_geometry_and_sightlines.png
01_prior_comparison_metrics.png
02_endpoint_sign_scatter.png
03_rm_fit.png
04_healpix_outer_shell_sky.png   # HEALPix only
04_cartesian_Bz_slice.png         # Cartesian only
05_paired_sign_outcomes.png
06_accuracy_vs_signal_strength.png
```

## Run a stronger HEALPix x log-r case locally

Start with this. It increases the number of pulsars relative to the earlier safe run but keeps the GraphGP point count moderate:

```bash
NSIDE=4 \
N_RADIAL=16 \
R_MIN=0.12 \
R_MAX=0.90 \
N_PULSAR_DIRECTIONS=0 \
N_PULSAR_SHELLS=6 \
GRAPH_N0=128 \
GRAPH_K=10 \
ELL=0.20 \
N_RAY_STEPS=96 \
N_VI_ITER=1 \
N_VI_SAMPLES=2 \
bash run_local_healpix_logr_nifty.sh
```

This gives all 192 HEALPix angular pixels and 6 radial pulsar shells, i.e. about 1152 RM sightlines.

If memory is okay and you want a more ambitious local run:

```bash
NSIDE=8 \
N_RADIAL=10 \
R_MIN=0.14 \
R_MAX=0.90 \
N_PULSAR_DIRECTIONS=0 \
N_PULSAR_SHELLS=3 \
GRAPH_N0=256 \
GRAPH_K=10 \
ELL=0.22 \
N_RAY_STEPS=72 \
N_VI_ITER=1 \
N_VI_SAMPLES=2 \
bash run_local_healpix_logr_nifty.sh
```

That uses all 768 HEALPix angular directions and 3 radial shells, i.e. about 2304 RM sightlines.

## Run a stronger Cartesian case locally

```bash
N=24 \
N_DIRECTIONS=192 \
N_SHELLS=5 \
R_MIN=0.08 \
R_MAX=0.46 \
GRAPH_N0=128 \
GRAPH_K=10 \
ELL=0.10 \
N_RAY_STEPS=96 \
N_VI_ITER=1 \
N_VI_SAMPLES=2 \
bash run_local_cartesian_nifty.sh
```

If this is stable, try:

```bash
N=26 \
N_DIRECTIONS=256 \
N_SHELLS=5 \
GRAPH_N0=192 \
GRAPH_K=10 \
ELL=0.095 \
N_RAY_STEPS=96 \
N_VI_ITER=1 \
N_VI_SAMPLES=2 \
bash run_local_cartesian_nifty.sh
```

## Parameter meanings

- `NSIDE`: angular HEALPix resolution. Number of angular pixels is `12*NSIDE^2`.
- `N_RADIAL`: number of log-r grid shells in the reconstruction field.
- `N_PULSAR_DIRECTIONS`: number of angular directions used for pulsars. `0` means all HEALPix pixels.
- `N_PULSAR_SHELLS`: number of pulsar distances per angular direction.
- `N`: Cartesian grid side length. Total vector-field grid points are `N^3`.
- `N_DIRECTIONS`: Cartesian pulsar angular directions.
- `N_SHELLS`: Cartesian pulsar distances per direction.
- `GRAPH_N0`, `GRAPH_K`: GraphGP initial dense block size and neighbor count.
- `ELL`: correlation length of the magnetic-field prior.
- `N_RAY_STEPS`: quadrature samples per RM sightline.
- `N_VI_ITER`, `N_VI_SAMPLES`: NIFTy variational inference work. Increase after geometry and plots look good.

## If memory fails

Use:

```bash
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
```

Then reduce, in this order:

1. `GRAPH_N0`
2. `GRAPH_K`
3. `N_RADIAL` or `N`
4. `N_VI_SAMPLES`
5. `N_PULSAR_SHELLS`

## If HEALPix gives NaNs

Increase the inner radius or the stabilizer:

```bash
R_MIN=0.14 ELL=0.22 bash run_local_healpix_logr_nifty.sh
```

or edit `rm_common.py` and change the covariance jitter from `1e-5` to `1e-4`.
