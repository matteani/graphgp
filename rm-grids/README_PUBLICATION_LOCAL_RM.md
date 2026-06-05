# Publication-ready local RM reconstruction package

This update keeps the same mock-data / NIFTy reconstruction pipeline, but refreshes the plotting so the outputs are cleaner for slides or a paper draft.

Main changes:

- a cleaner **pulsar geometry** figure with the line-of-sight rays, reconstruction box/domain, and pulsar distance from the observer
- a publication-style **summary comparison** with point estimates and confidence intervals
- a direct panel showing **how much the div-free prior helps**
- cleaner legends placed outside the middle of the data panels
- simplified output set: geometry, summary, endpoint-scatter, RM-fit, plus one geometry-specific field plot

---

## Files to copy into your working folder

Copy the full folder contents into your local `rm-grids/` directory:

```bash
cd ~/Documents/MPA/graphgp/rm-grids
cp /mnt/data/rm_nifty_local_pubready/* .
```

If you want to keep the old files, copy into a fresh directory first.

---

## Re-make plots from reconstruction files you already have

### HEALPix × log-r

```bash
python 03_plot_compare_reconstructions.py \
  --divfree outputs/local_healpix_logr_recon_divfree_prior.npz \
  --standard outputs/local_healpix_logr_recon_standard_prior.npz \
  --outdir figures/local_healpix_logr_compare
```

### Cartesian

```bash
python 03_plot_compare_reconstructions.py \
  --divfree outputs/local_cartesian_recon_divfree_prior.npz \
  --standard outputs/local_cartesian_recon_standard_prior.npz \
  --outdir figures/local_cartesian_compare
```

---

## New main output figures

These are the ones I would use first in a meeting:

```text
00_pulsar_geometry_and_sightlines.png
01_prior_comparison_metrics.png
02_endpoint_sign_scatter.png
03_rm_fit.png
04_healpix_outer_shell_sky.png   # HEALPix only
04_cartesian_Bz_slice.png        # Cartesian only
```

---

## Recommended stronger local runs

Below are parameter sets that are noticeably stronger than the earlier toy runs, but still aimed at a modest local GPU.

## 1) HEALPix × log-r: stronger but still local-safe

This is the best first upgrade for demonstrating the benefit of the div-free prior.

```bash
NSIDE=4 \
N_RADIAL=20 \
R_MIN=0.12 \
R_MAX=0.90 \
N_PULSAR_DIRECTIONS=0 \
N_PULSAR_SHELLS=8 \
GRAPH_N0=128 \
GRAPH_K=10 \
ELL=0.20 \
N_RAY_STEPS=128 \
N_VI_ITER=2 \
N_VI_SAMPLES=3 \
SEED=123 \
bash run_local_healpix_logr_nifty.sh
```

What that means:

- `NSIDE=4` gives all **192 angular HEALPix directions**
- `N_PULSAR_SHELLS=8` gives **8 pulsar distances per direction**
- total sightlines = **192 × 8 = 1536**
- `N_RADIAL=20` increases the reconstruction grid in radius
- `N_VI_ITER=2`, `N_VI_SAMPLES=3` makes the inference more serious than a 1-iteration smoke test

If that is stable and you still have memory headroom, try:

```bash
NSIDE=6 \
N_RADIAL=18 \
R_MIN=0.12 \
R_MAX=0.90 \
N_PULSAR_DIRECTIONS=0 \
N_PULSAR_SHELLS=6 \
GRAPH_N0=160 \
GRAPH_K=12 \
ELL=0.20 \
N_RAY_STEPS=128 \
N_VI_ITER=2 \
N_VI_SAMPLES=3 \
SEED=123 \
bash run_local_healpix_logr_nifty.sh
```

That gives:

- `12 * NSIDE^2 = 432` angular directions
- `432 × 6 = 2592` sightlines

That is much denser and often gives a cleaner sign-recovery story.

---

## 2) Cartesian: stronger local comparison

```bash
N=22 \
N_DIRECTIONS=256 \
N_SHELLS=6 \
R_MIN=0.08 \
R_MAX=0.46 \
GRAPH_N0=160 \
GRAPH_K=12 \
ELL=0.10 \
N_RAY_STEPS=128 \
N_VI_ITER=2 \
N_VI_SAMPLES=3 \
SEED=42 \
bash run_local_cartesian_nifty.sh
```

If that is comfortable, try:

```bash
N=24 \
N_DIRECTIONS=320 \
N_SHELLS=6 \
GRAPH_N0=192 \
GRAPH_K=12 \
ELL=0.095 \
N_RAY_STEPS=128 \
N_VI_ITER=2 \
N_VI_SAMPLES=3 \
SEED=42 \
bash run_local_cartesian_nifty.sh
```

---

## How to change parameters from the command line

You do **not** edit the script each time.

You just prepend environment-variable assignments before `bash ...`.

Example:

```bash
NSIDE=6 N_RADIAL=18 N_PULSAR_SHELLS=6 N_VI_ITER=2 N_VI_SAMPLES=3 bash run_local_healpix_logr_nifty.sh
```

or

```bash
N=22 N_DIRECTIONS=256 N_SHELLS=6 N_VI_ITER=2 N_VI_SAMPLES=3 bash run_local_cartesian_nifty.sh
```

---

## Parameter meanings

### HEALPix × log-r

- `NSIDE`: angular HEALPix resolution; number of angular directions is `12 * NSIDE^2`
- `N_RADIAL`: number of radial shells in the reconstruction field
- `N_PULSAR_DIRECTIONS`: number of angular pulsar directions; `0` means use all HEALPix pixels
- `N_PULSAR_SHELLS`: number of pulsar distances per angular direction
- `R_MIN`, `R_MAX`: inner and outer pulsar distance range

### Cartesian

- `N`: linear side length of the reconstruction cube; total voxel count is `N^3`
- `N_DIRECTIONS`: number of pulsar line-of-sight directions
- `N_SHELLS`: number of pulsar distances per direction

### Shared GraphGP / inference settings

- `GRAPH_N0`: GraphGP dense-core size
- `GRAPH_K`: number of graph neighbors
- `ELL`: prior correlation length
- `N_RAY_STEPS`: quadrature points per RM line-of-sight integral
- `N_VI_ITER`: number of outer NIFTy VI iterations
- `N_VI_SAMPLES`: number of samples per VI iteration
- `SEED`: random seed

---

## If memory gets tight

Use the built-in memory-safe environment already present in the run scripts. If needed, reduce parameters in this order:

1. `GRAPH_N0`
2. `GRAPH_K`
3. `N_VI_SAMPLES`
4. `N_RADIAL` or `N`
5. `N_PULSAR_SHELLS`
6. `N_RAY_STEPS`

For HEALPix, if you get NaNs, try slightly safer geometry first:

```bash
R_MIN=0.14 ELL=0.22 bash run_local_healpix_logr_nifty.sh
```

---

## Interpreting the main comparison plot

`01_prior_comparison_metrics.png` now shows three things:

1. **endpoint sign accuracy** with confidence intervals
2. **accuracy gain** of the div-free prior over the standard prior
3. **reduced chi²** of the RM data-space fit

So the figure directly answers:

- does the div-free prior improve sign recovery?
- by how much?
- is the data fit still good?

