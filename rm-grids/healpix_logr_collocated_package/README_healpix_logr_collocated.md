# HEALPix/log-r shell-collocated RM reconstruction

This package sets up a HEALPix x log-r GraphGP reconstruction where the RM response uses existing grid points directly.

The GraphGP model points are

```text
x_{shell,pixel} = r_shell * n_hat_pixel
```

Pulsars are placed at selected HEALPix pixels and selected radial shells, so every pulsar endpoint is already a GraphGP point.  The RM integral is approximated as a shell-centred sum along the same HEALPix radial column:

```math
RM_p \approx \sum_{q \le q_p} n_e(r_q \hat n_p)\,\Delta r_q\,B(r_q \hat n_p)\cdot \hat n_p.
```

There is no spatial interpolation in this response.  The log-r radial shells themselves are the line-of-sight sample points.

## Why not add separate LOS points?

For a HEALPix/log-r grid, separate LOS points are unnecessary if the pulsar directions are HEALPix pixel directions and the pulsar distances are radial shell centres.  Adding separate ray nodes would turn the grid into a bigger irregular point cloud without giving much extra benefit.  If you need arbitrary pulsar distances or directions, then adding LOS/ray points is useful; for this controlled grid-aligned mock, the shell-collocated response is cleaner.

## Files

```text
01_make_mock_healpix_logr_collocated.py
05_component_recovery_healpix_logr.py
run_local_healpix_logr_collocated_nifty.sh
submit_healpix_logr_collocated_compare.slurm
```

The run scripts reuse your existing:

```text
02_reconstruct_rm_nifty_graphgp.py
03_plot_compare_reconstructions.py
rm_common.py
```

They will find these either in the same directory or in the parent directory.

## Local run

Conservative local run:

```bash
NSIDE=8 \
N_RADIAL=32 \
N_PULSAR_DIRECTIONS=384 \
N_PULSAR_SHELLS=6 \
GRAPH_N0=256 \
GRAPH_K=14 \
ELL=0.20 \
N_VI_ITER=2 \
N_VI_SAMPLES=3 \
SEED=123 \
bash run_local_healpix_logr_collocated_nifty.sh
```

This has

```text
12 * 8^2 * 32 = 24576 GraphGP points.
```

Equivalent-size Viper-scale run:

```bash
NSIDE=16 \
N_RADIAL=18 \
N_PULSAR_DIRECTIONS=384 \
N_PULSAR_SHELLS=6 \
GRAPH_N0=512 \
GRAPH_K=16 \
ELL=0.20 \
N_VI_ITER=4 \
N_VI_SAMPLES=4 \
SEED=123 \
bash run_local_healpix_logr_collocated_nifty.sh
```

This has

```text
12 * 16^2 * 18 = 55296 GraphGP points.
```

## Viper submission

```bash
sbatch submit_healpix_logr_collocated_compare.slurm
```

Override parameters:

```bash
sbatch --export=ALL,NSIDE=16,N_RADIAL=18,N_PULSAR_DIRECTIONS=384,N_PULSAR_SHELLS=6,N_VI_ITER=4,N_VI_SAMPLES=4 submit_healpix_logr_collocated_compare.slurm
```

If your venv path differs:

```bash
sbatch --export=ALL,GRAPHGP_VENV=/path/to/graphgp-venv submit_healpix_logr_collocated_compare.slurm
```

## Outputs

```text
figures/local_healpix_logr_collocated_compare/prior_comparison_summary.txt
figures/local_healpix_logr_collocated_compare/04_healpix_outer_shell_sky.png
figures/local_healpix_logr_collocated_compare/component_recovery/05_healpix_component_shell_scatter_maps.png
figures/local_healpix_logr_collocated_compare/component_recovery/06_healpix_component_summary.png
figures/local_healpix_logr_collocated_compare/component_recovery/07_healpix_component_radial_profiles.png
figures/local_healpix_logr_collocated_compare/component_recovery/component_recovery_healpix_summary.txt
```

## Is this an irregular grid?

It is not a Cartesian tensor-product grid.  It is a structured spherical point cloud: HEALPix angular pixels crossed with logarithmic radial shells.  For GraphGP, it is just a list of 3D points and the graph is built from Euclidean distances between those points.  Therefore it is best described as a structured irregular 3D point cloud.
