# Local GPU GraphGP + NIFTy RM reconstruction wrappers

These are wrapper scripts for running the same RM mock-reconstruction idea locally while the Viper jobs wait.

They assume you run from the folder containing the Python files:

- `01_make_mock_regular_cartesian.py`
- `01_make_mock_healpix_logr.py`
- `02_reconstruct_rm_nifty_graphgp.py`
- `03_plot_compare_reconstructions.py`
- `rm_common.py`

## What is real NIFTy and what is toy?

These local wrapper scripts call `02_reconstruct_rm_nifty_graphgp.py`, so they run `nifty.re.optimize_kl` and use GraphGP as the prior model.

The separate `toy_rm_plots` package does **not** run NIFTy. It is only a fast visual proxy.

## Truth/prior setup

The mock data are generated from a `divfree_rbf` GraphGP truth. The reconstructions use the same data twice:

1. `divfree_rbf`: same kernel family, same fixed `ell` and `sigma2` as the truth.
2. `standard_vector_rbf`: same fixed `ell` and `sigma2`, but no divergence-free vector coupling.

So the div-free reconstruction is an oracle-prior controlled experiment: the kernel class and hyperparameters match the truth. The standard prior is matched in scale/amplitude but lacks the physical constraint.

## Run order

From your local RM scripts folder:

```bash
source /path/to/your/graphgp-venv/bin/activate
bash check_local_gpu_env.sh
bash run_local_smoke_nifty.sh
bash run_local_cartesian_nifty.sh
bash run_local_healpix_logr_nifty.sh
```

For a stronger local Cartesian run:

```bash
N=36 N_DIRECTIONS=192 N_SHELLS=5 N_VI_ITER=3 bash run_local_cartesian_nifty.sh
```

For a stronger local HEALPix run:

```bash
NSIDE=8 N_RADIAL=24 N_PULSAR_DIRECTIONS=256 N_PULSAR_SHELLS=5 N_VI_ITER=2 bash run_local_healpix_logr_nifty.sh
```

If your local GPU is small, keep the defaults.

## Main outputs

Cartesian:

- `figures/local_cartesian_compare/prior_comparison_summary.txt`
- `figures/local_cartesian_compare/01_prior_comparison_metrics.png`
- `figures/local_cartesian_compare/02_endpoint_sign_scatter.png`
- `figures/local_cartesian_compare/03_rm_fit.png`
- `figures/local_cartesian_compare/04_cartesian_Bz_slice.png`

HEALPix x log-r:

- `figures/local_healpix_logr_compare/prior_comparison_summary.txt`
- `figures/local_healpix_logr_compare/01_prior_comparison_metrics.png`
- `figures/local_healpix_logr_compare/02_endpoint_sign_scatter.png`
- `figures/local_healpix_logr_compare/03_rm_fit.png`
- `figures/local_healpix_logr_compare/04_healpix_outer_shell_sky.png`
