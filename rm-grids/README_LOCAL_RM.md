# Local GraphGP + NIFTy RM reconstruction files

Run from this directory after activating the Python environment where `graphgp`, `jax`, and `nifty.re` are installed.

```bash
bash check_local_gpu_env.sh
bash run_local_smoke_nifty.sh
bash run_local_cartesian_nifty.sh
bash run_local_healpix_logr_nifty.sh
```

Important core files:

- `rm_common.py`: shared GraphGP, response-matrix, HEALPix/log-r, and NIFTy helper functions.
- `01_make_mock_regular_cartesian.py`: regular Cartesian mock RM data from a GraphGP vector-field truth.
- `01_make_mock_healpix_logr.py`: HEALPix × log-r mock RM data from a GraphGP vector-field truth.
- `02_reconstruct_rm_nifty_graphgp.py`: NIFTy.re VI reconstruction using either the div-free or standard vector prior.
- `03_plot_compare_reconstructions.py`: comparison plots for div-free versus standard prior.

Quick smoke test:

```bash
bash run_local_smoke_nifty.sh
```

A stronger Cartesian run:

```bash
N=36 N_DIRECTIONS=192 N_SHELLS=5 N_VI_ITER=2 N_VI_SAMPLES=4 bash run_local_cartesian_nifty.sh
```

A stronger HEALPix/log-r run:

```bash
NSIDE=8 N_RADIAL=24 N_PULSAR_DIRECTIONS=256 N_PULSAR_SHELLS=5 N_VI_ITER=2 N_VI_SAMPLES=4 bash run_local_healpix_logr_nifty.sh
```
