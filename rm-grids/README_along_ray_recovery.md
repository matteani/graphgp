# Along-ray B_parallel recovery diagnostic

This compares local line-of-sight magnetic-field recovery along the full RM ray, not only at the endpoint pulsar position.

For each pulsar ray,

```text
x(s) = x_obs + s n_hat, 0 <= s <= d_p
B_parallel(s) = B(x(s)) dot n_hat
```

The diagnostic evaluates `B_parallel(s)` for the truth, the div-free reconstruction, and the standard-vector reconstruction.

## Cartesian

```bash
python 04_along_ray_recovery.py \
  --divfree outputs/local_cartesian_recon_divfree_prior.npz \
  --standard outputs/local_cartesian_recon_standard_prior.npz \
  --outdir figures/local_cartesian_compare/along_ray \
  --n-ray-samples 128
```

or:

```bash
GEOM=cartesian N_RAY_SAMPLES=128 bash run_along_ray_recovery.sh
```

## HEALPix/log-r

```bash
python 04_along_ray_recovery.py \
  --divfree outputs/local_healpix_logr_recon_divfree_prior.npz \
  --standard outputs/local_healpix_logr_recon_standard_prior.npz \
  --outdir figures/local_healpix_logr_compare/along_ray \
  --n-ray-samples 128
```

or:

```bash
GEOM=healpix N_RAY_SAMPLES=128 bash run_along_ray_recovery.sh
```

## Outputs

```text
05_along_ray_recovery_summary.png
06_along_ray_gain_distribution.png
07_along_ray_example_profiles.png
along_ray_recovery_summary.txt
```

Key quantities:

```text
sign_accuracy_gain = div-free along-ray sign accuracy - standard along-ray sign accuracy
correlation_gain   = div-free correlation - standard correlation
nrmse_reduction    = standard normalized RMSE - div-free normalized RMSE
```

Positive values mean the div-free prior performs better.
