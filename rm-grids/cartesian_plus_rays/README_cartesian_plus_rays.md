# Cartesian base grid + exact RM ray-node reconstruction

This package adds a new mock/reconstruction setup where the GraphGP model points are

```text
regular Cartesian base grid points
+ exact quadrature nodes along every RM line of sight
+ exact pulsar endpoint positions
```

The RM response then does **no spatial interpolation**.  Each RM quadrature node is itself a GraphGP model point, so the response is just a sparse selection/projection of the local vector field:

```math
\mathrm{RM}_p \approx \sum_q n_e(x_{pq})\,\Delta s_{pq}\,B(x_{pq})\cdot\hat n_p.
```

## Files

```text
01_make_mock_cartesian_plus_rays.py
02_reconstruct_rm_nifty_graphgp_embedded.py
03_plot_compare_reconstructions_plus_rays.py
run_local_cartesian_plus_rays_nifty.sh
submit_cartesian_plus_rays_compare.slurm
```

## Run locally

```bash
N=30 \
N_DIRECTIONS=384 \
N_SHELLS=6 \
N_RAY_STEPS=160 \
GRAPH_N0=256 \
GRAPH_K=14 \
ELL=0.09 \
N_VI_ITER=4 \
N_VI_SAMPLES=4 \
SEED=42 \
bash run_local_cartesian_plus_rays_nifty.sh
```

Outputs:

```text
outputs/local_cartesian_plus_rays_mock_divfree_truth.npz
outputs/local_cartesian_plus_rays_recon_divfree_prior.npz
outputs/local_cartesian_plus_rays_recon_standard_prior.npz
figures/local_cartesian_plus_rays_compare/
```

## Submit on Viper GPU

```bash
sbatch submit_cartesian_plus_rays_compare.slurm
```

Override parameters at submission:

```bash
sbatch --export=ALL,N=30,N_DIRECTIONS=384,N_SHELLS=6,N_RAY_STEPS=160,N_VI_ITER=4,N_VI_SAMPLES=4 submit_cartesian_plus_rays_compare.slurm
```

If your virtual environment has a different path:

```bash
sbatch --export=ALL,GRAPHGP_VENV=/path/to/venv submit_cartesian_plus_rays_compare.slurm
```

## Is this grid irregular?

Yes, globally it is a hybrid/irregular point cloud.  It contains a regular Cartesian base-grid subset, but the added line-of-sight quadrature nodes and pulsar endpoints are generally off-lattice.  Therefore the full GraphGP point set is no longer a tensor-product Cartesian grid.

The reconstruction script still saves the Cartesian base-grid subset separately:

```text
B_true_cartesian_base
B_est_cartesian_base
base_grid_indices
```

so Cartesian slice plots and finite-difference divergence diagnostics can still be evaluated on the structured subset.
