# Native CUDA graph sharding benchmark

This directory contains a standalone, single-process CUDA/NCCL benchmark for
comparing balanced random graph ownership, parent-affinity ownership, and
replicated ancestor closures. It does not change GraphGP's Python or JAX APIs.

The graph and communication plans are created once on the host. Timed work is
performed in device buffers: Philox draws, level-ordered graph updates, packed
NCCL peer exchange, and the final checksum all-reduce. Results are written as
JSON. Static setup time is reported separately.

The optional iterative mode keeps graph plans and mutable state resident on
the GPUs. It compares parent affinity with ancestor closure using

```text
y[i] = x[i] + mean(y[parent])
loss = sum(y[i]^2) / (2 N)
```

Only primary owners update `x`. Parent affinity exchanges forward values and
returns remote-parent adjoints after each reverse level. Ancestor closure first
synchronizes primary latents to replicas, evaluates and differentiates its
closure locally with loss seeds only on primary outputs, then reduces partial
latent gradients to primary owners. In iterative mode, `--strategy all` runs
only these two strategies; random-balanced iterative execution is intentionally
left out of the first milestone. Latents are initialized once with device
Philox; each iteration begins by applying the gradient produced by the previous
iteration (the initial gradient is zero).

## Build

On an NVIDIA system with CMake, CUDA, and NCCL available:

```bash
cmake -S benchmarks/cuda_graph_sharding \
      -B build/cuda_graph_sharding \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build/cuda_graph_sharding -j
ctest --test-dir build/cuda_graph_sharding --output-on-failure
```

On an allocated GPU node, add `-DGRAPH_SHARDING_ENABLE_GPU_TESTS=ON` when
configuring to register one- and two-GPU forward/iterative validation tests in
CTest.

Set `NCCL_ROOT` or `CMAKE_PREFIX_PATH` if NCCL is installed outside the default
search paths. The default architecture is `sm_80` for Raven's A100 GPUs; pass a
different `CMAKE_CUDA_ARCHITECTURES` value for another cluster.

Without CUDA, configuring with `-DGRAPH_SHARDING_ENABLE_CUDA=OFF` builds only
the host planner tests.

## Run

CUDA-visible device IDs become NCCL ranks `0..M-1`:

```bash
srun build/cuda_graph_sharding/cuda_graph_sharding_bench \
  --nodes 1048576 \
  --parents 8 \
  --levels 16 \
  --gpus 4 \
  --communities 4 \
  --locality 0.5 \
  --strategy all \
  --mode forward \
  --warmup 5 \
  --runs 20 \
  --output graph-sharding.json
```

`--locality 0` samples parents uniformly from earlier levels. `--locality 1`
restricts every parent to the node's planted community. Each strategy uses the
same graph and device RNG mapping. The last measured output is compared against
a one-GPU CUDA reference whenever that reference passes the memory preflight.

For the smallest iterative comparison:

```bash
srun build/cuda_graph_sharding/cuda_graph_sharding_bench \
  --nodes 1048576 --parents 8 --levels 16 \
  --gpus 4 --communities 4 --locality 0.5 \
  --mode iterative --strategy all \
  --warmup 5 --runs 20 --evaluations-per-run 100 \
  --learning-rate 0.01 --output iterative.json
```

`--evaluations-per-run` executes several complete iterations inside one outer
CUDA-event timing interval and reports both batch and per-iteration times. It
reduces measurement noise but does not increase message sizes.
`--compute-iters` adds a differentiable affine transform at every node in
forward and its derivative in backward, providing a controlled way to find the
point where closure recomputation stops winning. Zero preserves the minimal
graph kernel.

For Raven, load the currently available CUDA, NCCL, and CMake modules before
building. The included `raven.sbatch` requests one task controlling all four
A100 GPUs. Adjust its account, time limit, output location, and executable path
for your project. Raven currently documents four A100 40 GB GPUs with NVLink 3
per GPU node.

For an initial iterative job using the batch script:

```bash
MODE=iterative EVALUATIONS_PER_RUN=100 sbatch benchmarks/cuda_graph_sharding/raven.sbatch
```

To collect the strong-scaling/locality grid inside an allocation, use:

```bash
bash benchmarks/cuda_graph_sharding/sweep_raven.sh \
  build/cuda_graph_sharding/cuda_graph_sharding_bench results
```

The default Raven sweep covers `M={1,2,4}` and locality
`q={0,0.5,0.9,1}` while holding the planted community count fixed at four, so
strong-scaling runs use the same graph distribution. On a node with eight
visible GPUs, set `GPU_COUNTS="1 2 4 8"`. Useful staged sweeps are:

```bash
# Synchronization depth and payload scaling.
MODE=iterative NODE_COUNTS="1048576 4194304 16777216" \
LEVEL_COUNTS="4 16 64" LOCALITIES="0.5" SEEDS="41 42 43" \
EVALUATIONS_PER_RUN=10 bash benchmarks/cuda_graph_sharding/sweep_raven.sh \
  build/cuda_graph_sharding/cuda_graph_sharding_bench results/scale

# Communication/computation crossover after choosing a practical N and L.
MODE=iterative NODE_COUNTS="16777216" LEVEL_COUNTS="16" \
COMPUTE_ITERS_LIST="0 4 16 64" SEEDS="41 42 43" \
bash benchmarks/cuda_graph_sharding/sweep_raven.sh \
  build/cuda_graph_sharding/cuda_graph_sharding_bench results/compute
```

The script accepts `GPU_COUNTS`, `LOCALITIES`, `NODE_COUNTS`, `LEVEL_COUNTS`,
`SEEDS`, `COMPUTE_ITERS_LIST`, `MODE`, `COMMUNITIES`, `PARENTS`, `DTYPE`,
`WARMUP`, `RUNS`, `EVALUATIONS_PER_RUN`, and `LEARNING_RATE`.

## Metrics

The JSON report includes per-rank primary and stored node counts, imbalance,
cut edges, deduplicated values and bytes sent, message count, ancestor
replication, estimated allocation, checksum validation, raw timings, and
min/median/p95 throughput. The reported runtime is the maximum same-device CUDA
event duration across ranks.

Iterative JSON additionally reports per-iteration latent-update, latent-sync,
forward, backward-compute, gradient-sync, and total timings. Communication
values, bytes, and messages are split by phase. Because all benchmark buffers
are allocated before iteration zero and no timed allocation occurs,
`peak_benchmark_bytes_per_rank` is the compiled persistent allocation. CUDA and
NCCL may reserve additional internal memory that is not attributed to the
benchmark. In iterative mode, `objective` and the compatibility field
`checksum` both contain the final quadratic loss. Communication accounting
describes graph-state traffic and excludes the final scalar reporting
all-reduce. Each phase is the maximum accumulated same-device event time over
ranks; because the critical rank can differ by phase, phase maxima need not sum
exactly to the independently measured total. Iterative total timing spans the
five reported execution phases; calculation of the final reporting objective
is outside that interval.

When the installed toolkit exposes the CMake `CUDA::nvtx3` target, the binary
also emits NVTX ranges. A larger run can be inspected with:

```bash
nsys profile --trace=cuda,nvtx,nccl --sample=none \
  build/cuda_graph_sharding/cuda_graph_sharding_bench \
  --mode iterative --strategy all --gpus 4 --nodes 16777216 \
  --parents 8 --levels 16 --communities 4 --locality 0.5 \
  --warmup 1 --runs 2 --evaluations-per-run 10
```

The first implementation intentionally uses one ordered stream per GPU and a
pack/exchange/compute sequence at every level. Communication/compute overlap,
CUDA Graph capture, raw peer copies, alternative optimizers, nonlinear graph
models, and NCCL one-sided operations are outside this baseline.
