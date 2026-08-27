# Native CUDA graph sharding benchmark

This directory contains a standalone, single-process CUDA/NCCL benchmark for
comparing balanced random graph ownership, parent-affinity ownership, and
replicated ancestor closures. It does not change GraphGP's Python or JAX APIs.

The graph and communication plans are created once on the host. Timed work is
performed in device buffers: Philox draws, level-ordered graph updates, packed
NCCL peer exchange, and the final checksum all-reduce. Results are written as
JSON. Static setup time is reported separately.

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
  --warmup 5 \
  --runs 20 \
  --output graph-sharding.json
```

`--locality 0` samples parents uniformly from earlier levels. `--locality 1`
restricts every parent to the node's planted community. Each strategy uses the
same graph and device RNG mapping. The last measured output is compared against
a one-GPU CUDA reference whenever that reference passes the memory preflight.

For Raven, load the currently available CUDA, NCCL, and CMake modules before
building. The included `raven.sbatch` requests one task controlling all four
A100 GPUs. Adjust its account, time limit, output location, and executable path
for your project. Raven currently documents four A100 40 GB GPUs with NVLink 3
per GPU node.

To collect the strong-scaling/locality grid inside an allocation, use:

```bash
bash benchmarks/cuda_graph_sharding/sweep_raven.sh \
  build/cuda_graph_sharding/cuda_graph_sharding_bench results
```

The default Raven sweep covers `M={1,2,4}` and locality
`q={0,0.5,0.9,1}`. On a node with eight visible GPUs, set
`GPU_COUNTS="1 2 4 8"`. Other parameters can be overridden through the
uppercase environment variables used by the script.

## Metrics

The JSON report includes per-rank primary and stored node counts, imbalance,
cut edges, deduplicated values and bytes sent, message count, ancestor
replication, estimated allocation, checksum validation, raw timings, and
min/median/p95 throughput. The reported runtime is the maximum same-device CUDA
event duration across ranks.

The first implementation intentionally uses one ordered stream per GPU and a
pack/exchange/compute sequence at every level. Communication/compute overlap,
CUDA Graph capture, raw peer copies, and NCCL one-sided operations are outside
this baseline.
