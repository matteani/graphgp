#!/bin/bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 PATH_TO_BENCHMARK [OUTPUT_DIRECTORY]" >&2
  exit 2
fi

benchmark_executable=$1
result_directory=${2:-graph-sharding-results}
gpu_counts=${GPU_COUNTS:-"1 2 4"}
localities=${LOCALITIES:-"0.0 0.5 0.9 1.0"}
node_counts=${NODE_COUNTS:-"1048576"}
level_counts=${LEVEL_COUNTS:-"16"}
seeds=${SEEDS:-"42"}
compute_counts=${COMPUTE_ITERS_LIST:-"0"}
mode=${MODE:-forward}
communities=${COMMUNITIES:-4}

mkdir -p "${result_directory}"

for node_count in ${node_counts}; do
  for level_count in ${level_counts}; do
    for gpu_count in ${gpu_counts}; do
      for locality in ${localities}; do
        for seed in ${seeds}; do
          for compute_iters in ${compute_counts}; do
            output_path="${result_directory}/${mode}-n${node_count}-l${level_count}-m${gpu_count}-q${locality}-s${seed}-c${compute_iters}.json"
            "${benchmark_executable}" \
              --nodes "${node_count}" \
              --parents "${PARENTS:-8}" \
              --levels "${level_count}" \
              --gpus "${gpu_count}" \
              --communities "${communities}" \
              --locality "${locality}" \
              --seed "${seed}" \
              --dtype "${DTYPE:-float}" \
              --mode "${mode}" \
              --strategy all \
              --warmup "${WARMUP:-5}" \
              --runs "${RUNS:-20}" \
              --evaluations-per-run "${EVALUATIONS_PER_RUN:-1}" \
              --learning-rate "${LEARNING_RATE:-0.01}" \
              --compute-iters "${compute_iters}" \
              --output "${output_path}"
          done
        done
      done
    done
  done
done
