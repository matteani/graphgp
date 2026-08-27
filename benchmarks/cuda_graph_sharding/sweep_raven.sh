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

mkdir -p "${result_directory}"

for gpu_count in ${gpu_counts}; do
  for locality in ${localities}; do
    output_path="${result_directory}/m${gpu_count}-q${locality}.json"
    "${benchmark_executable}" \
      --nodes "${NODES:-1048576}" \
      --parents "${PARENTS:-8}" \
      --levels "${LEVELS:-16}" \
      --gpus "${gpu_count}" \
      --communities "${gpu_count}" \
      --locality "${locality}" \
      --seed "${SEED:-42}" \
      --dtype "${DTYPE:-float}" \
      --strategy all \
      --warmup "${WARMUP:-5}" \
      --runs "${RUNS:-20}" \
      --output "${output_path}"
  done
done
