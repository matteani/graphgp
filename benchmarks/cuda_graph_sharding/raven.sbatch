#!/bin/bash -l
#SBATCH --job-name=graph-sharding
#SBATCH --output=graph-sharding.%j.out
#SBATCH --error=graph-sharding.%j.err
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matteani@mpa-garching.mpg.de
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=500000
#SBATCH --time=00:30:00

set -euo pipefail

module purge
module load cuda/13.2 nccl/2.30.4

benchmark_root="${SLURM_SUBMIT_DIR}/build/cuda_graph_sharding"

srun "${benchmark_root}/cuda_graph_sharding_bench" \
  --nodes "${NODES:-1048576}" \
  --parents "${PARENTS:-8}" \
  --levels "${LEVELS:-16}" \
  --gpus "${GPUS:-4}" \
  --communities "${COMMUNITIES:-4}" \
  --locality "${LOCALITY:-0.5}" \
  --seed "${SEED:-42}" \
  --dtype "${DTYPE:-float}" \
  --strategy "${STRATEGY:-all}" \
  --mode "${MODE:-forward}" \
  --warmup "${WARMUP:-5}" \
  --runs "${RUNS:-20}" \
  --evaluations-per-run "${EVALUATIONS_PER_RUN:-1}" \
  --learning-rate "${LEARNING_RATE:-0.01}" \
  --compute-iters "${COMPUTE_ITERS:-0}" \
  --output "${SLURM_SUBMIT_DIR}/graph-sharding.${SLURM_JOB_ID}.json"
