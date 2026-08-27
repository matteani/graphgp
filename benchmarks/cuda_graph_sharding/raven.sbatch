#!/bin/bash -l
#SBATCH --job-name=graph-sharding
#SBATCH --output=graph-sharding.%j.out
#SBATCH --error=graph-sharding.%j.err
#SBATCH --nodes=1
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
  --nodes 1048576 \
  --parents 8 \
  --levels 16 \
  --gpus 4 \
  --communities 4 \
  --locality 0.5 \
  --strategy all \
  --warmup 5 \
  --runs 20 \
  --output "${SLURM_SUBMIT_DIR}/graph-sharding.${SLURM_JOB_ID}.json"
