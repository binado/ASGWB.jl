#!/bin/bash
#SBATCH --job-name=asgwb-inference
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --hint=nomultithread

# Usage:
#   sbatch scripts/run_inference.sbatch [path/to/config.toml]
#
# Config resolution (first hit wins):
#   1. MCMC_CONFIG_FILEPATH env var (set below from $1 if provided)
#   2. positional ARGS[1] passed to julia
#   3. scripts/run_inference.toml default

set -euo pipefail

# Cluster-specific: load Julia. Uncomment / adjust for your site.
# module purge
# module load julia/1.12

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

if [[ $# -ge 1 ]]; then
    export MCMC_CONFIG_FILEPATH="$(realpath "$1")"
fi

export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# Avoid BLAS oversubscription with MCMCThreads (the script also sets this,
# but exporting here covers any pre-sample BLAS calls during init).
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export INPUT="*.h5 scripts/*.jl"
export OUTPUT="*.nc *.jls"

echo "[$(date -Iseconds)] host=$(hostname) job=${SLURM_JOB_ID:-local} cpus=${SLURM_CPUS_PER_TASK:-?} threads=$JULIA_NUM_THREADS config=${MCMC_CONFIG_FILEPATH:-<default>}"

jobnanny julia --project=. run_inference.jl
