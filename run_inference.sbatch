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
#   cd /path/to/ASGWB.jl
#   sbatch run_inference.sbatch [ASGWBInference/run_inference.toml]
#
# Config resolution (first hit wins):
#   1. MCMC_CONFIG_FILEPATH env var (set below from $1 if provided)
#   2. positional ARGS[1] passed to julia
#   3. ASGWBInference/run_inference.toml default

set -euo pipefail

# Cluster-specific: load Julia. Uncomment / adjust for your site.
# module purge
# module load julia/1.12

# Slurm executes a copied batch script from its spool directory on some
# clusters, so BASH_SOURCE/dirname can point at /var/spool/slurmd. Anchor all
# relative paths to the directory where sbatch was submitted.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
mkdir -p logs

if [[ ! -f Project.toml || ! -d src || ! -d ASGWBInference ]]; then
    echo "run_inference.sbatch must be submitted from the ASGWB.jl repository root" >&2
    echo "current directory: $PWD" >&2
    exit 2
fi

if [[ $# -ge 1 ]]; then
    CONFIG_PATH="$(realpath "$1")"
    case "$CONFIG_PATH" in
        "$PWD"/*) export MCMC_CONFIG_FILEPATH="${CONFIG_PATH#"$PWD"/}" ;;
        *) export MCMC_CONFIG_FILEPATH="$CONFIG_PATH" ;;
    esac
else
    export MCMC_CONFIG_FILEPATH="ASGWBInference/run_inference.toml"
fi

export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# Avoid BLAS oversubscription with MCMCThreads (the inference code also sets this,
# but exporting here covers any pre-sample BLAS calls during init).
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PATH=/home/bernardoveronese/.juliaup/bin:$PATH

# Files copied by job-nanny to the execution partition. Keep these paths
# relative to REPO_ROOT so the copied tree still has the package layout.
export INPUT="Project.toml Manifest.toml src assets ASGWBInference *.h5 *.toml"
export OUTPUT="ASGWBInference scripts output *.nc *.jls *.partial.jls"

echo "[$(date -Iseconds)] host=$(hostname) job=${SLURM_JOB_ID:-local} cpus=${SLURM_CPUS_PER_TASK:-?} threads=$JULIA_NUM_THREADS config=${MCMC_CONFIG_FILEPATH:-<default>}"

INF_PROJECT_ABS="$(realpath "$REPO_ROOT/ASGWBInference")"

job-nanny julia --project="$INF_PROJECT_ABS" -e 'using ASGWBInference; exit(ASGWBInference.command_main())' mcmc
