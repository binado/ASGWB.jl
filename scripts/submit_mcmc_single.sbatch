#!/bin/bash
#SBATCH --job-name=gwbackground-mcmc
#SBATCH --output=logs/mcmc-%j.out
#SBATCH --error=logs/mcmc-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Headless GWBackground NUTS run on SLURM. One config file == one experiment.
#
# Submit from the repository root with the config as the single positional argument:
#   cd /path/to/GWBackground.jl
#   sbatch scripts/submit_mcmc_single.sbatch config/mcmc/example.toml
#
# One Turing chain runs per allocated CPU thread (MCMCThreads), so set
# --cpus-per-task to the number of chains you want.

set -euo pipefail

CONFIG="${1:?usage: sbatch scripts/submit_mcmc_single.sbatch <config.toml>}"

# Slurm executes a copied batch script from its spool directory on some
# clusters, so BASH_SOURCE/dirname can point at /var/spool/slurmd. Anchor all
# relative paths to the directory where sbatch was submitted.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
mkdir -p logs

if [[ ! -d GWBackground || ! -d GWBackgroundInference || ! -f scripts/run_mcmc.jl ]]; then
    echo "submit_mcmc_single.sbatch must be submitted from the GWBackground.jl repository root" >&2
    echo "current directory: $PWD" >&2
    exit 2
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "config file does not exist: $CONFIG" >&2
    exit 2
fi

# Adjust this to your cluster's module system / Julia install.
# module load julia/1.12

# Locate the raw Julia binary to bypass the juliaup config lock issue on HPC.
JULIA_EXE=$(ls -d "$HOME"/.julia/juliaup/julia-*/bin/julia 2>/dev/null | sort -V | tail -n 1 || echo "julia")

# Each Julia thread runs one MCMC chain.
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# Keep BLAS single-threaded; the runner also enforces this.
export OPENBLAS_NUM_THREADS=1
export INPUT="*"
export OUTPUT="*"

# Set up a node-local depot to avoid file-locking contention on shared filesystems (NFS/Lustre).
# Julia will read from ~/.julia but write any temporary/architecture cache to the local SSD.
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    export JULIA_DEPOT_PATH="$SLURM_TMPDIR/julia_depot:$HOME/.julia"
else
    export JULIA_DEPOT_PATH="/tmp/$USER/julia_depot:$HOME/.julia"
fi
mkdir -p "${JULIA_DEPOT_PATH%%:*}"

echo "[submit_mcmc_single] config=$CONFIG julia=$JULIA_EXE threads=$JULIA_NUM_THREADS depot=${JULIA_DEPOT_PATH%%:*}"

# NOTE: Do NOT run Pkg.instantiate() on compute nodes.
# Instead, ensure you run `just setup-run` on the login node BEFORE submitting.
#
# julia --project=scripts/run -e 'using Pkg; Pkg.instantiate()'

# Run the MCMC sampler.
job-nanny "$JULIA_EXE" --project=scripts/run -t "$JULIA_NUM_THREADS" \
    scripts/run_mcmc.jl "$CONFIG"
