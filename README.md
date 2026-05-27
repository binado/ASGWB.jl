# ASGWB.jl

Julia workspace for modeling and inferring the **astrophysical stochastic gravitational-wave background** (ASGWB): importance-sampling caches, detector networks, spectral densities, and MCMC with Turing / AdvancedHMC.

**Requirements:** [Julia](https://julialang.org/) **1.12** (see `[compat]` in each package).

## Workspace layout

This repository is a [Pkg workspace](https://pkgdocs.julialang.org/v1/environments/#Using-someone-else's-project) rooted at `Project.toml`. Local packages are wired via `[sources]` path dependencies.

| Path | Role |
|------|------|
| [`ASGWB/`](ASGWB/) | Core library: cosmology-aware hyperparameters, HDF5 importance caches, redshift and spectral-density evaluation, detector PSDs/ORFs, likelihoods, and diagnostics. |
| [`ASGWBInference/`](ASGWBInference/) | Inference layer on top of `ASGWB`: Turing models, AdvancedHMC sampling, chain I/O, and the **production** TOML-driven entry point (`julia_main` / `run_inference`). |
| [`CBCDistributions/`](CBCDistributions/) | Shared building blocks: ΛCDM / *w*CDM cosmology, redshift distributions, intrinsic priors, and related `Distributions.jl` helpers used by `ASGWB`. |
| [`notebooks/`](notebooks/) | Interactive workflows (`NotebookSupport` subproject): MCMC exploration, plotting, and Fisher-amplitude checks. Depends on `ASGWB` and `ASGWBInference` as path packages plus Makie / PairPlots / IJulia. |
| [`config/`](config/) | TOML configs for production inference (e.g. `run_inference.toml`, smoke-test variants). |
| [`scripts/`](scripts/) | Developer utilities (benchmarks, chain tools, cluster batch scripts). |

Production inference does **not** take CLI arguments; behavior is controlled entirely by TOML and environment variables (see below).

## Installation

Clone the repository and instantiate the workspace from the repo root:

```bash
cd ASGWB.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

That resolves all workspace members (`ASGWB`, `ASGWBInference`, `CBCDistributions`, `notebooks`) and their shared manifest.

To work in a single package only:

```bash
julia --project=ASGWB -e 'using Pkg; Pkg.instantiate()'
julia --project=ASGWBInference -e 'using Pkg; Pkg.instantiate()'
julia --project=notebooks -e 'using Pkg; Pkg.instantiate()'
```

Optional: install [just](https://github.com/casey/just) for common tasks (`just test`, `just fmt`, `just compile`). If `just` is unavailable, run the equivalent `julia` commands from the [`justfile`](justfile).

Run tests:

```bash
just test
# or
julia --project=ASGWB -e 'using Pkg; Pkg.test()'
julia --project=ASGWBInference -e 'using Pkg; Pkg.test()'
```

## Production inference (`run_inference`)

Inference is driven by a TOML file. Paths in the TOML that are not absolute are resolved relative to **that TOML file’s directory** (not necessarily the repo root).

### Configuration

1. Copy or edit a config under [`config/`](config/), e.g. [`config/run_inference.toml`](config/run_inference.toml).
2. Set `cache_path` to an HDF5 importance cache (see `ASGWB.load_cache` in the package docs).
3. Adjust `detectors`, `sample_only`, `[init]`, and `[sampler]` (`n_samples`, `num_chains`, `checkpoint_every`, etc.).
4. Optional `[model]` / `output_dir` / `output_prefix` for cosmology model and chain output location.

For a short smoke run (few samples, `H0` only), use [`config/run_inference_smoke_h0.toml`](config/run_inference_smoke_h0.toml).

**Config resolution** (first match wins):

1. `MCMC_CONFIG_FILEPATH` environment variable (path relative to repo root or absolute).
2. Default: `config/run_inference.toml` (relative to the repository root).

The repo root is discovered by walking up from the current directory for `Project.toml` + `ASGWB/` + `ASGWBInference/`, or set explicitly:

```bash
export ASGWB_REPO_ROOT=/path/to/ASGWB.jl
```

### Local run

From the repository root, with `JULIA_NUM_THREADS` set to the desired chain parallelism:

```bash
export MCMC_CONFIG_FILEPATH=config/run_inference.toml   # optional if using default
export JULIA_NUM_THREADS=8

julia --project=ASGWBInference -e 'using ASGWBInference; exit(ASGWBInference.julia_main())'
```

Equivalent from Julia:

```julia
using ASGWBInference
ASGWBInference.run_inference("config/run_inference.toml")
# or
ASGWBInference.run_inference_from_env()
```

`julia_main` exits with a non-zero status on failure and **rejects command-line arguments**; use `MCMC_CONFIG_FILEPATH` instead.

### Compiled executable (optional)

To build a standalone app with [PackageCompiler.jl](https://github.com/JuliaLang/PackageCompiler.jl):

```bash
just compile
# equivalent:
julia --project=ASGWBInference ASGWBInference/deps/build.jl
```

The binary is written to `ASGWBInference/build/asgwb/bin/asgwb` and is invoked the same way (TOML / env only, no ARGS).

### Cluster (Slurm)

Submit from the **repository root**:

```bash
sbatch run_inference.sbatch [config/run_inference.toml]
```

A variant without site-specific `job-nanny` wiring lives at [`scripts/run_inference.sbatch`](scripts/run_inference.sbatch). Uncomment and adjust `module load julia/...` for your cluster. The batch script sets `JULIA_NUM_THREADS` from `SLURM_CPUS_PER_TASK` and pins BLAS to one thread to avoid oversubscription with multi-chain MCMC.

## Notebooks

Notebooks live under [`notebooks/`](notebooks/) as **Jupytext** “percent” Julia scripts (`.jl`). They activate the `notebooks/` project (`Pkg.activate(@__DIR__)`) and pull in `ASGWB` / `ASGWBInference` via path dependencies.

| Notebook | Purpose |
|----------|---------|
| [`notebooks/mcmc.jl`](notebooks/mcmc.jl) | End-to-end cache load, Ω_GW plots, and Turing NUTS sampling (or load an existing chain from JLD2). |
| [`notebooks/plots.jl`](notebooks/plots.jl) | MCMC diagnostics and figures from saved chains (`FlexiChains`, `PairPlots`, `CairoMakie`). |
| [`notebooks/amplitude_posterior_gaussian_approximation.jl`](notebooks/amplitude_posterior_gaussian_approximation.jl) | Compare a 1D posterior to a Fisher / SNR Gaussian approximation (single-parameter chains). |

### Setup

```bash
julia --project=notebooks -e 'using Pkg; Pkg.instantiate()'
```

For Jupyter, register a kernel (once) from the `notebooks/` directory:

```bash
cd notebooks
julia --project=. -e 'using IJulia; IJulia.installkernel("ASGWB notebooks"; "--project=$(abspath("."))")'
```

Then open the `.jl` files in Jupyter Lab, VS Code, or Cursor with the Julia/IJulia extension (they are valid Jupytext notebooks).

To sync paired `.ipynb` files if you use them:

```bash
just sync-notebook
# jupytext 'notebooks/*.ipynb' --to jl:percent
```

Notebook outputs and shared plotting helpers use [`notebooks/src/NotebookSupport.jl`](notebooks/src/NotebookSupport.jl); figures default under `output-test-figures/` unless `ASGWB_FIGURES_DIR` is set.

## Further reading

- [`AGENTS.md`](AGENTS.md) — contributor conventions, testing, and architecture notes.
- [`references/`](references/) — related papers (PDFs).
