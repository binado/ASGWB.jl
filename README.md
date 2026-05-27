# ASGWB.jl

Julia workspace for modeling and inferring the **astrophysical stochastic gravitational-wave background** (ASGWB): detector networks and responses, spectral density calculation, and MCMC with Turing / AdvancedHMC.

## Workspace layout

The root repository is organized as a monorepo comprised of different small packages:

| Path | Role |
|------|------|
| [`ASGWB/`](ASGWB/) | Core library: cosmology-aware hyperparameters, redshift and spectral-density evaluation, detector PSDs/ORFs, likelihoods |
| [`ASGWBInference/`](ASGWBInference/) | Inference layer on top of `ASGWB`: MCMC models with Turing, manipulating chains, and the main script. |
| [`CBCDistributions/`](CBCDistributions/) | Shared building blocks: ΛCDM / *w*CDM cosmology, redshift distributions, intrinsic priors, and related `Distributions.jl` helpers used by `ASGWB`. |
| [`notebooks/`](notebooks/) | Interactive workflows (`NotebookSupport` subproject): MCMC exploration, plotting, and Fisher-amplitude checks. Depends on `ASGWB` and `ASGWBInference` as path packages plus Makie / PairPlots / IJulia. |
| [`config/`](config/) | TOML configs for production inference (e.g. `run_inference.toml`, smoke-test variants). |
| [`scripts/`](scripts/) | Developer utilities (benchmarks, chain tools, cluster batch scripts). |


## Installation

Clone the repository and instantiate the workspace from the repo root:

```bash
cd ASGWB.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

That resolves all workspace members (`ASGWB`, `ASGWBInference`, `CBCDistributions`, `notebooks`) and their shared manifest.

Run tests:

```bash
just test
# or
julia --project=ASGWB -e 'using Pkg; Pkg.test()'
julia --project=ASGWBInference -e 'using Pkg; Pkg.test()'
```

## The main inference script (`run_inference`)

Inference is driven by a TOML configuration file. Paths in the TOML that are not absolute are resolved relative to **that TOML file’s directory** (not necessarily the repo root).

*Note: the package currently does not support waveform generation in Julia, so we expect a **waveform bundle** (`bundle.h5`) plus a matching **fiducial snapshot** (`cosmology.toml`). The bundle stores per-sample intrinsic parameters and per-frequency fluxes; the TOML records cosmology, population, and observation metadata. A SHA-256 fingerprint in the bundle must match the TOML file on disk. See [scripts/generate_waveforms.py](./scripts/generate_waveforms.py) for a standalone Python waveform accumulator (legacy layout; production bundles should follow `ASGWB.save_bundle` / `ASGWB.load_problem`).*

### Configuration

1. Copy or edit a config under [`config/`](config/), e.g. [`config/run_inference.toml`](config/run_inference.toml).
2. Set `bundle_path` and `cosmology_path` (see [`ASGWB.load_problem`](ASGWB/src/io.jl) in the package docs).
3. Adjust `detectors`, `sample_only`, `[init]`, and `[sampler]` (`n_samples`, `num_chains`, `checkpoint_every`, etc.).
4. Optional `[model].cosmology` / `output_dir` / `output_prefix` for inference cosmology type and chain output location.

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
