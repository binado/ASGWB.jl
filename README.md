# ASGWB.jl

Julia workspace for modeling and inferring the **astrophysical stochastic gravitational-wave background** (ASGWB): detector networks and responses, spectral density calculation, and MCMC with Turing / AdvancedHMC.

## Workspace layout

The root repository is organized as a monorepo comprised of different small packages:

| Path | Role |
|------|------|
| [`ASGWB/`](ASGWB/) | Core library: cosmology-aware hyperparameters, redshift and spectral-density evaluation, detector PSDs/ORFs, likelihoods, catalog I/O |
| [`ASGWBInference/`](ASGWBInference/) | Inference layer on top of `ASGWB`: Turing model construction, log-posterior helpers, chain I/O |
| [`CBCDistributions/`](CBCDistributions/) | Shared building blocks: ΛCDM / *w*CDM cosmology, redshift distributions, `PopulationModel` contract, and related `Distributions.jl` helpers used by `ASGWB`. |
| [`notebooks/`](notebooks/) | **Canonical MCMC workflows** (Pluto / Jupytext): inline population model, `load_catalog`, NUTS sampling, diagnostics. |
| [`config/`](config/) | TOML for developer scripts (e.g. [`config/profile_turing.toml`](config/profile_turing.toml) for `scripts/profile_turing.jl`). |
| [`scripts/`](scripts/) | Developer utilities (profiling, chain tools, benchmarks). |


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

## MCMC inference (notebook-first)

Production sampling is driven from the notebooks, not a TOML CLI. The canonical entry point is [`notebooks/mcmc_pluto.jl`](notebooks/mcmc_pluto.jl) (Pluto); [`notebooks/mcmc.jl`](notebooks/mcmc.jl) is the Jupytext equivalent.

### Data and model assembly

1. Provide a waveform **catalog** HDF5 file (`catalog.h5`) at the repo root or set `catalog_path` in the notebook. Catalogs store per-sample intrinsic parameters and a `(n_freq, n_samples)` flux matrix `|h₊|² + |h×|²` (before fiducial `(D_L/D_gw)²` scaling). Use [`ASGWB.load_catalog`](ASGWB/src/catalog/io.jl) / [`ASGWB.save_catalog`](ASGWB/src/catalog/io.jl).
2. Define a concrete [`PopulationModel`](CBCDistributions/src/physical_model.jl) subtype in Julia (`hyperparameters`, `single_event_prior`) and build hyperparameter priors with `product_distribution(...)` (see the notebook cells).
3. Restructure catalog columns into the `NamedTuple` expected by `single_event_prior` (see `bns_samples_from_catalog` in the notebook).
4. Build [`ImportanceSamplingProblem`](ASGWB/src/inference_types.jl) with fiducial hyperparameters, then [`build_model_context`](ASGWB/src/context.jl) for detector PSDs and fiducial caches.
5. Sample with `ASGWBInference.build_turing_model`, `condition_turing_model`, and Turing NUTS; save chains via `ASGWBInference.atomic_save_chain`.

Waveform generation is not part of the Julia packages; see [scripts/generate_waveforms.py](./scripts/generate_waveforms.py) for a standalone Python accumulator (legacy layout).

### Launch Pluto MCMC

From the repository root:

```bash
just pluto
# or
julia --project=notebooks -e 'using Pkg; Pkg.instantiate(); using Pluto; Pluto.run(notebook="notebooks/mcmc_pluto.jl")'
```

Edit fiducials, hyperprior bounds, detectors, and sampler settings in the notebook cells (no `model.toml` or `MCMC_CONFIG_FILEPATH`).

### Profiling the log-density

To profile a NUTS gradient evaluation without running a full notebook:

```bash
julia --project=ASGWBInference scripts/profile_turing.jl --config-file=config/profile_turing.toml
```

## Notebooks

Notebooks live under [`notebooks/`](notebooks/) as Pluto (`.jl` with Pluto cell markers) or **Jupytext** “percent” Julia scripts. They activate the `notebooks/` project (`Pkg.activate(@__DIR__)`) and pull in `ASGWB` / `ASGWBInference` via path dependencies.

| Notebook | Purpose |
|----------|---------|
| [`notebooks/mcmc_pluto.jl`](notebooks/mcmc_pluto.jl) | **Canonical** end-to-end catalog load, Ω_GW plots, Turing NUTS, chain save/load. |
| [`notebooks/mcmc.jl`](notebooks/mcmc.jl) | Jupytext version of the MCMC workflow. |
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

Then open the `.jl` files in Jupyter Lab, VS Code, or Cursor with the Julia/IJulia extension (Jupytext notebooks).

To sync paired `.ipynb` files if you use them:

```bash
just sync-notebook
# jupytext 'notebooks/*.ipynb' --to jl:percent
```

Notebook outputs and shared plotting helpers use [`notebooks/src/NotebookSupport.jl`](notebooks/src/NotebookSupport.jl); figures default under `output-test-figures/` unless `ASGWB_FIGURES_DIR` is set.

## Further reading

- [`AGENTS.md`](AGENTS.md) — contributor conventions, testing, and architecture notes.
