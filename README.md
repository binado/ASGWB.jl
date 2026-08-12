# AstroSGWB.jl

Julia workspace for modeling and inferring the **astrophysical stochastic gravitational-wave background** (AstroSGWB): detector networks and responses, spectral density calculation, and MCMC with Turing / AdvancedHMC.

## Workspace layout

The root repository is organized as a monorepo comprised of different small packages:

| Path | Role |
|------|------|
| [`Trapezoid/`](Trapezoid/) | Shared trapezoidal integration (`trapz` / `cumtrapz`) |
| [`AstroSGWB/`](AstroSGWB/) | Core library: redshift and spectral-density evaluation, detector PSDs/ORFs, catalog I/O (re-exports cosmology helpers) |
| [`AstroSGWBInference/`](AstroSGWBInference/) | Inference layer on top of `AstroSGWB`: Turing model construction, log-posterior helpers, chain I/O |
| [`AstroSGWBImportanceModels/`](AstroSGWBImportanceModels/) | Canonical concrete importance adapters, including the BNS Madau–Dickinson model used by production workflows |
| [`AstroSGWBDistributions/`](AstroSGWBDistributions/) | Shared population-distribution building blocks and the optional `PopulationModel` contract |
| [`AstroSGWBCosmology/`](AstroSGWBCosmology/) | Cosmology and GW-propagation models, distances, and reusable interpolation caches |
| [`notebooks/`](notebooks/) | **Canonical MCMC workflows** (Pluto / Jupytext): model configuration, `load_catalog`, NUTS sampling, diagnostics. |
| [`config/`](config/) | TOML for developer scripts and headless MCMC runs (e.g. [`config/mcmc/example.toml`](config/mcmc/example.toml)). |
| [`scripts/`](scripts/) | Developer utilities (profiling, chain tools, benchmarks) and [`scripts/run_mcmc.jl`](scripts/run_mcmc.jl) for config-driven cluster runs. |


## Installation

Clone the repository and instantiate the workspace from the repo root:

```bash
cd AstroSGWB.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

That resolves all workspace members, including `AstroSGWBImportanceModels`, and their
shared manifest.

Run tests:

```bash
just test
# or
julia --project=AstroSGWB -e 'using Pkg; Pkg.test()'
julia --project=AstroSGWBInference -e 'using Pkg; Pkg.test()'
julia --project=AstroSGWBImportanceModels -e 'using Pkg; Pkg.test()'
```

## MCMC inference

### Data and model assembly

1. Provide a waveform **catalog** HDF5 file (`catalog.h5`) at the repo root or set `catalog_path` in the notebook. Catalogs store per-sample intrinsic parameters and a `(nfreq, nsamples)` polarization-power matrix `|h₊|² + |h×|²` (before fiducial `(D_L/D_gw)²` scaling). Use [`AstroSGWB.load_catalog`](AstroSGWB/src/catalog/io.jl) / [`AstroSGWB.save_catalog`](AstroSGWB/src/catalog/io.jl).
2. Select an importance adapter. The built-in BNS Madau–Dickinson path is
   `AstroSGWBImportanceModels.BNSMadauDickinsonImportanceModel`; custom caller-owned
   adapters remain supported through the same callable inference contract.
3. The catalog's `samples` NamedTuple already carries both `redshift` and
   `luminosity_distance`; pass it through directly.
4. Keep the catalog polarization power, restructured samples, and fiducial hyperparameters as explicit values; these are passed directly to forward-model and inference helpers.
5. Prepare the built-in model with `prepare_bns_madau_dickinson_model(...)`, or assemble
   a caller-owned callable implementing `merger_rate_and_log_weights_fn(Λ, samples) -> (rate, log_weights)`.
   The prior declares every hyperparameter name. Compute the detector network's
   effective PSD separately with `effective_psd(frequencies, detectors)`.
6. Synthesize `observed` at the fiducials with `AstroSGWBInference.forward_model(model, polarization_power, samples, fiducials).spectral_density` when there is no external spectrum to fit, so the modified-propagation factors `Ξ(z)` are applied consistently; construct the Turing model directly with `AstroSGWBInference.astrosgwb_importance_turing_model(model, polarization_power, samples, prior, observed, frequencies, effective_psd, observation_time, average_mode, track)` — the prior declares all hyperparameter names, and fixing one is conditioning, e.g. `model | (; R₀ = fiducials.R₀)` — sample with Turing NUTS, and save chains to netCDF via `InferenceObjects.convert_to_inference_data(chain)` + `InferenceObjects.to_netcdf`.

Waveform generation is not part of the Julia packages; see [scripts/generate_waveforms.py](./scripts/generate_waveforms.py) for a standalone Python accumulator (legacy layout).

### Launch Pluto MCMC

From the repository root:

```bash
just pluto
# or
julia --project=notebooks -e 'using Pkg; Pkg.instantiate(); using Pluto; Pluto.run(notebook="notebooks/mcmc.jl")'
```

Edit fiducials, hyperprior bounds, detectors, and sampler settings in the notebook cells.

### Headless MCMC (config-driven)

[`scripts/run_mcmc.jl`](scripts/run_mcmc.jl) mirrors the sampling cells of
[`notebooks/mcmc.jl`](notebooks/mcmc.jl) but reads run-specific settings from a TOML file.
The cosmology family (`W0CDM`), GW propagation family (`ModifiedPropagation`), built-in
BNS Madau–Dickinson adapter, and hyperprior bounds are fixed in the script.

**One-time setup** (separate Julia project at [`scripts/run/`](scripts/run/)):

```bash
just setup-run
# or
julia --project=scripts/run -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Copy [`config/mcmc/example.toml`](config/mcmc/example.toml) per experiment and edit catalog path, detectors, fiducials, `sample_only`, sampler settings, and output paths. Catalog paths are resolved relative to the repository root unless absolute. Use ASCII keys in `[fiducials]` (e.g. `Omega_m`, `Xi_0`, `gamma`).

**Run locally** (one Turing chain per Julia thread via `MCMCThreads()`):

```bash
just run-mcmc config/mcmc/my_run.toml
# or
julia --project=scripts/run -t auto scripts/run_mcmc.jl config/mcmc/my_run.toml
```

`sampler.num_chains` defaults to `0`, which uses `Base.Threads.nthreads()`. If set explicitly, it must equal the thread count passed to `-t` (or `SLURM_CPUS_PER_TASK` on a cluster). The runner supports `ad_backend = "ForwardDiff"` (default) and `"Enzyme"` (reverse-mode with runtime activity). Chains are written as netCDF under `output_dir` (default `chains/`); generated filenames include the config basename so array outputs can be traced back to their input TOML.

**Submit on SLURM** from the repository root (pre-instantiate on the login node with `just setup-run`; the batch scripts do not run `Pkg.instantiate()` on compute nodes):

```bash
just submit-mcmc config/mcmc/my_run.toml
# or
mkdir -p logs
sbatch scripts/submit_mcmc_single.sbatch config/mcmc/my_run.toml
```

Set `#SBATCH --cpus-per-task` in [`scripts/submit_mcmc_single.sbatch`](scripts/submit_mcmc_single.sbatch) to the number of chains you want; adjust the Julia module load line for your cluster.

For sweeps, put one TOML config per run in a directory and submit a SLURM job array:

```bash
just submit-mcmc-array config/mcmc/sweep 8
# or
scripts/submit_mcmc_array.sh config/mcmc/sweep 8
```

The array launcher submits all `*.toml` files directly under the config directory, sorted by path. The optional second argument is the maximum number of array tasks to run at once.

### Profiling the log-density

To profile a NUTS gradient evaluation without running a full notebook:

```bash
julia --project=AstroSGWBInference scripts/profile_turing.jl --config-file=config/profile_turing.toml
```

## Notebooks

Notebooks live under [`notebooks/`](notebooks/) as Pluto (`.jl` with Pluto cell markers) or **Jupytext** “percent” Julia scripts. They activate the `notebooks/` project (`Pkg.activate(@__DIR__)`) and pull in `AstroSGWB` / `AstroSGWBInference` via path dependencies.

| Notebook | Purpose |
|----------|---------|
| [`notebooks/mcmc.jl`](notebooks/mcmc.jl) | **Canonical** end-to-end catalog load, Ω_GW plots, Turing NUTS, chain save/load. |
| [`notebooks/plots.jl`](notebooks/plots.jl) | MCMC diagnostics and figures from saved chains (`FlexiChains`, `PairPlots`, `CairoMakie`). |
| [`notebooks/amplitude_posterior_gaussian_approximation.jl`](notebooks/amplitude_posterior_gaussian_approximation.jl) | Compare a 1D posterior to a Fisher / SNR Gaussian approximation (single-parameter chains). |

### Setup

```bash
julia --project=notebooks -e 'using Pkg; Pkg.instantiate()'
```

For Jupyter, register a kernel (once) from the `notebooks/` directory:

```bash
cd notebooks
julia --project=. -e 'using IJulia; IJulia.installkernel("AstroSGWB notebooks"; "--project=$(abspath("."))")'
```

Then open the `.jl` files in Jupyter Lab, VS Code, or Cursor with the Julia/IJulia extension (Jupytext notebooks).

To sync paired `.ipynb` files if you use them:

```bash
just sync-notebook
# jupytext 'notebooks/*.ipynb' --to jl:percent
```

Notebook outputs and shared plotting helpers use [`notebooks/src/NotebookSupport.jl`](notebooks/src/NotebookSupport.jl); figures default under `output-test-figures/` unless `AstroSGWB_FIGURES_DIR` is set.

## Further reading

- [`AGENTS.md`](AGENTS.md) — contributor conventions, testing, and architecture notes.
