# Repository Guidelines

## Project Structure & Module Organization

AstroSGWB.jl is a Julia workspace for astrophysical stochastic gravitational-wave background modeling and inference. The package entry point is `AstroSGWB/src/AstroSGWB.jl`, which includes and exports reusable physics and array kernels. The generic caller-owned model contract, spectral-density forward model, Turing/AdvancedHMC wrappers, and sampling helpers live in `AstroSGWBInference/`. Canonical concrete adapters live in `AstroSGWBImportanceModels/`; its BNS Madau–Dickinson adapter is the built-in production model, while caller-owned adapters remain supported. Core modules live in `AstroSGWB/src/`, with detector-specific code under `AstroSGWB/src/detector/`. Tests live beside each package and are listed from its `test/runtests.jl`. Detector definitions and noise curves are stored in `AstroSGWB/assets/detector/`. **Production MCMC is notebook-first** (see `notebooks/mcmc.jl`): load a waveform catalog with [`load_catalog`](AstroSGWB/src/catalog/io.jl), derive the inclination-averaging convention with `average_mode(catalog)`, adapt catalog samples with `bns_samples_from_catalog`, keep `catalog.fluxes`, restructured samples, and fiducial hyperparameters explicit, prepare the model with `prepare_bns_madau_dickinson_model`, construct an [`ObservationContext`](AstroSGWB/src/detector/observation.jl) separately from `catalog.frequencies` and `catalog.in_band_mask`, build a Turing model via `AstroSGWBInference.build_turing_model(model, fluxes, samples, fiducials, observation, prior; average_mode)`, optionally subset parameters with `condition_turing_model`, sample with NUTS, and save chains via `AstroSGWBInference.atomic_save_chain`. Developer utilities (profiling, chain stacking, benchmarks) live in `scripts/` (for example `stack_partial_chains.jl` and `profile_turing.jl`). Pluto and Jupytext notebooks are under `notebooks/`.

Note: the package is still in the prototyping phase, so breaking changes are allowed and backwards compatibility is not a concern.

## Build, Test, and Development Commands

- `just fmt`: format the repository using JuliaFormatter.
- `just test`: run the package test suite through `Pkg.test()`.
- `just pluto`: instantiate the notebook environment and launch Pluto.
- `julia --project=notebooks -e 'using Pkg; Pkg.instantiate()'` then open `notebooks/mcmc.jl` in Pluto for the canonical MCMC workflow.

If `just` is unavailable, use the Julia commands in the `justfile`, for example `julia --project=AstroSGWB -e 'using Pkg; Pkg.test()'`.

## Coding Style & Naming Conventions

Target Julia version is 1.12. Formatting is controlled by `.JuliaFormatter.toml` with `style = "sciml"`; run `just fmt` before committing. Keep code type-stable where practical, especially in likelihood, redshift, and spectral-density paths.

Use existing Unicode scientific identifiers, including `Ωm`, `Ξ₀`, `Ξₙ`, `γ`, `κ`, `Λ`, `χ₁`, and `χ₂`. Live inference hyperparameters are a flat `NamedTuple`. Prepared models implement `AstroSGWBInference.hyperparameters(model)` and `AstroSGWBInference.merger_rate_and_log_weights(model, Λ, samples)`; parameter tuple order has no semantic meaning, while `keys(prior.dists)` controls Turing variable creation order. The generic seam requires no abstract model type. Concrete built-in adapters may constrain their own type parameters, as `BNSMadauDickinsonImportanceModel` does for cosmology and propagation. Test files follow `test_<area>.jl`, such as `test_redshift.jl`.

## Testing Guidelines

Tests use Julia's standard `Test` framework and are orchestrated by each package's `test/runtests.jl`. Add new test files to the relevant runner; otherwise they will not run in CI or `Pkg.test()`. Prefer focused numerical tests with explicit tolerances for cosmology, redshift integrals, likelihoods, and interpolation behavior. Minimal catalog fixtures for core tests are written on demand via `parity_catalog_dir` (see `AstroSGWB/test/parity_test_cache.jl`), which synthesizes v1 polarizations from a desired flux with `parity_polarizations`; the `:sampled_inclination` variant exercises the `CatalogInclination()` branch of `average_mode`. Cross-language flux parity against the Python stack is covered by a gitignored fixture pair regenerated with `../astrogwb/.venv/bin/python3 scripts/generate_catalog_parity_fixture.py` (the test skips when absent, matching the GWFast ORF fixture convention). Physical adapter parity, cache, AD, and concrete Turing integration tests belong in `AstroSGWBImportanceModels/test/`; generic interface and contract tests belong in `AstroSGWBInference/test/`.

Run `just test` before opening a pull request. For narrow changes, also run the closest individual test file during development with `julia --project=AstroSGWB AstroSGWB/test/test_<area>.jl`.

## Commit & Pull Request Guidelines

Recent history uses Conventional Commit-style subjects: `feat:`, `chore:`, `refactor:`, `perf:`, and `refactor!:` for breaking changes. Keep subjects imperative and scoped to one logical change.

Pull requests should include a short summary, testing performed, and links to relevant issues or discussions. Note behavior changes, fixture updates, performance-sensitive changes, and any compatibility impact. Include plots or screenshots only when changing notebooks or user-facing visual outputs.

## Architecture Notes

The inference layer is **cosmology-agnostic**. The main inference flow is: load `catalog.h5` with `load_catalog`, restructure per-event samples, keep raw catalog fluxes and fiducial hyperparameters explicit, prepare an importance model, build an [`ObservationContext`](AstroSGWB/src/detector/observation.jl) separately with `build_observation_context`, then run importance weighting, spectral-density evaluation, and posterior sampling. The generic dispatch boundary belongs to `AstroSGWBInference`; reusable astrophysical adapters belong to `AstroSGWBImportanceModels`; `AstroSGWB` and its lower-level siblings contain the kernels below them. Detector state never belongs in the importance model. Respect matrix layout conventions: cached spectral flux arrays are `(nfreq, nsamples)` in memory; full catalogs use intrinsic proposal vectors `(nsamples, n_intrinsic)`, while the canonical BNS path reduces samples to slim `NamedTuple` columns (`redshift`, `luminosity_distance`).

**Catalog format.** `catalog.h5` is a `waveform_catalog` v1 file — the same format the Python `astrogwb` package consumes, generated by `astrogwb-generate-waveform-catalog`. Format IO belongs to `PlusCross.jl` (an unregistered git dep declared in `AstroSGWB/Project.toml`'s `[sources]`); `load_catalog` only reduces the stored complex `/polarizations/{plus,cross}` to `|h₊|² + |h×|²` and re-bands, returning an [`SGWBCatalog`](AstroSGWB/src/catalog/catalog.jl). The frequency axis and in-band mask are *read from the file*, never derived from `(duration, sampling_frequency)`. PlusCross exports both `WaveformCatalog` and `load_catalog`, so it must be `import`ed and qualified, never `using`'d. HDF5's C-order `(nsamples, nfreq)` datasets read into Julia as `(nfreq, nsamples)` with no transpose.

**Inclination averaging is a catalog property, not a constant.** `spectral_density` scales by [`inclination_factor`](AstroSGWB/src/spectral_density.jl) of an `AbstractAverageMode` token: `AnalyticInclination()` (catalog is face-on, apply `2/5`) or `CatalogInclination()` (catalog samples ι, apply `1`). `average_mode(catalog)` derives it from the catalog's `inclination` column, falling back to `AnalyticInclination()` when the column is absent. Thread the resolved value through `build_turing_model`, which forwards **one** value to both the synthesized `observed` and the model that scores it — splitting them biases the fit by a constant factor with no other symptom. Config names (`"analytic_inclination"`, `"catalog_inclination"`) match astrogwb's `AverageMode` literals.
