# Repository Guidelines

## Project Structure & Module Organization

ASGWB.jl is a Julia package for astrophysical stochastic gravitational-wave background modeling and inference. The package entry point is `ASGWB/src/ASGWB.jl`, which includes and exports the public API (physics, likelihoods, importance caches). Turing/AdvancedHMC model wrappers and sampling helpers live in the sibling package `ASGWBInference/` (see `ASGWBInference/src/InferenceImpl.jl`). Core modules live in `ASGWB/src/`, with detector-specific code under `ASGWB/src/detector/`. Tests live in `ASGWB/test/` and are listed from `ASGWB/test/runtests.jl`. Detector definitions and noise curves are stored in `ASGWB/assets/detector/`. **Production MCMC is notebook-first** (see `notebooks/mcmc_pluto.jl`): callers define a concrete [`PopulationModel`](ASGWB/src/models/base.jl), load a waveform catalog with [`load_catalog`](ASGWB/src/catalog/io.jl), build an [`ImportanceSamplingProblem`](ASGWB/src/inference_types.jl) and [`ModelContext`](ASGWB/src/context.jl), then sample via `ASGWBInference.build_turing_model`. Developer utilities (profiling, chain stacking, benchmarks) live in `scripts/` (for example `stack_partial_chains.jl` and `profile_turing.jl`). Pluto and Jupytext notebooks are under `notebooks/`.

Note: the package is still in the prototyping phase, so breaking changes are allowed and backwards compatibility is not a concern.

## Build, Test, and Development Commands

- `just fmt`: format the repository using JuliaFormatter.
- `just test`: run the package test suite through `Pkg.test()`.
- `just pluto`: instantiate the notebook environment and launch Pluto on `notebooks/mcmc_pluto.jl`.
- `julia --project=notebooks -e 'using Pkg; Pkg.instantiate()'` then open `notebooks/mcmc_pluto.jl` in Pluto or Jupyter for the canonical MCMC workflow.

If `just` is unavailable, use the Julia commands in the `justfile`, for example `julia --project=ASGWB -e 'using Pkg; Pkg.test()'`.

## Coding Style & Naming Conventions

Target Julia version is 1.12. Formatting is controlled by `.JuliaFormatter.toml` with `style = "sciml"`; run `just fmt` before committing. Keep code type-stable where practical, especially in likelihood, redshift, and spectral-density paths.

Use existing Unicode scientific identifiers, including `Ωm`, `Ξ₀`, `Ξₙ`, `γ`, `κ`, `Λ`, `χ₁`, and `χ₂`. Live inference hyperparameters are a flat `NamedTuple`; `canonical_hyperparameters` validates, orders, and converts boundary inputs. Unconstrained HMC/Turing vector layout follows `full_hyperparameters(C, pop)` on the caller's `PopulationModel`. Test files follow `test_<area>.jl`, such as `test_redshift.jl`.

## Testing Guidelines

Tests use Julia's standard `Test` framework and are orchestrated by `ASGWB/test/runtests.jl`. Add new test files to `runtests.jl`; otherwise they will not run in CI or `Pkg.test()`. Prefer focused numerical tests with explicit tolerances for cosmology, redshift integrals, likelihoods, and interpolation behavior. Minimal catalog fixtures for tests are written on demand via `parity_catalog_dir` (see `ASGWB/test/parity_test_cache.jl`); shared smoke-test hyperparameters live in `ASGWB/test/parity_fixtures.jl`. The reference BNS population for tests is `ParityBNSPopulation` in `ASGWB/test/fixture_population.jl`.

Run `just test` before opening a pull request. For narrow changes, also run the closest individual test file during development with `julia --project=ASGWB ASGWB/test/test_<area>.jl`.

## Commit & Pull Request Guidelines

Recent history uses Conventional Commit-style subjects: `feat:`, `chore:`, `refactor:`, `perf:`, and `refactor!:` for breaking changes. Keep subjects imperative and scoped to one logical change.

Pull requests should include a short summary, testing performed, and links to relevant issues or discussions. Note behavior changes, fixture updates, performance-sensitive changes, and any compatibility impact. Include plots or screenshots only when changing notebooks or user-facing visual outputs.

## Architecture Notes

The main inference flow is: load `catalog.h5` with `load_catalog`, restructure per-event samples to match `single_event_prior`, construct `ImportanceSamplingProblem` (raw fluxes + fiducial hyperparameters), build `ModelContext` with `build_model_context`, then run importance weighting, spectral-density evaluation, and posterior sampling. The cosmology family `C` is passed as a type argument on atomic calls, not stored on the problem. Respect matrix layout conventions: cached spectral flux arrays are `(n_freq, n_samples)` in memory, and intrinsic proposal vectors are `(n_samples, n_intrinsic)`.
