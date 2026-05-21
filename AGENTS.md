# Repository Guidelines

## Project Structure & Module Organization

ASGWB.jl is a Julia package for astrophysical stochastic gravitational-wave background modeling and inference. The package entry point is `ASGWB/src/ASGWB.jl`, which includes and exports the public API (physics, likelihoods, importance caches). Turing/AdvancedHMC model wrappers and sampling helpers live in the sibling package `ASGWBInference/` (see `ASGWBInference/src/InferenceImpl.jl`). Core modules live in `ASGWB/src/`, with detector-specific code under `ASGWB/src/detector/`. Tests live in `ASGWB/test/` and are listed from `ASGWB/test/runtests.jl`. Detector definitions and noise curves are stored in `ASGWB/assets/detector/`. Production inference is driven by TOML/env config through `ASGWBInference.julia_main()`, while stack/profile helpers remain callable Julia modules under `ASGWBInference/src/cli/`. Developer scripts and utilities remain in `scripts/`, while Pluto files are in `notebooks/`.

## Build, Test, and Development Commands

- `just fmt`: format the repository using JuliaFormatter.
- `just test`: run the package test suite through `Pkg.test()`.
- `just pluto`: instantiate the notebook environment and launch Pluto.
- `julia --project=ASGWBInference -e 'using ASGWBInference; exit(ASGWBInference.julia_main())'`: run the TOML-configured inference workflow (set `MCMC_CONFIG_FILEPATH`; defaults to `config/run_inference.toml`).

If `just` is unavailable, use the Julia commands in the `justfile`, for example `julia --project=ASGWB -e 'using Pkg; Pkg.test()'`.

## Coding Style & Naming Conventions

Target Julia version is 1.12. Formatting is controlled by `.JuliaFormatter.toml` with `style = "sciml"`; run `just fmt` before committing. Keep code type-stable where practical, especially in likelihood, redshift, and spectral-density paths.

Use existing Unicode scientific identifiers, including `Ωm`, `Ξ₀`, `Ξₙ`, `γ`, `κ`, `Λ`, `χ₁`, and `χ₂`. Preserve the canonical hyperparameter order `(:H0, :Ωm, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)`. Test files follow `test_<area>.jl`, such as `test_redshift.jl`.

## Testing Guidelines

Tests use Julia's standard `Test` framework and are orchestrated by `ASGWB/test/runtests.jl`. Add new test files to `runtests.jl`; otherwise they will not run in CI or `Pkg.test()`. Prefer focused numerical tests with explicit tolerances for cosmology, redshift integrals, likelihoods, and interpolation behavior. Minimal importance caches for tests are written on demand via `parity_cache_path` (see `ASGWB/test/parity_test_cache.jl`); shared smoke-test hyperparameters live in `ASGWB/test/parity_fixtures.jl`. Production inference (`run_inference.jl`) always loads real cache files from `cache_path` in TOML.

Run `just test` before opening a pull request. For narrow changes, also run the closest individual test file during development with `julia --project=ASGWB ASGWB/test/test_<area>.jl`.

## Commit & Pull Request Guidelines

Recent history uses Conventional Commit-style subjects: `feat:`, `chore:`, `refactor:`, `perf:`, and `refactor!:` for breaking changes. Keep subjects imperative and scoped to one logical change.

Pull requests should include a short summary, testing performed, and links to relevant issues or discussions. Note behavior changes, fixture updates, performance-sensitive changes, and any compatibility impact. Include plots or screenshots only when changing notebooks or user-facing visual outputs.

## Architecture Notes

The main inference flow is cache loading, redshift bundle construction, importance weighting, spectral-density evaluation, and posterior sampling. Respect matrix layout conventions: cached spectral flux arrays are `(n_freq, n_samples)` in memory, and intrinsic proposal vectors are `(n_samples, n_intrinsic)`.
