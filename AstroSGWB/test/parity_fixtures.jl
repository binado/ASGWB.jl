# Shared hyperparameters and priors for inference smoke tests.
# Catalog fixtures are materialized on demand via `parity_catalog_dir` (see `parity_test_cache.jl`).
# Included from test files that need `PARITY_THETA` (not from `runtests.jl`).

using Distributions: Uniform, product_distribution

# Cosmology and propagation bounds for inference smoke tests.
const PARITY_PRIORS = product_distribution((
    H0 = Uniform(20.0, 140.0),
    Ωm = Uniform(0.05, 0.95),
    Ξ₀ = Uniform(0.5, 5.0),
    Ξₙ = Uniform(0.05, 3.0),
    γ = Uniform(0.0, 5.0),
    κ = Uniform(0.0, 8.0),
    zpeak = Uniform(0.5, 5.0)
))

const PARITY_THETA = (;
    H0 = 70.0,
    Ωm = 0.3,
    Ξ₀ = 1.1,
    Ξₙ = 0.2,
    γ = 2.9,
    κ = 3.1,
    zpeak = 2.2
)
