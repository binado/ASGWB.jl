# Shared hyperparameters and priors for inference smoke tests.
# Importance caches are materialized on demand via `parity_cache_path` (see `ASGWB/src/parity_test_cache.jl`).

using Distributions: product_distribution, Uniform

const PARITY_THETA = HyperParameters(;
    H0 = 70.0,
    Ωm = 0.3,
    Ξ₀ = 1.1,
    Ξₙ = 0.2,
    γ = 2.9,
    κ = 6.0,
    zpeak = 2.2
)

const PARITY_PRIORS = product_distribution((
    H0 = Uniform(20.0, 140.0),
    Ωm = Uniform(0.0, 1.0),
    Ξ₀ = Uniform(0.0, 2.0),
    Ξₙ = Uniform(-1.0, 1.0),
    γ = Uniform(0.0, 5.0),
    κ = Uniform(0.0, 10.0),
    zpeak = Uniform(0.0, 5.0)
))

const PARITY_PRIOR_BOUNDS = Dict(
    "H0" => (20.0, 140.0),
    "Omega_m" => (0.0, 1.0),
    "chi0" => (0.0, 2.0),
    "chin" => (-1.0, 1.0),
    "gamma" => (0.0, 5.0),
    "kappa" => (0.0, 10.0),
    "z_peak" => (0.0, 5.0),
)
