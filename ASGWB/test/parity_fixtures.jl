# Shared hyperparameters and priors for inference smoke tests.
# Catalog fixtures are materialized on demand via `parity_catalog_dir` (see `parity_test_cache.jl`).
# Included from test files that need `PARITY_THETA` (not from `runtests.jl`).

using ASGWB: canonical_hyperparameters, full_hyperparameters, ModifiedPropagation, LambdaCDM
using Distributions: Uniform, product_distribution

if !@isdefined ParityBNSPopulation
    include(joinpath(@__DIR__, "fixture_population.jl"))
end

const _PARITY_C = ModifiedPropagation{LambdaCDM}
const _PARITY_POP = ParityBNSPopulation()
const _PARITY_ORDER = full_hyperparameters(_PARITY_C, _PARITY_POP)

const PARITY_THETA = canonical_hyperparameters(
    _PARITY_ORDER,
    (;
        H0 = 70.0,
        Ωm = 0.3,
        Ξ₀ = 1.1,
        Ξₙ = 0.2,
        γ = 2.9,
        κ = 3.1,
        zpeak = 2.2
    )
)

# Cosmology bounds duplicated from CBCDistributions/test/fixtures.jl (canonical test values).
const PARITY_PRIORS = product_distribution(merge(
    (
        H0 = Uniform(20.0, 140.0),
        Ωm = Uniform(0.05, 0.95),
        Ξ₀ = Uniform(0.5, 5.0),
        Ξₙ = Uniform(0.05, 3.0),
    ),
    parity_population_hyperprior().dists,
))
