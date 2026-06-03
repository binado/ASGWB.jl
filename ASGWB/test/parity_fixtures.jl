# Shared hyperparameters and priors for inference smoke tests.
# Catalog fixtures are materialized on demand via `parity_catalog_dir` (see `parity_test_cache.jl`).
# Included from test files that need `PARITY_THETA` (not from `runtests.jl`).

using ASGWB: canonical_hyperparameters, full_hyperparameters, full_hyperprior,
             ModifiedPropagation, LambdaCDM

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
        κ = 6.0,
        zpeak = 2.2
    )
)

const PARITY_PRIORS = full_hyperprior(_PARITY_C, _PARITY_POP)
