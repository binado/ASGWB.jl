"""Ordering of parameters in the flat `HyperParameters` NamedTuple used by
`product_distribution` / Bijectors / HMC."""
const DEFAULT_PARAMETER_ORDER = (:H0, :Omega_m, :chi0, :chin, :gamma, :kappa, :z_peak)

"""
    HyperParameters

Flat Madau–Dickinson inference state as a concrete `Float64` `NamedTuple` alias keyed by
[`DEFAULT_PARAMETER_ORDER`](@ref). Used directly by the product-distribution prior
(`logpdf(prior, h)`), by Bijectors (`Bijectors.link(prior, h)`), and inside the Turing
`@model`. The keyword constructor [`HyperParameters(; H0, Omega_m, …)`](@ref) and the
from-NamedTuple constructor [`HyperParameters(nt)`](@ref) coerce inputs to `Float64`
for user-facing code. Inner loops that see `ForwardDiff.Dual` values (e.g. HMC
log-density gradient) work against the [`HyperParametersNT`](@ref) UnionAll alias which
accepts any element types.
"""
const HyperParameters = @NamedTuple{
    H0::Float64,
    Omega_m::Float64,
    chi0::Float64,
    chin::Float64,
    gamma::Float64,
    kappa::Float64,
    z_peak::Float64
}

"""
    HyperParametersNT

UnionAll NamedTuple type keyed by [`DEFAULT_PARAMETER_ORDER`](@ref) that matches any
element types. Used in inner-loop function signatures (`logprior`, `logposterior`,
`loglikelihood`, `evaluate_importance_terms`, `build_redshift_grid_bundle`,
`compute_importance_weights`) so `ForwardDiff.Dual`-valued hyperparameters from HMC
gradients flow through unchanged.
"""
const HyperParametersNT = NamedTuple{DEFAULT_PARAMETER_ORDER}

"""
    HyperParameters(; H0, Omega_m, chi0=1.0, chin=0.0, gamma, kappa, z_peak) -> HyperParameters

Convenience keyword constructor returning a flat [`HyperParameters`](@ref) NamedTuple
with `Float64` coercion.
"""
function HyperParameters(;
        H0::Real,
        Omega_m::Real,
        chi0::Real = 1.0,
        chin::Real = 0.0,
        gamma::Real,
        kappa::Real,
        z_peak::Real
)::HyperParameters
    return (
        H0 = Float64(H0),
        Omega_m = Float64(Omega_m),
        chi0 = Float64(chi0),
        chin = Float64(chin),
        gamma = Float64(gamma),
        kappa = Float64(kappa),
        z_peak = Float64(z_peak)
    )
end

"""
    HyperParameters(nt::NamedTuple) -> HyperParameters

Build a [`HyperParameters`](@ref) NamedTuple (with `Float64` coercion) from any
NamedTuple carrying at least `:H0, :Omega_m, :gamma, :kappa, :z_peak`. `chi0` /
`chin` default to `1.0` / `0.0` when absent.
"""
function HyperParameters(nt::NamedTuple)::HyperParameters
    return HyperParameters(;
        H0 = nt.H0,
        Omega_m = nt.Omega_m,
        chi0 = haskey(nt, :chi0) ? nt.chi0 : 1.0,
        chin = haskey(nt, :chin) ? nt.chin : 0.0,
        gamma = nt.gamma,
        kappa = nt.kappa,
        z_peak = nt.z_peak
    )
end

abstract type ProposalSampleBundle end

"""
    FullBNSSamplesSoA

Struct-of-arrays proposal-sample container matching the NamedTuple returned by
`rand(prior, n)` when `prior = intrinsic_prior(FullBNS(), bundle)`:

- `mass::Matrix{Float64}` of size `(2, n)`; row 1 is `mass_1_source`, row 2 is `mass_2_source`.
- `redshift`, `chi_1`, `chi_2`, `lambda_1`, `lambda_2` are `Vector{Float64}` of length `n`.

Matches `keys(prior.dists)` for the full-BNS intrinsic prior.
"""
const FullBNSSamplesSoA = @NamedTuple{
    mass::Matrix{Float64},
    redshift::Vector{Float64},
    chi_1::Vector{Float64},
    chi_2::Vector{Float64},
    lambda_1::Vector{Float64},
    lambda_2::Vector{Float64}
}

"""
    stack_source_masses(mass_1_source, mass_2_source) -> Matrix{Float64}

Pack two same-length mass vectors into the `2 × n` matrix expected by
[`FullBNSSamplesSoA`](@ref)`.mass` (row 1 = `mass_1_source`, row 2 = `mass_2_source`).
"""
function stack_source_masses(
        mass_1_source::AbstractVector{<:Real},
        mass_2_source::AbstractVector{<:Real}
)::Matrix{Float64}
    n = length(mass_1_source)
    length(mass_2_source) == n ||
        throw(ArgumentError("mass_1_source and mass_2_source must have matching lengths"))
    return permutedims(
        hcat(collect(Float64, mass_1_source), collect(Float64, mass_2_source)),
    )
end

"""
    ProposalData

Proposal-sample bundle for the importance-sampling problem.

Matrix layouts:
- `intrinsic_vector` is `(n_samples, n_intrinsic)` (rows = samples, columns = intrinsic sites).
- `cached_flux_over_dgw2` is `(n_freq, n_samples)` (column-major friendly: each proposal
  sample is a contiguous column; `fluxes * weights` contracts to a per-frequency vector).
"""
struct ProposalData
    intrinsic_site_order::Vector{String}
    samples::FullBNSSamplesSoA
    log_prob::Vector{Float64}
    intrinsic_vector::Matrix{Float64}
    cached_flux_over_dgw2::Matrix{Float64}
    dgw_fid_sq::Vector{Float64}
end

"""
    ImportanceSamplingProblem

In-memory importance-sampling context. See [`importance_sampling_problem`](@ref) and
[`load_cache`](@ref). `fiducial_parameters` merges population scalars from HDF5
`hyperparameters` and `redshift_prior_spec` ([`ProposalFiducialParameters`](@ref)), not the live
[`HyperParameters`](@ref) state. `redshift_integral_fiducial` is carried for cache round-trip and
may differ from [`fiducial_redshift_integral`](@ref) when the file’s optional
`redshift_integral_fiducial` attribute overrides the recomputed value; likelihood evaluation uses
the integral implied by the live [`HyperParameters`](@ref), not this field.
"""
struct ImportanceSamplingProblem
    proposal::ProposalData
    observation::ObservationConfig
    redshift_prior_spec::RedshiftPriorSpec
    local_merger_rate::Float64
    redshift_integral_fiducial::Float64
    fiducial_parameters::ProposalFiducialParameters
    strategy::FullBNS
end

redshift(s::NamedTuple) = s.redshift

redshift(problem::ImportanceSamplingProblem) = redshift(problem.proposal.samples)

function _validate_strategy_bundle(strategy::FullBNS, proposal::ProposalData)
    proposal.samples isa FullBNSSamplesSoA ||
        throw(ArgumentError("proposal samples must match the FullBNSSamplesSoA layout"))
    return nothing
end

"""
    importance_sampling_problem(
        proposal, observation, redshift_prior_spec,
        local_merger_rate, fiducial_parameters,
    ) -> ImportanceSamplingProblem

Five-argument form: the stored fiducial integral is
[`fiducial_redshift_integral`](@ref) applied to `fiducial_parameters` and `redshift_prior_spec`.

    importance_sampling_problem(
        proposal, observation, redshift_prior_spec,
        local_merger_rate, redshift_integral_fiducial, fiducial_parameters,
    ) -> ImportanceSamplingProblem

Six-argument form: supply a precomputed fiducial redshift integral (e.g. read from a cache file).

Both forms validate [`IntrinsicPriorStrategy`](@ref) against the proposal sample bundle type.
"""
function importance_sampling_problem(
        proposal::ProposalData,
        observation::ObservationConfig,
        redshift_prior_spec::RedshiftPriorSpec,
        local_merger_rate::Real,
        redshift_integral_fiducial::Real,
        fiducial_parameters::ProposalFiducialParameters
)
    strategy = resolve_intrinsic_strategy(proposal.intrinsic_site_order)
    _validate_strategy_bundle(strategy, proposal)
    return ImportanceSamplingProblem(
        proposal,
        observation,
        redshift_prior_spec,
        Float64(local_merger_rate),
        Float64(redshift_integral_fiducial),
        fiducial_parameters,
        strategy
    )
end

"""
    ImportanceCache

Deprecated alias for [`ImportanceSamplingProblem`](@ref); use `ImportanceSamplingProblem` in new code.
"""
const ImportanceCache = ImportanceSamplingProblem
