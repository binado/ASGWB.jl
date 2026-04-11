"""
    RedshiftPriorFamily

Closed set of redshift population models supported by [`RedshiftPriorSpec`](@ref).
File-backed caches store snake-case strings; use [`parse_redshift_prior_family`](@ref) when reading.
"""
@enum RedshiftPriorFamily MadauDickinson PowerLaw

"""
    parse_redshift_prior_family(s::AbstractString) -> RedshiftPriorFamily

Parse the HDF5 / Python cache string for `redshift_prior_spec.family`.
"""
function parse_redshift_prior_family(s::AbstractString)
    s == "madau_dickinson" && return MadauDickinson
    s == "power_law" && return PowerLaw
    throw(ArgumentError("unsupported redshift prior family $(repr(s))"))
end

"""
    RedshiftPriorSpec

Redshift grid settings for [`build_redshift_grid_bundle`](@ref). `time_delay_model`
is reserved for future parity with the Python stack; unsupported values must be
empty or `nothing` at load time.
"""
struct RedshiftPriorSpec
    family::RedshiftPriorFamily
    z_min::Float64
    z_max::Float64
    num_interp::Int
    time_delay_model::Union{String,Nothing}
end

abstract type IntrinsicPriorStrategy end

"""Intrinsic site order is redshift-only (no BNS mass/spin/tidal parameters)."""
struct RedshiftOnly <: IntrinsicPriorStrategy end

"""Full binary neutron star intrinsic variables in proposal samples."""
struct FullBNS <: IntrinsicPriorStrategy end

struct ProposalData
    intrinsic_site_order::Vector{String}
    samples::Dict{String,Vector{Float64}}
    log_prob::Vector{Float64}
    intrinsic_vector::Matrix{Float64}
    cached_flux_over_dgw2::Matrix{Float64}
    dgw_fid_sq::Vector{Float64}
end

"""
    ObservationConfig

Detector-side SGWB observation layout: frequency grid, uncertainties, band mask,
and observation time metadata used by the likelihood.
"""
struct ObservationConfig
    frequencies::Vector{Float64}
    covariance::Vector{Float64}
    sgwb_scale::Vector{Float64}
    in_band_mask::BitVector
    fiducial_spectral_density::Vector{Float64}
    observation_time_sec::Float64
    observation_time_yr::Float64
    sgwb_scale_in_band::Vector{Float64}
    fiducial_spectral_density_in_band::Vector{Float64}
end

function ObservationConfig(
    frequencies::Vector{Float64},
    covariance::Vector{Float64},
    sgwb_scale::Vector{Float64},
    in_band_mask::BitVector,
    fiducial_spectral_density::Vector{Float64},
    observation_time_sec::Float64,
    observation_time_yr::Float64,
)
    return ObservationConfig(
        frequencies,
        covariance,
        sgwb_scale,
        in_band_mask,
        fiducial_spectral_density,
        observation_time_sec,
        observation_time_yr,
        sgwb_scale[in_band_mask],
        fiducial_spectral_density[in_band_mask],
    )
end

"""
    HyperParameters

Cosmology / propagation hyperparameters stored with an importance cache (proposal
reference values). Keys in HDF5 must match the field names exactly with no extras.
"""
Base.@kwdef struct HyperParameters
    H0::Float64
    Omega_m::Float64
    chi0::Float64
    chin::Float64
end

"""
    ImportanceSamplingProblem

In-memory importance-sampling context: proposal draws, observation configuration,
redshift prior spec, scalar metadata, and [`HyperParameters`](@ref).

Construct via [`importance_sampling_problem`](@ref) or load from disk with
[`load_cache`](@ref).
"""
struct ImportanceSamplingProblem{S<:IntrinsicPriorStrategy}
    proposal::ProposalData
    observation::ObservationConfig
    redshift_prior_spec::RedshiftPriorSpec
    local_merger_rate::Float64
    redshift_integral_fiducial::Float64
    hyperparameters::HyperParameters
    strategy::S
end

redshift(problem::ImportanceSamplingProblem) = problem.proposal.samples["redshift"]

const FULL_BNS_INTRINSIC_ORDER = [
    "mass_1_source", "mass_2_source", "redshift",
    "chi_1", "chi_2", "lambda_1", "lambda_2",
]

function resolve_intrinsic_strategy(intrinsic_site_order::Vector{String})::IntrinsicPriorStrategy
    if intrinsic_site_order == ["redshift"]
        return RedshiftOnly()
    elseif intrinsic_site_order == FULL_BNS_INTRINSIC_ORDER
        return FullBNS()
    else
        throw(
            ArgumentError(
                "unsupported intrinsic_site_order $(intrinsic_site_order); supported layouts are redshift-only and the full BNS intrinsic prior",
            ),
        )
    end
end

"""
    importance_sampling_problem(
        proposal::ProposalData,
        observation::ObservationConfig,
        redshift_prior_spec::RedshiftPriorSpec,
        local_merger_rate::Real,
        redshift_integral_fiducial::Real,
        hyperparameters::HyperParameters,
    ) -> ImportanceSamplingProblem

Canonical in-memory constructor. Chooses [`IntrinsicPriorStrategy`](@ref) from
`proposal.intrinsic_site_order` and validates consistency.
"""
function importance_sampling_problem(
    proposal::ProposalData,
    observation::ObservationConfig,
    redshift_prior_spec::RedshiftPriorSpec,
    local_merger_rate::Real,
    redshift_integral_fiducial::Real,
    hyperparameters::HyperParameters,
)
    strategy = resolve_intrinsic_strategy(proposal.intrinsic_site_order)
    return ImportanceSamplingProblem(
        proposal,
        observation,
        redshift_prior_spec,
        Float64(local_merger_rate),
        Float64(redshift_integral_fiducial),
        hyperparameters,
        strategy,
    )::ImportanceSamplingProblem{typeof(strategy)}
end

"""
    ImportanceCache

Deprecated alias for [`ImportanceSamplingProblem`](@ref); use `ImportanceSamplingProblem` in new code.
"""
const ImportanceCache = ImportanceSamplingProblem
