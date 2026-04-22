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
    time_delay_model::Union{String, Nothing}
end

abstract type IntrinsicPriorStrategy end

"""Full binary neutron star intrinsic variables in proposal samples."""
struct FullBNS <: IntrinsicPriorStrategy end

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
        observation_time_yr::Float64
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
        fiducial_spectral_density[in_band_mask]
    )
end

"""
    ProposalFiducialParameters

Fiducial cosmology, propagation, and population scalars read from the HDF5 cache
(`hyperparameters` and optionally matching keys under `redshift_prior_spec`), not the live MCMC state.
"""
Base.@kwdef struct ProposalFiducialParameters
    H0::Float64
    Omega_m::Float64
    chi0::Float64
    chin::Float64
    """Madau–Dickinson population scalars for reconstructing proposal redshift density (format v3)."""
    gamma::Union{Nothing, Float64} = nothing
    kappa::Union{Nothing, Float64} = nothing
    z_peak::Union{Nothing, Float64} = nothing
    """Power-law redshift index `lamb` when `redshift_prior_spec.family` is `PowerLaw`."""
    lamb::Union{Nothing, Float64} = nothing
end

const FULL_BNS_INTRINSIC_ORDER = [
    "mass_1_source", "mass_2_source", "redshift", "chi_1", "chi_2", "lambda_1", "lambda_2"]

"""HDF5 `proposal_samples` group attribute naming the compact-object proposal class."""
const PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR = "source_type"

"""`proposal_samples` / [`PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR`](@ref) value for BNS importance samples."""
const PROPOSAL_SAMPLES_SOURCE_TYPE_BNS = "BNS"

function resolve_intrinsic_strategy(intrinsic_site_order::Vector{String})::FullBNS
    if intrinsic_site_order == FULL_BNS_INTRINSIC_ORDER
        return FullBNS()
    end
    throw(
        ArgumentError(
        "unsupported intrinsic_site_order $(repr(intrinsic_site_order)); " *
        "only the full BNS layout is supported: $(repr(FULL_BNS_INTRINSIC_ORDER)). " *
        "Redshift-only caches are no longer supported.",
    ),
    )
end
