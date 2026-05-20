"""
    ObservationConfig

Detector-side SGWB observation layout: frequency grid, per-bin effective strain PSD
amplitude from the detector network (ORFs and tabulated PSDs; square matches network
variance), Gaussian bin scales for the likelihood, analysis band mask, and observation
time metadata.
"""
struct ObservationConfig
    frequencies::Vector{Float64}
    effective_psd::Vector{Float64}
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
        effective_psd::Vector{Float64},
        sgwb_scale::Vector{Float64},
        in_band_mask::BitVector,
        fiducial_spectral_density::Vector{Float64},
        observation_time_sec::Float64,
        observation_time_yr::Float64
)
    return ObservationConfig(
        frequencies,
        effective_psd,
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
    Ωm::Float64
    Ξ₀::Float64
    Ξₙ::Float64
    """Madau–Dickinson population scalars for reconstructing proposal redshift density (format v3)."""
    γ::Union{Nothing, Float64} = nothing
    κ::Union{Nothing, Float64} = nothing
    zpeak::Union{Nothing, Float64} = nothing
    """Power-law redshift index when `redshift_prior_spec.family` is `PowerLaw` (HDF5 key `lamb`)."""
    Λ::Union{Nothing, Float64} = nothing
end

"""HDF5 `proposal_samples` group attribute naming the compact-object proposal class."""
const PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR = "source_type"

"""`proposal_samples` / [`PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR`](@ref) value for BNS importance samples."""
const PROPOSAL_SAMPLES_SOURCE_TYPE_BNS = "BNS"
