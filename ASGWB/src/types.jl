"""
    ObservationContext

Detector-side SGWB observation layout: frequency grid, per-bin effective strain PSD
amplitude from the detector network (ORFs and tabulated PSDs; square matches network
variance), Gaussian bin scales for the likelihood, analysis band mask, and observation
time metadata. The in-band Gaussian scale is precomputed for the likelihood hot path.

The observed/fiducial spectral density is intentionally *not* part of this object; it is
the default observed data of a built [`ModelContext`](@ref) and lives there.
"""
struct ObservationContext
    frequencies::Vector{Float64}
    effective_psd::Vector{Float64}
    sgwb_scale::Vector{Float64}
    in_band_mask::BitVector
    observation_time_sec::Float64
    observation_time_yr::Float64
    sgwb_scale_in_band::Vector{Float64}
end

function ObservationContext(
        frequencies::Vector{Float64},
        effective_psd::Vector{Float64},
        sgwb_scale::Vector{Float64},
        in_band_mask::BitVector,
        observation_time_sec::Float64,
        observation_time_yr::Float64
)
    return ObservationContext(
        frequencies,
        effective_psd,
        sgwb_scale,
        in_band_mask,
        observation_time_sec,
        observation_time_yr,
        sgwb_scale[in_band_mask]
    )
end
