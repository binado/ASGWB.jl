function frequency_bin_width(frequencies::AbstractVector{<:Real})
    f = Float64.(collect(frequencies))
    length(f) >= 2 || throw(ArgumentError("at least two frequency bins are required"))
    df = f[2] - f[1]
    for k in 2:length(f)
        d = f[k] - f[k - 1]
        tol = 1e-6 * max(abs(df), 1.0)
        abs(d - df) > tol && throw(
            ArgumentError("frequencies must be uniformly spaced for gaussian_bin_scale"),
        )
    end
    return df
end

function gaussian_bin_variance(;
        covariance::AbstractVector{<:Real},
        frequencies::AbstractVector{<:Real},
        in_band_mask::BitVector,
        observation_time_sec::Real
)
    df = frequency_bin_width(frequencies)
    cov_ib = covariance[in_band_mask]
    return cov_ib ./ (2.0 * Float64(observation_time_sec) * df)
end

function gaussian_bin_scale(;
        covariance::AbstractVector{<:Real},
        frequencies::AbstractVector{<:Real},
        in_band_mask::BitVector,
        observation_time_sec::Real
)
    return sqrt.(
        gaussian_bin_variance(;
        covariance = covariance,
        frequencies = frequencies,
        in_band_mask = in_band_mask,
        observation_time_sec = observation_time_sec
    ),
    )
end

function _sgwb_scale_vector(
        covariance::AbstractVector{Float64},
        frequencies::AbstractVector{Float64},
        observation_time_sec::Float64
)
    df = frequency_bin_width(frequencies)
    return sqrt.(covariance ./ (2.0 * observation_time_sec * df))
end

"""
    build_observation_config(frequencies, detectors, in_band_mask, fiducial_spectral_density, observation_time_sec, observation_time_yr)

Reconstruct [`ObservationConfig`](@ref) from a detector network and tabulated PSDs
(isotropic ORF network covariance and per-bin Gaussian scales).
"""
function build_observation_config(
        frequencies::Vector{Float64},
        detectors::AbstractVector{Detector},
        in_band_mask::BitVector,
        fiducial_spectral_density::Vector{Float64},
        observation_time_sec::Float64,
        observation_time_yr::Float64
)
    cov = covariance_on_grid(frequencies, detectors)
    sgwb = _sgwb_scale_vector(cov, frequencies, observation_time_sec)
    return ObservationConfig(
        frequencies,
        cov,
        sgwb,
        in_band_mask,
        fiducial_spectral_density,
        observation_time_sec,
        observation_time_yr
    )
end
