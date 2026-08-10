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
        effective_psd::AbstractVector{<:Real},
        frequencies::AbstractVector{<:Real},
        observation_time_sec::Real
)
    df = frequency_bin_width(frequencies)
    # effective_psd is amplitude √(variance); bin variance is (effective_psd)² / (2 T Δf)
    return effective_psd .^ 2 ./ (2.0 * Float64(observation_time_sec) * df)
end

function gaussian_bin_scale(;
        effective_psd::AbstractVector{<:Real},
        frequencies::AbstractVector{<:Real},
        observation_time_sec::Real
)
    return sqrt.(
        gaussian_bin_variance(;
        effective_psd = effective_psd,
        frequencies = frequencies,
        observation_time_sec = observation_time_sec
    ),
    )
end

function _sgwb_scale_vector(
        effective_psd::AbstractVector{Float64},
        frequencies::AbstractVector{Float64},
        observation_time_sec::Float64
)
    df = frequency_bin_width(frequencies)
    # √(variance / (2 T Δf)) with variance = effective_psd^2
    denom = sqrt(2.0 * observation_time_sec * df)
    return effective_psd ./ denom
end

"""
    build_observation_context(frequencies, detectors, observation_time)

Build an [`ObservationContext`](@ref) from a detector network and tabulated PSDs
(isotropic ORF network [`effective_psd`](@ref) and per-bin Gaussian scales).

`frequencies` must already be restricted to the analysis band (slice the catalog's
`frequencies` and the rows of its `fluxes` with the same mask beforehand); every bin
passed here is scored by the likelihood. `observation_time` is the observation
duration in years (Julian year).
"""
function build_observation_context(
        frequencies::Vector{Float64},
        detectors::AbstractVector{Detector},
        observation_time::Float64
)
    eff = effective_psd(frequencies, detectors)
    obs_sec = year_to_second(observation_time)
    sgwb = _sgwb_scale_vector(eff, frequencies, obs_sec)
    return ObservationContext(
        frequencies,
        eff,
        sgwb,
        observation_time
    )
end
