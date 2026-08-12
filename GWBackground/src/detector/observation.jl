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

function gaussian_bin_scale(;
        effective_psd::AbstractVector{<:Real},
        frequencies::AbstractVector{<:Real},
        observation_time_sec::Real
)
    df = frequency_bin_width(frequencies)
    # effective_psd is amplitude √(variance); bin variance is (effective_psd)² / (2 T Δf)
    return sqrt.(effective_psd .^ 2 ./ (2.0 * observation_time_sec * df))
end
