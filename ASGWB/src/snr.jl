"""
    spectral_snr_squared(spectral_density, effective_psd, observation_time_sec, df) -> Real

Discrete matched-filter **SNR²** for a diagonal Gaussian noise model:

``\\mathrm{SNR}^2 = \\sum_i S_{h,i}^2 / \\sigma_i^2``,

where ``S_{h,i}`` is the strain spectral density in bin ``i`` and

``\\sigma_i = \\mathrm{effective\\_psd}_i / \\sqrt{2 T \\Delta f}``,

with observation time ``T`` in seconds, frequency bin width ``\\Delta f =`` `df` in Hz, and
network [`effective_psd`](@ref) in the same convention as [`gaussian_bin_scale`](@ref) and
[`ObservationConfig`](@ref) (per-bin `sgwb_scale` from [`build_observation_config`](@ref) matches
this `σ` path when `df` is the same width used there, e.g. from [`frequency_bin_width`](@ref) on
the analysis frequency grid).
"""
function spectral_snr_squared(
        spectral_density::AbstractVector{<:Real},
        effective_psd::AbstractVector{<:Real},
        observation_time_sec::Real,
        df::Real
)
    denom = sqrt(2 * observation_time_sec * df)
    sgwb_scale = effective_psd ./ denom
    return sum(abs2, spectral_density ./ sgwb_scale)
end

"""
    spectral_snr(spectral_density, effective_psd, observation_time_sec, df) -> Real

``\\mathrm{SNR} = \\sqrt{\\mathrm{SNR}^2}`` with ``\\mathrm{SNR}^2`` from
[`spectral_snr_squared`](@ref).
"""
function spectral_snr(
        spectral_density::AbstractVector{<:Real},
        effective_psd::AbstractVector{<:Real},
        observation_time_sec::Real,
        df::Real
)
    return sqrt(spectral_snr_squared(
        spectral_density,
        effective_psd,
        observation_time_sec,
        df
    ))
end
