"""
    inner_product(a, b, effective_psd, observation_time_sec, df) -> Real

Discrete frequency-domain inner product ``\\langle a, b \\rangle`` for the diagonal Gaussian
noise model shared by [`spectral_snr_squared`](@ref) and [`gaussian_bin_scale`](@ref):

``\\langle a, b \\rangle = 2 T \\Delta f \\sum_i a_i b_i / \\sigma_i^2``,

where ``\\sigma_i = \\mathrm{effective\\_psd}_i / \\sqrt{2 T \\Delta f}`` (equivalently
``\\langle a, b \\rangle = \\sum_i a_i b_i / (\\mathrm{effective\\_psd}_i^2 / (2 T \\Delta f))``).

Vectors `a`, `b`, and `effective_psd` must have the same length. Observation time ``T`` is in
seconds, bin width ``\\Delta f =`` `df` is in Hz, and network [`effective_psd`](@ref) follows the
same convention as [`ObservationConfig`](@ref) (use the same `df` as
[`frequency_bin_width`](@ref) on the analysis grid when binning is uniform).

With ``a = b`` equal to a strain spectral density ``S_h``, ``\\langle S_h, S_h \\rangle`` is
matched-filter **SNR²**; see [`spectral_snr_squared`](@ref).
"""
function inner_product(
        a::AbstractVector{<:Real},
        b::AbstractVector{<:Real},
        effective_psd::AbstractVector{<:Real},
        observation_time_sec::Real,
        df::Real
)
    prefactor = 2 * observation_time_sec * df
    return prefactor * sum(a .* b ./ effective_psd .^ 2)
end

"""
    spectral_snr_squared(spectral_density, effective_psd, observation_time_sec, df) -> Real

Discrete matched-filter **SNR²** for a diagonal Gaussian noise model:

``\\mathrm{SNR}^2 = \\langle S_h, S_h \\rangle = \\sum_i S_{h,i}^2 / \\sigma_i^2``,

where ``S_{h,i}`` is the strain spectral density in bin ``i`` and

``\\sigma_i = \\mathrm{effective\\_psd}_i / \\sqrt{2 T \\Delta f}``,

with observation time ``T`` in seconds, frequency bin width ``\\Delta f =`` `df` in Hz, and
network [`effective_psd`](@ref) in the same convention as [`gaussian_bin_scale`](@ref) and
[`ObservationConfig`](@ref) (per-bin `sgwb_scale` from [`build_observation_config`](@ref) matches
this `σ` path when `df` is the same width used there, e.g. from [`frequency_bin_width`](@ref) on
the analysis frequency grid).

Implemented as [`inner_product`](@ref) with `a = b = spectral_density`.
"""
function spectral_snr_squared(
        spectral_density::AbstractVector{<:Real},
        effective_psd::AbstractVector{<:Real},
        observation_time_sec::Real,
        df::Real
)
    return inner_product(
        spectral_density, spectral_density, effective_psd, observation_time_sec, df)
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
