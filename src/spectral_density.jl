using ForwardDiff

"""
    spectral_density(fluxes, merger_rate_per_sec; weights=nothing) -> Vector

Collapse per-sample flux contributions into a spectral density vector.

`fluxes` is a `(n_freq, n_samples)` matrix (column-major friendly). When `weights`
is `nothing`, samples are averaged uniformly: `mean_flux = sum(fluxes; dims=2) / n_samples`.
When `weights` is supplied, the contraction is `fluxes * weights / n_samples`
(no normalization of `weights`). The `0.4 = 2/5` prefactor captures the
average over the inclination angle.
"""
function spectral_density(
        fluxes::AbstractMatrix{<:Real},
        merger_rate_per_sec::Real;
        weights::Union{Nothing, AbstractVector{<:Real}} = nothing
)
    weights === nothing && return _spectral_density_unweighted(fluxes, merger_rate_per_sec)
    return _spectral_density_weighted(fluxes, merger_rate_per_sec, weights)
end

function _spectral_density_unweighted(
        fluxes::AbstractMatrix{<:Real},
        merger_rate_per_sec::Real
)
    n_samples = size(fluxes, 2)
    mean_flux = vec(sum(fluxes; dims = 2)) ./ n_samples
    return 0.4 .* merger_rate_per_sec .* mean_flux
end

function _spectral_density_weighted_generic(
        fluxes::AbstractMatrix{<:Real},
        merger_rate_per_sec::Real,
        weights::AbstractVector{<:Real}
)
    n_samples = size(fluxes, 2)
    mean_flux = (fluxes * weights) ./ n_samples
    return 0.4 .* merger_rate_per_sec .* mean_flux
end

function _spectral_density_weighted(
        fluxes::AbstractMatrix{<:Real},
        merger_rate_per_sec::Real,
        weights::AbstractVector{<:Real}
)
    return _spectral_density_weighted_generic(fluxes, merger_rate_per_sec, weights)
end

# Avoid `Matrix{Float64} * Vector{Dual}` here: on realistic caches the generic
# Dual matvec dominated ForwardDiff/Turing gradient profiles. Splitting primal
# values and partials lets BLAS handle the two dense contractions.
function _spectral_density_weighted(
        fluxes::AbstractMatrix{<:Real},
        merger_rate_per_sec::Real,
        weights::AbstractVector{<:ForwardDiff.Dual{Tag, V, N}}
) where {Tag, V, N}
    merger_rate_per_sec isa ForwardDiff.Dual &&
        return _spectral_density_weighted_generic(fluxes, merger_rate_per_sec, weights)
    rate_value = V(merger_rate_per_sec)
    rate_partials = ntuple(_ -> zero(V), Val(N))
    return _spectral_density_weighted_forwarddiff(
        fluxes,
        rate_value,
        rate_partials,
        weights
    )
end

function _spectral_density_weighted(
        fluxes::AbstractMatrix{<:Real},
        merger_rate_per_sec::ForwardDiff.Dual{Tag, V, N},
        weights::AbstractVector{<:ForwardDiff.Dual{Tag, V, N}}
) where {Tag, V, N}
    rate_value = ForwardDiff.value(merger_rate_per_sec)
    rate_partials = ntuple(j -> ForwardDiff.partials(merger_rate_per_sec)[j], Val(N))
    return _spectral_density_weighted_forwarddiff(
        fluxes,
        rate_value,
        rate_partials,
        weights
    )
end

function _spectral_density_weighted_forwarddiff(
        fluxes::AbstractMatrix{<:Real},
        rate_value::V,
        rate_partials::NTuple{N, V},
        weights::AbstractVector{<:ForwardDiff.Dual{Tag, V, N}}
) where {Tag, V, N}
    n_freq, n_samples = size(fluxes)
    length(weights) == n_samples ||
        throw(DimensionMismatch("weight length must match flux sample dimension"))

    primal_weights = Vector{V}(undef, n_samples)
    partial_weights = Matrix{V}(undef, n_samples, N)
    @inbounds for i in 1:n_samples
        w = weights[i]
        primal_weights[i] = ForwardDiff.value(w)
        p = ForwardDiff.partials(w)
        for j in 1:N
            partial_weights[i, j] = p[j]
        end
    end

    primal_sum = fluxes * primal_weights
    partial_sums = fluxes * partial_weights
    scale = V(0.4) / V(n_samples)
    out = Vector{ForwardDiff.Dual{Tag, V, N}}(undef, n_freq)
    @inbounds for i in 1:n_freq
        value = scale * rate_value * primal_sum[i]
        partials = ntuple(
            j -> scale *
                 (rate_partials[j] * primal_sum[i] + rate_value * partial_sums[i, j]),
            Val(N)
        )
        out[i] = ForwardDiff.Dual{Tag, V, N}(value, ForwardDiff.Partials(partials))
    end
    return out
end

"""1 Mpc in meters (IAU/CODATA convention)."""
const METERS_PER_MPC = 3.085677581e22

"""
    hubble_constant_si(H0_km_s_mpc::Real) -> Float64

Hubble constant in s⁻¹ from ``H_0`` in **km/s/Mpc** (same units as [`HyperParameters`](@ref).`H0`).
"""
function hubble_constant_si(H0_km_s_mpc::Real)
    return Float64(H0_km_s_mpc) * 1000.0 / METERS_PER_MPC
end

"""
    omegagw(spectral_density, frequency, H0::Real)
    omegagw(spectral_density, frequency, parameters::HyperParameters)

Dimensionless gravitational-wave energy density per logarithmic frequency,

``\\Omega_{\\mathrm{GW}}(f) = \\frac{4\\pi^2}{3 H_0^2} f^3 S_h(f)``,

where ``S_h(f)`` is the strain spectral density (same units as [`spectral_density`](@ref) on fluxes)
and ``H_0`` is the Hubble constant in **s⁻¹**.

The `H0::Real` method takes ``H_0`` in **km/s/Mpc** (matching the rest of this package) and converts
it internally to s⁻¹. The [`HyperParameters`](@ref) method uses `parameters.H0`.

`frequency` and `spectral_density` may be scalars or arrays; they broadcast together (e.g. same-length
vectors for one spectrum per frequency bin).
"""
function omegagw(spectral_density, frequency, H0::Real)
    h0_si = hubble_constant_si(H0)
    pre = 4 * pi^2 / (3 * h0_si^2)
    return @. pre * frequency^3 * spectral_density
end

function omegagw(spectral_density, frequency, parameters::HyperParameters)
    omegagw(spectral_density, frequency, parameters.H0)
end

"""
    spectral_snr_squared(spectral_density, effective_psd, frequencies, observation_time_sec, df) -> Real

Discrete matched-filter **SNR²** for a diagonal Gaussian noise model:

``\\mathrm{SNR}^2 = \\sum_i S_{h,i}^2 / \\sigma_i^2``,

where ``S_{h,i}`` is the strain spectral density in bin ``i`` and

``\\sigma_i = \\mathrm{effective\\_psd}_i / \\sqrt{2 T \\Delta f}``,

with observation time ``T`` in seconds, frequency bin width ``\\Delta f =`` `df` in Hz, and
network [`effective_psd`](@ref) in the same convention as [`gaussian_bin_scale`](@ref) and
[`ObservationConfig`](@ref) (per-bin `sgwb_scale` from [`build_observation_config`](@ref) matches
this `σ` path when `df` is the same width used there, e.g. from [`frequency_bin_width`](@ref) on
the analysis frequency grid).

`frequencies` is part of the signature for call-site consistency with per-bin vectors; it is not
read by this implementation.
"""
function spectral_snr_squared(
        spectral_density::AbstractVector{<:Real},
        effective_psd::AbstractVector{<:Real},
        frequencies::AbstractVector{<:Real},
        observation_time_sec::Real,
        df::Real
)
    denom = sqrt(2 * observation_time_sec * df)
    sgwb_scale = effective_psd ./ denom
    return sum(abs2, spectral_density ./ sgwb_scale)
end

"""
    spectral_snr(spectral_density, effective_psd, frequencies, observation_time_sec, df) -> Real

``\\mathrm{SNR} = \\sqrt{\\mathrm{SNR}^2}`` with ``\\mathrm{SNR}^2`` from
[`spectral_snr_squared`](@ref).
"""
function spectral_snr(
        spectral_density::AbstractVector{<:Real},
        effective_psd::AbstractVector{<:Real},
        frequencies::AbstractVector{<:Real},
        observation_time_sec::Real,
        df::Real
)
    return sqrt(spectral_snr_squared(
        spectral_density,
        effective_psd,
        frequencies,
        observation_time_sec,
        df
    ))
end
