using ForwardDiff

# ---------------------------------------------------------------------------
# Inclination averaging: a property of the *catalog*, not of the kernel. The
# spectral density needs `⟨|h₊|² + |h×|²⟩` averaged over the inclination angle
# ι, but `fluxes` already carries whatever ι convention the waveform catalog
# was generated under. Only two conventions exist in practice, so the choice is
# a type token resolved to a scalar prefactor at the public boundary, mirroring
# the `AbstractPropagation` idiom in `Cosmology/src/model.jl`.
# ---------------------------------------------------------------------------

"""Abstract supertype for inclination-averaging conventions."""
abstract type AbstractAverageMode end

"""
Catalog waveforms were generated face-on (`ι ≡ 0`), so the inclination average
must still be applied analytically:

``\\langle |h_+|^2 + |h_\\times|^2 \\rangle_\\iota / (|h_+|^2 + |h_\\times|^2)|_{\\iota=0} = 2/5``.
"""
struct AnalyticInclination <: AbstractAverageMode end

"""
Catalog waveforms sample the inclination angle, so the Monte Carlo average over
catalog samples already performs the inclination average and the prefactor is 1.
"""
struct CatalogInclination <: AbstractAverageMode end

Base.broadcastable(m::AbstractAverageMode) = Ref(m)

"""
    inclination_factor(mode::AbstractAverageMode) -> Float64

Scalar prefactor applied to the sample-averaged flux under `mode`.
"""
inclination_factor(::AnalyticInclination) = 0.4
inclination_factor(::CatalogInclination) = 1.0

"""Supported configurable averaging modes (registration order)."""
const SUPPORTED_AVERAGE_MODES = (AnalyticInclination, CatalogInclination)

"""
    average_mode_config_name(::Type{M}) -> String

Config/TOML name for averaging mode `M`. These strings match the `AverageMode`
literals used by the Python `astrogwb` package, so a convention travels across
both implementations unchanged.
"""
average_mode_config_name(::Type{AnalyticInclination}) = "analytic_inclination"
average_mode_config_name(::Type{CatalogInclination}) = "catalog_inclination"

const _AVERAGE_MODE_BY_CONFIG_NAME = Dict(
    average_mode_config_name(M) => M for M in SUPPORTED_AVERAGE_MODES
)

"""
    average_mode_type(name::AbstractString) -> Type{<:AbstractAverageMode}

Resolve a config/TOML averaging-mode name to a concrete subtype.
"""
function average_mode_type(name::AbstractString)
    M = get(_AVERAGE_MODE_BY_CONFIG_NAME, String(name), nothing)
    M === nothing && throw(
        ArgumentError(
        "unknown average mode \"$(name)\"; valid choices: $(sort(collect(keys(_AVERAGE_MODE_BY_CONFIG_NAME))))",
    ),
    )
    return M
end

"""
    spectral_density(fluxes, merger_rate_per_sec; weights=nothing,
                     average_mode=AnalyticInclination()) -> Vector

Collapse per-sample flux contributions into a spectral density vector.

`fluxes` is a `(nfreq, nsamples)` matrix (column-major friendly). When `weights`
is `nothing`, samples are averaged uniformly: `mean_flux = sum(fluxes; dims=2) / nsamples`.
When `weights` is supplied, the contraction is `fluxes * weights / nsamples`
(no normalization of `weights`).

`average_mode` selects the inclination-averaging convention of the catalog that
produced `fluxes`; the result is scaled by [`inclination_factor`](@ref). The
default [`AnalyticInclination`](@ref) assumes face-on waveforms and applies
`2/5`; pass [`CatalogInclination`](@ref) when the catalog samples ι, otherwise
the result is a factor `2.5` low. See [`average_mode`](@ref) for deriving the
convention from a loaded catalog.
"""
function spectral_density(
        fluxes::AbstractMatrix{<:Real},
        merger_rate_per_sec::Real;
        weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
        average_mode::AbstractAverageMode = AnalyticInclination()
)
    return _spectral_density(
        fluxes, merger_rate_per_sec, weights, inclination_factor(average_mode))
end

# `factor` is a required trailing positional in every `_spectral_density`
# method: a branch left un-updated is then a `MethodError`, not a silently
# stale `0.4`.
function _spectral_density(
        fluxes::AbstractMatrix{<:Real},
        merger_rate_per_sec::Real,
        ::Nothing,
        factor::Real
)
    nsamples = size(fluxes, 2)
    mean_flux = vec(sum(fluxes; dims = 2)) ./ nsamples
    return factor .* merger_rate_per_sec .* mean_flux
end

function _spectral_density(
        fluxes::AbstractMatrix{<:Real},
        merger_rate_per_sec::Real,
        weights::AbstractVector{<:Real},
        factor::Real
)
    nsamples = size(fluxes, 2)
    mean_flux = (fluxes * weights) ./ nsamples
    return factor .* merger_rate_per_sec .* mean_flux
end

# Avoid `Matrix{Float64} * Vector{Dual}` here: on realistic caches the generic
# Dual matvec dominated ForwardDiff/Turing gradient profiles. Splitting primal
# values and partials lets BLAS handle the two dense contractions (see
# `_spectral_density_forwarddiff`). The rate may itself be a same-tag Dual.
function _spectral_density(
        fluxes::AbstractMatrix{<:Real},
        merger_rate_per_sec::Real,
        weights::AbstractVector{<:ForwardDiff.Dual{Tag, V, N}},
        factor::Real
) where {Tag, V, N}
    rate_value, rate_partials = _rate_value_partials(
        merger_rate_per_sec, Tag, V, Val(N))
    # Convert here so the inner kernel takes `factor::V`: a `Dual` prefactor is
    # then a `MethodError`, since the averaging convention is a constant that is
    # never differentiated.
    return _spectral_density_forwarddiff(
        fluxes, rate_value, rate_partials, weights, V(factor))
end

# Extract `(value, partials)` from a rate that is either a plain `Real` (zero
# partials) or a `Dual` whose tag/lane count match the weights'. Mismatched-tag
# Duals are not supported in this dispatch family.
function _rate_value_partials(x::Real, ::Type, ::Type{V}, ::Val{N}) where {V, N}
    (V(x), ntuple(_ -> zero(V), Val(N)))
end

function _rate_value_partials(
        x::ForwardDiff.Dual{Tag, V, N}, ::Type{Tag}, ::Type{V}, ::Val{N}
) where {Tag, V, N}
    (ForwardDiff.value(x), Tuple(ForwardDiff.partials(x)))
end

# See comment above `_spectral_density` for the Dual-weighted dispatch rationale.
function _spectral_density_forwarddiff(
        fluxes::AbstractMatrix{<:Real},
        rate_value::V,
        rate_partials::NTuple{N, V},
        weights::AbstractVector{<:ForwardDiff.Dual{Tag, V, N}},
        factor::V
) where {Tag, V, N}
    nfreq, nsamples = size(fluxes)
    length(weights) == nsamples ||
        throw(DimensionMismatch("weight length must match flux sample dimension"))

    # Pack value + partials into one contiguous `(nsamples, N+1)` buffer so a single
    # gemm `fluxes * weight_block` yields the primal sum (column 1) and every partial
    # sum (columns 2:N+1) at once, instead of a separate gemv + gemm.
    weight_block = Matrix{V}(undef, nsamples, N + 1)
    @inbounds for i in 1:nsamples
        w = weights[i]
        weight_block[i, 1] = ForwardDiff.value(w)
        p = ForwardDiff.partials(w)
        for j in 1:N
            weight_block[i, j + 1] = p[j]
        end
    end

    sums = fluxes * weight_block
    scale = factor / V(nsamples)
    out = Vector{ForwardDiff.Dual{Tag, V, N}}(undef, nfreq)
    @inbounds for i in 1:nfreq
        primal_sum = sums[i, 1]
        value = scale * rate_value * primal_sum
        partials = ntuple(
            j -> scale * (rate_partials[j] * primal_sum + rate_value * sums[i, j + 1]),
            Val(N)
        )
        out[i] = ForwardDiff.Dual{Tag, V, N}(value, ForwardDiff.Partials(partials))
    end
    return out
end

"""
    Ωgw(spectral_density, frequency, H0::Real)

Dimensionless gravitational-wave energy density per logarithmic frequency,

``\\Omega_{\\mathrm{GW}}(f) = \\frac{4\\pi^2}{3 H_0^2} f^3 S_h(f)``,

where ``S_h(f)`` is the strain spectral density (same units as [`spectral_density`](@ref) on fluxes)
and ``H_0`` is the Hubble constant in **s⁻¹**.

``H_0`` is passed in **km/s/Mpc** (matching hyperparameter `H0` and [`LambdaCDM`](@ref).`H0`)
and converted internally via [`Cosmology.hubble_constant_si`](@ref).

`frequency` and `spectral_density` may be scalars or arrays; they broadcast together (e.g. same-length
vectors for one spectrum per frequency bin).
"""
function Ωgw(spectral_density, frequency, H0::Real)
    h0_si = Cosmology.hubble_constant_si(H0)
    pre = 4 * pi^2 / (3 * h0_si^2)
    return @. pre * frequency^3 * spectral_density
end
