using QuadGK

struct CosmologyCache{C <: AbstractCosmology, I <: CumulativeIntegral1D, TD <: Real}
    cosmology::C
    inv_E_integral::I
    d_h::TD
end

function CosmologyCache(cosmology::AbstractCosmology, z_grid::AbstractVector{<:Real})
    inv_E_integral = CumulativeIntegral1D(z_grid, z -> inv(E(z, cosmology)))
    d_h = SPEED_OF_LIGHT_KM_S / H0(cosmology)
    return CosmologyCache(cosmology, inv_E_integral, d_h)
end

function comoving_distance(z::Real, c::AbstractCosmology)
    Ez = E(z, c)
    pref = SPEED_OF_LIGHT_KM_S / (H0(c) * Ez)
    z == zero(z) && return zero(pref)
    integral, _ = quadgk(x -> inv(E(x, c)), zero(z), z)
    return pref * integral * Ez
end

function luminosity_distance(z::Real, c::AbstractCosmology)
    (1 + z) * comoving_distance(z, c)
end

function differential_comoving_volume(z::Real, c::AbstractCosmology)
    d_h = SPEED_OF_LIGHT_KM_S / H0(c)
    d_c = comoving_distance(z, c)
    return d_h * d_c^2 / E(z, c)
end

function comoving_distance(z::Real, cache::CosmologyCache)
    cache.d_h * cdf(cache.inv_E_integral, z)
end

function luminosity_distance(z::Real, cache::CosmologyCache)
    (1 + z) * comoving_distance(z, cache)
end

function differential_comoving_volume(z::Real, cache::CosmologyCache)
    d_c = comoving_distance(z, cache)
    return cache.d_h * d_c^2 / E(z, cache.cosmology)
end

"""
    comoving_distance(z, c::AbstractCosmology, dist::CumulativeIntegral1D) -> Real

Comoving distance using a precomputed [`CumulativeIntegral1D`](@ref) of
`w -> 1/E(w, c)`. Uses [`cdf`](@ref) which returns the exact integral under
the linear interpolant (analytic trapezoidal rule).
"""
function comoving_distance(z::Real, c::AbstractCosmology, dist::CumulativeIntegral1D)
    (SPEED_OF_LIGHT_KM_S / H0(c)) * cdf(dist, z)
end

function luminosity_distance(z::Real, c::AbstractCosmology, dist::CumulativeIntegral1D)
    (1 + z) * comoving_distance(z, c, dist)
end

function differential_comoving_volume(z::Real, c::AbstractCosmology, dist::CumulativeIntegral1D)
    d_h = SPEED_OF_LIGHT_KM_S / H0(c)
    d_c = comoving_distance(z, c, dist)
    return d_h * d_c^2 / E(z, c)
end

"""
    distance_and_volume_grid(c::AbstractCosmology, z_grid)
        -> (; comoving_distance, luminosity_distance, differential_comoving_volume)

Tabulate the three distance quantities on `z_grid` in a single pass, sharing one
`1/E(z)` evaluation and one cumulative trapezoid between them.

Equivalent to the [`CosmologyCache`](@ref) scalar path evaluated at every node — `d_c`
and `d_L` bit-for-bit, `dV_c/dz` to ~1 ulp, since `d_h · d_c² · inv_E` reuses the
already-computed `inv_E` instead of calling `E(z, c)` a second time. That second call is
what this replaces on the inference hot path; `CosmologyCache` itself stays, backing the
inverse-CDF sampling path.

Takes the grid **array**, not `(z_min, z_max, n)`, so the caller's grid is the grid used
— there is no way for the tabulation and the interpolation to disagree about nodes.
"""
function distance_and_volume_grid(c::AbstractCosmology, z_grid::AbstractVector{<:Real})
    inv_E = inv.(E.(z_grid, Ref(c)))
    d_h = SPEED_OF_LIGHT_KM_S / H0(c)
    d_c = d_h .* cumtrapz(z_grid, inv_E)
    return (;
        comoving_distance = d_c,
        luminosity_distance = (1 .+ z_grid) .* d_c,
        differential_comoving_volume = @. d_h * d_c^2 * inv_E
    )
end

"""
    gw_em_distance_ratio(z, prop) -> Real

Ratio `Ξ(z) = D_gw / D_L` between the gravitational-wave and electromagnetic luminosity
distances at redshift `z`. [`GR`](@ref) recovers `Ξ ≡ 1`; a [`ModifiedPropagation`](@ref)
applies the `(Ξ₀, Ξₙ)` factor

``\\Xi(z) = \\Xi_0 + (1 - \\Xi_0) / (1 + z)^{\\Xi_n}``.

This is the single source of truth for the propagation factor; the GW luminosity
distance is `gw_em_distance_ratio(z, prop) * D_L`.
"""
gw_em_distance_ratio(z::Real, Ξ₀::Real, Ξₙ::Real) = Ξ₀ + (1 - Ξ₀) / (1 + z)^Ξₙ
gw_em_distance_ratio(z::Real, ::GR) = one(z)
gw_em_distance_ratio(z::Real, p::ModifiedPropagation) = gw_em_distance_ratio(z, p.Ξ₀, p.Ξₙ)

"""
    apply_gw_distance_correction!(polarization_power, z, prop) -> polarization_power
    apply_gw_distance_correction(polarization_power, z, prop)  -> Matrix

Re-reference a `(nfreq, nsamples)` EM-distance polarization-power matrix to the fiducial GW luminosity
distance: `F_GW[:, j] = F_EM[:, j] / Ξ(z[j])²`.

Waveform catalogs store `|h₊|² + |h×|²` referenced to the *electromagnetic* luminosity
distance `D_L`, while the importance weights reweight the *gravitational-wave* distance
`D_gw = Ξ(z) D_L`. Under a non-GR fiducial propagation the two disagree by a constant
`Ξ_fid²`; applying this correction once at setup makes the polarization-power matrix agree with the
single weight formula (the one carrying `+2 log Ξ_fid`).

Identity under [`GR`](@ref); the bang form then returns `polarization_power` itself, while the
out-of-place form always copies.

**Not idempotent** — applying it twice gives `Ξ⁻⁴`. Prefer the out-of-place form in
reactive contexts (Pluto) where a cell may re-run.
"""
function apply_gw_distance_correction!(polarization_power::AbstractMatrix{<:Real},
        z::AbstractVector{<:Real}, prop::AbstractPropagation)
    _check_polarization_power_columns(polarization_power, z)
    @inbounds @views for j in eachindex(z)
        polarization_power[:, j] ./= gw_em_distance_ratio(z[j], prop)^2
    end
    return polarization_power
end

# Ξ ≡ 1: a true no-op resolved at compile time. The shape check is kept deliberately, so a
# dimension bug does not stay hidden until the day someone changes the fiducial propagation.
function apply_gw_distance_correction!(polarization_power::AbstractMatrix{<:Real},
        z::AbstractVector{<:Real}, ::GR)
    _check_polarization_power_columns(polarization_power, z)
    return polarization_power
end

function apply_gw_distance_correction(polarization_power::AbstractMatrix{<:Real},
        z::AbstractVector{<:Real}, prop::AbstractPropagation)
    return apply_gw_distance_correction!(copy(polarization_power), z, prop)
end

@inline function _check_polarization_power_columns(polarization_power, z)
    size(polarization_power, 2) == length(z) || throw(DimensionMismatch(
        "polarization-power matrix has $(size(polarization_power, 2)) sample columns but got $(length(z)) redshifts"))
    return nothing
end


