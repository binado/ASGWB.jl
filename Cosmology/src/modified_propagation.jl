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
