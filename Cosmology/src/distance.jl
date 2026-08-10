using QuadGK

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

"""
    distance_and_volume_grid(c::AbstractCosmology, z_grid)
        -> (; comoving_distance, luminosity_distance, differential_comoving_volume)

Tabulate the three distance quantities on `z_grid` in a single pass, sharing one
`1/E(z)` evaluation and one cumulative trapezoid between them.

This is the efficient batched path for models that already evaluate and normalize
quantities on a redshift grid. Scalar distance calls use adaptive QuadGK integration
instead. The grid approximation and subsequent interpolation policy belong to the
caller.

Takes the grid **array**, not `(z_min, z_max, n)`, so the caller's grid is the grid used
— there is no way for tabulation and interpolation to disagree about nodes. The grid
must contain at least two strictly increasing nodes and start at zero, because the
cumulative comoving-distance integral assumes `d_c(0) = 0`.
"""
function distance_and_volume_grid(c::AbstractCosmology, z_grid::AbstractVector{<:Real})
    length(z_grid) >= 2 || throw(ArgumentError(
        "distance_and_volume_grid requires at least two grid points"))
    first(z_grid) == 0 || throw(ArgumentError(
        "distance_and_volume_grid requires a grid starting at zero"))
    all(diff(z_grid) .> 0) || throw(ArgumentError(
        "distance_and_volume_grid requires a strictly increasing grid"))
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
