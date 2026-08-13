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

"""
    differential_comoving_volume(z, c) -> Real

Differential comoving volume element `4π · d_h · d_c(z)² / E(z)`, i.e. the
solid-angle-integrated `dV_c/dz` in Mpc³ per unit redshift. The `4π` is owned here, not
by the redshift-distribution consumers, so a tabulated grid from
[`distance_and_volume_grid`](@ref) and this scalar function agree.
"""
function differential_comoving_volume(z::Real, c::AbstractCosmology)
    d_h = SPEED_OF_LIGHT_KM_S / H0(c)
    d_c = comoving_distance(z, c)
    return 4π * d_h * d_c^2 / E(z, c)
end

"""
    distance_and_volume_grid(c::AbstractCosmology, z)
        -> (; comoving_distance, luminosity_distance, differential_comoving_volume)

Tabulate the three distance quantities on `z` in a single pass, sharing one
`1/E(z)` evaluation and one cumulative trapezoid between them.

`differential_comoving_volume` is the solid-angle-integrated `4π · dV_c/dz` in
Mpc³ per unit redshift — the `4π` lives here, not in the redshift-distribution
consumers, matching the Python `astrogwb` stack.

This is the efficient batched path for models that already evaluate and normalize
quantities on a redshift grid. Scalar distance calls use adaptive QuadGK integration
instead. The grid approximation and subsequent interpolation policy belong to the
caller.

Takes the grid **array**, not `(z_min, z_max, n)`, so the caller's grid is the grid used
— there is no way for tabulation and interpolation to disagree about nodes. The grid
must contain at least two strictly increasing nodes and start at zero, because the
cumulative comoving-distance integral assumes `d_c(0) = 0`.
"""
function distance_and_volume_grid(c::AbstractCosmology, z::AbstractVector{<:Real})
    length(z) >= 2 || throw(ArgumentError(
        "distance_and_volume_grid requires at least two grid points"))
    first(z) == 0 || throw(ArgumentError(
        "distance_and_volume_grid requires a grid starting at zero"))
    all(diff(z) .> 0) || throw(ArgumentError(
        "distance_and_volume_grid requires a strictly increasing grid"))
    inv_E = inv.(E.(z, Ref(c)))
    d_h = SPEED_OF_LIGHT_KM_S / H0(c)
    d_c = d_h .* cumtrapz(inv_E, z)
    return (;
        comoving_distance = d_c,
        luminosity_distance = (1 .+ z) .* d_c,
        differential_comoving_volume = @. 4π * d_h * d_c^2 * inv_E
    )
end
