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
