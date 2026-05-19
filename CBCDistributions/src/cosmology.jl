using QuadGK

const SPEED_OF_LIGHT_KM_S = 299792.458

struct Cosmology{TH0 <: Real, TΩm <: Real}
    H0::TH0
    Ωm::TΩm
end

Cosmology(h::NamedTuple) = Cosmology(h.H0, h.Ωm)

struct CosmologyCache{C <: Cosmology, I <: CumulativeIntegral1D}
    cosmology::C
    inv_E_integral::I
end

function CosmologyCache(cosmology::Cosmology, z_grid::AbstractVector{<:Real})
    inv_E_integral = CumulativeIntegral1D(z_grid, z -> inv(E(z, cosmology.Ωm)))
    return CosmologyCache(cosmology, inv_E_integral)
end

function CosmologyCache(h::NamedTuple, z_grid::AbstractVector{<:Real})
    CosmologyCache(Cosmology(h), z_grid)
end

function E(z::Real, Ωm::Real)
    return sqrt(Ωm * (1 + z)^3 + (1 - Ωm))
end

function comoving_distance(z::Real, H0::Real, Ωm::Real)
    z == zero(z) && return zero(float(promote_type(typeof(z), typeof(H0), typeof(Ωm))))
    integral, _ = quadgk(x -> inv(E(x, Ωm)), zero(z), z)
    return (SPEED_OF_LIGHT_KM_S / H0) * integral
end

function luminosity_distance(z::Real, H0::Real, Ωm::Real)
    (1 + z) * comoving_distance(z, H0, Ωm)
end

function differential_comoving_volume(z::Real, H0::Real, Ωm::Real)
    d_h = SPEED_OF_LIGHT_KM_S / H0
    d_c = comoving_distance(z, H0, Ωm)
    return d_h * d_c^2 / E(z, Ωm)
end

function comoving_distance(z::Real, cache::CosmologyCache)
    comoving_distance(z, cache.cosmology.H0, cache.cosmology.Ωm, cache.inv_E_integral)
end

function luminosity_distance(z::Real, cache::CosmologyCache)
    luminosity_distance(z, cache.cosmology.H0, cache.cosmology.Ωm, cache.inv_E_integral)
end

function differential_comoving_volume(z::Real, cache::CosmologyCache)
    differential_comoving_volume(
        z,
        cache.cosmology.H0,
        cache.cosmology.Ωm,
        cache.inv_E_integral
    )
end

"""
    comoving_distance(z, H0, Ωm, dist::CumulativeIntegral1D) -> Real

Comoving distance using a precomputed [`CumulativeIntegral1D`](@ref) of
`w -> 1/E(w, Ωm)`. Uses [`cdf`](@ref) which returns the exact integral under
the linear interpolant (analytic trapezoidal rule).
"""
function comoving_distance(z::Real, H0::Real, Ωm::Real, dist::CumulativeIntegral1D)
    (SPEED_OF_LIGHT_KM_S / H0) * cdf(dist, z)
end

function luminosity_distance(z::Real, H0::Real, Ωm::Real, dist::CumulativeIntegral1D)
    (1 + z) * comoving_distance(z, H0, Ωm, dist)
end

function differential_comoving_volume(
        z::Real,
        H0::Real,
        Ωm::Real,
        dist::CumulativeIntegral1D
)
    d_h = SPEED_OF_LIGHT_KM_S / H0
    d_c = comoving_distance(z, H0, Ωm, dist)
    return d_h * d_c^2 / E(z, Ωm)
end

function gravitational_wave_distance(
        z::Real,
        luminosity_distance::Real,
        Ξ₀::Real,
        Ξₙ::Real
)
    return (Ξ₀ + (1 - Ξ₀) / (1 + z)^Ξₙ) * luminosity_distance
end
