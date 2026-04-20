using QuadGK

const SPEED_OF_LIGHT_KM_S = 299792.458

function E(z::Real, Omega_m::Real)
    return sqrt(Omega_m * (1 + z)^3 + (1 - Omega_m))
end

function comoving_distance(z::Real, H0::Real, Omega_m::Real)
    z == zero(z) && return zero(float(promote_type(typeof(z), typeof(H0), typeof(Omega_m))))
    integral, _ = quadgk(x -> inv(E(x, Omega_m)), zero(z), z)
    return (SPEED_OF_LIGHT_KM_S / H0) * integral
end

luminosity_distance(z::Real, H0::Real, Omega_m::Real) =
    (1 + z) * comoving_distance(z, H0, Omega_m)

function differential_comoving_volume(z::Real, H0::Real, Omega_m::Real)
    d_h = SPEED_OF_LIGHT_KM_S / H0
    d_c = comoving_distance(z, H0, Omega_m)
    return d_h * d_c^2 / E(z, Omega_m)
end

"""
    comoving_distance(z, H0, Omega_m, dist::RadialInterpolant) -> Real

Comoving distance using a precomputed `RadialInterpolant` of `w -> 1/E(w, Omega_m)`.
Equivalent to the scalar `quadgk` path but O(1) per call.
"""
comoving_distance(z::Real, H0::Real, Omega_m::Real, dist::RadialInterpolant) =
    (SPEED_OF_LIGHT_KM_S / H0) * integrate(dist, z, w -> inv(E(w, Omega_m)))

luminosity_distance(z::Real, H0::Real, Omega_m::Real, dist::RadialInterpolant) =
    (1 + z) * comoving_distance(z, H0, Omega_m, dist)

function differential_comoving_volume(
    z::Real, H0::Real, Omega_m::Real, dist::RadialInterpolant,
)
    d_h = SPEED_OF_LIGHT_KM_S / H0
    d_c = comoving_distance(z, H0, Omega_m, dist)
    return d_h * d_c^2 / E(z, Omega_m)
end

function gravitational_wave_distance(
    z::Real,
    luminosity_distance::Real,
    chi0::Real,
    chin::Real,
)
    return (chi0 + (1 - chi0) / (1 + z)^chin) * luminosity_distance
end
