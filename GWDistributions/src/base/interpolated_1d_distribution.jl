using Distributions
using Random

export Interpolated1DDistribution, normalizer

"""
    Interpolated1DDistribution(x, y)

Tabulated unnormalized density on a 1-D grid with a cached trapezoid normalizer and
linear interpolant. `y` need not integrate to 1; [`logpdf`](@ref) and [`rand`](@ref)
normalize by [`normalizer`](@ref).
"""
struct Interpolated1DDistribution{
    I, TX <: AbstractVector, TY <: AbstractVector, TZ <: Real
} <: ContinuousUnivariateDistribution
    x::TX
    y::TY
    itp::I
    Z::TZ
end

function Interpolated1DDistribution(x::AbstractVector, y::AbstractVector)
    length(x) == length(y) || throw(DimensionMismatch(
        "Interpolated1DDistribution grid and density must have the same length"))
    length(x) >= 2 || throw(ArgumentError(
        "Interpolated1DDistribution requires at least two grid points"))
    Z = trapz(y, x)
    return Interpolated1DDistribution(x, y, LinearInterpolation(y, x), Z)
end

"""
    normalizer(d::Interpolated1DDistribution) -> Real

Cached trapezoid integral of the tabulated density (`trapz(d.y, d.x)`).
"""
normalizer(d::Interpolated1DDistribution) = d.Z

Base.minimum(d::Interpolated1DDistribution) = first(d.x)
Base.maximum(d::Interpolated1DDistribution) = last(d.x)
Base.eltype(d::Interpolated1DDistribution) = promote_type(eltype(d.y), typeof(d.Z))

function Distributions.insupport(d::Interpolated1DDistribution, value::Real)
    return minimum(d) <= value <= maximum(d)
end

@inline function _normalized_log_density(pdf_at_value, norm, tiny)
    return log(max(pdf_at_value / max(norm, tiny), tiny))
end

function Distributions.logpdf(d::Interpolated1DDistribution, value::Real)
    insupport(d, value) || return -Inf
    T = promote_type(eltype(d.y), typeof(d.Z))
    tiny = floatmin(T)
    return _normalized_log_density(d.itp(value), d.Z, tiny)
end

function Random.rand(rng::AbstractRNG, d::Interpolated1DDistribution)
    target = rand(rng) * d.Z
    cumulative = cumtrapz(d.y, d.x)
    x = d.x
    n = length(cumulative)
    idx = searchsortedlast(cumulative, target)
    idx <= 0 && return x[1]
    idx >= n && return x[end]
    c0, c1 = cumulative[idx], cumulative[idx + 1]
    x0, x1 = x[idx], x[idx + 1]
    c1 > c0 || return x0
    return x0 + (target - c0) * (x1 - x0) / (c1 - c0)
end
