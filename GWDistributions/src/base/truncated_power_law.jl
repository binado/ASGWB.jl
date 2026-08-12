using Distributions
using Random

const _SQRT_EPS_FLOAT64 = sqrt(eps(Float64))

@inline function _power_integral(low::Real, high::Real, exponent::Real)
    high > low || return zero(promote_type(typeof(low), typeof(high), typeof(exponent)))
    T = promote_type(typeof(low), typeof(high), typeof(exponent))
    a = exponent + one(T)
    if abs(a) <= _SQRT_EPS_FLOAT64
        return log(high / low)
    end
    return (high^a - low^a) / a
end

struct TruncatedPowerLaw{T <: Real} <: ContinuousUnivariateDistribution
    α::T       # power-law slope (density ∝ (m / pivot)^(-α))
    pivot::T   # shared pivot for scale continuity
    low::T
    high::T
end

function TruncatedPowerLaw(α::Real, pivot::Real, low::Real, high::Real)
    T = promote_type(Float64, typeof(α), typeof(pivot), typeof(low), typeof(high))
    α = T(α);
    pivot = T(pivot);
    low = T(low);
    high = T(high)
    0 < low < high || throw(ArgumentError("bounds must satisfy 0 < low < high"))
    d = TruncatedPowerLaw{T}(α, pivot, low, high)
    normalizer(d) > 0 ||
        throw(ArgumentError("truncated power-law normalizer must be positive"))
    return d
end

Base.minimum(d::TruncatedPowerLaw) = d.low
Base.maximum(d::TruncatedPowerLaw) = d.high
Base.eltype(::Type{<:TruncatedPowerLaw{T}}) where {T} = T
Base.eltype(d::TruncatedPowerLaw) = eltype(typeof(d))

function Distributions.insupport(d::TruncatedPowerLaw, value::Real)
    return d.low <= value < d.high
end

function Distributions.logpdf(d::TruncatedPowerLaw, value::Real)
    insupport(d, value) || return -Inf
    return -d.α * log(value / d.pivot) - log(normalizer(d))
end

@inline function normalizer(d::TruncatedPowerLaw)
    d.high > d.low || return zero(d.α)
    a = one(d.α) - d.α
    if abs(a) <= _SQRT_EPS_FLOAT64
        return d.pivot * log(d.high / d.low)
    end
    return d.pivot * ((d.high / d.pivot)^a - (d.low / d.pivot)^a) / a
end

function _rand_scaled_power(rng::AbstractRNG, low::Real, high::Real, exponent::Real)
    u = rand(rng)
    a = exponent + 1
    if abs(a) <= _SQRT_EPS_FLOAT64
        return low * exp(u * log(high / low))
    end
    return (low^a + u * (high^a - low^a))^(1 / a)
end

function Random.rand(rng::AbstractRNG, d::TruncatedPowerLaw)
    return _rand_scaled_power(rng, d.low, d.high, -d.α)
end
