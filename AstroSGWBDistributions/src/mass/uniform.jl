using Distributions
using Random

export OrderedUniformSourceMassPair, BNS_MASS_LOW, BNS_MASS_HIGH

const BNS_MASS_LOW = 1.1
const BNS_MASS_HIGH = 2.5

abstract type SourceMassPairDistribution <: ContinuousMultivariateDistribution end

struct OrderedUniformSourceMassPair{T <: Real} <: SourceMassPairDistribution
    low::T
    high::T
end

function OrderedUniformSourceMassPair(;
        low::Real = BNS_MASS_LOW,
        high::Real = BNS_MASS_HIGH
)
    low < high || throw(ArgumentError("low must be smaller than high"))
    T = promote_type(typeof(low), typeof(high))
    return OrderedUniformSourceMassPair(T(low), T(high))
end

Base.length(::OrderedUniformSourceMassPair) = 2
Base.size(::OrderedUniformSourceMassPair) = (2,)
Base.eltype(::Type{<:OrderedUniformSourceMassPair{T}}) where {T} = T
Base.eltype(d::OrderedUniformSourceMassPair) = typeof(d.low)

function Distributions.insupport(d::OrderedUniformSourceMassPair, value::NTuple{2, <:Real})
    m1, m2 = value
    return m1 >= m2 && m2 >= d.low && m1 <= d.high
end

function Distributions.insupport(
        d::OrderedUniformSourceMassPair,
        value::AbstractVector{<:Real}
)
    length(value) == 2 || return false
    return insupport(d, (value[1], value[2]))
end

function Distributions.logpdf(d::OrderedUniformSourceMassPair, value::NTuple{2, <:Real})
    insupport(d, value) || return -Inf
    T = typeof(d.low)
    return log(T(2)) - T(2) * log(d.high - d.low)
end

function Distributions._logpdf(
        d::OrderedUniformSourceMassPair,
        x::AbstractVector{<:Real}
)
    return logpdf(d, (x[1], x[2]))
end

function Random.rand(rng::AbstractRNG, d::OrderedUniformSourceMassPair)
    T = typeof(d.low)
    span = d.high - d.low
    x = d.low + span * rand(rng, T)
    y = d.low + span * rand(rng, T)
    return x >= y ? [x, y] : [y, x]
end

function Distributions._rand!(
        rng::AbstractRNG,
        d::OrderedUniformSourceMassPair,
        x::AbstractVector{<:Real}
)
    length(x) == 2 || throw(ArgumentError("ordered mass pair expects length-2 output"))
    T = typeof(d.low)
    span = d.high - d.low
    a = d.low + span * rand(rng, T)
    b = d.low + span * rand(rng, T)
    if a >= b
        x[1] = a
        x[2] = b
    else
        x[1] = b
        x[2] = a
    end
    return x
end
