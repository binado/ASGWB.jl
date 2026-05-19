using Distributions
using Random

export OrderedUniformSourceMassPair, AlignedSpinChiSimple, BNS_MASS_LOW, BNS_MASS_HIGH,
       BNS_LAMBDA_HIGH, BNS_SPIN_A_MAX

const BNS_MASS_LOW = 1.1
const BNS_MASS_HIGH = 2.5
const BNS_LAMBDA_HIGH = 5000.0
const BNS_SPIN_A_MAX = 0.99

struct OrderedUniformSourceMassPair{T <: Real} <: ContinuousMultivariateDistribution
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

function Distributions.logpdf(
        d::OrderedUniformSourceMassPair,
        value::AbstractVector{<:Real}
)
    length(value) == 2 || throw(ArgumentError("ordered mass pair expects two coordinates"))
    return logpdf(d, (value[1], value[2]))
end

function Distributions._logpdf(
        d::OrderedUniformSourceMassPair,
        value::AbstractVector{<:Real}
)
    return logpdf(d, value)
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

struct AlignedSpinChiSimple{T <: Real} <: ContinuousUnivariateDistribution
    a_max::T
end

function AlignedSpinChiSimple(; a_max::Real = BNS_SPIN_A_MAX)
    a_max > 0 || throw(ArgumentError("a_max must be positive"))
    T = typeof(a_max)
    return AlignedSpinChiSimple(T(a_max))
end

Base.eltype(::Type{<:AlignedSpinChiSimple{T}}) where {T} = T
Base.eltype(d::AlignedSpinChiSimple) = typeof(d.a_max)

Base.minimum(d::AlignedSpinChiSimple) = -d.a_max
Base.maximum(d::AlignedSpinChiSimple) = d.a_max

Distributions.insupport(d::AlignedSpinChiSimple, value::Real) = abs(value) <= d.a_max

function Distributions.logpdf(d::AlignedSpinChiSimple, value::Real)
    insupport(d, value) || return -Inf
    T = typeof(d.a_max)
    eps_value = eps(T)
    density = -log(max(abs(value), eps_value) / d.a_max) / (T(2) * d.a_max)
    return log(max(density, floatmin(T)))
end

function Random.rand(rng::AbstractRNG, d::AlignedSpinChiSimple)
    T = typeof(d.a_max)
    magnitude = d.a_max * rand(rng, T) * rand(rng, T)
    return rand(rng, Bool) ? magnitude : -magnitude
end
