using Distributions
using Random

export AlignedSpinChiSimple, BNS_SPIN_A_MAX

const BNS_SPIN_A_MAX = 0.99

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
