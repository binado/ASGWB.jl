using Distributions
using Random

struct BrokenPowerLaw{T <: Real, P1 <: TruncatedPowerLaw, P2 <: TruncatedPowerLaw} <:
       ContinuousUnivariateDistribution
    low_weight::T  # z1 / (z1 + z2): P(a draw lands below the break)
    log_norm::T    # log(z1 + z2)
    lower::P1      # truncated power law on [low, m_break]
    upper::P2      # truncated power law on [m_break, high]
end

function BrokenPowerLaw(α1::Real, α2::Real, m_break::Real, low::Real, high::Real)
    T = promote_type(
        Float64, typeof(α1), typeof(α2), typeof(m_break), typeof(low), typeof(high))
    0 < low < m_break < high ||
        throw(ArgumentError("broken power-law bounds must satisfy 0 < low < m_break < high"))
    lower = TruncatedPowerLaw(T(α1), T(m_break), T(low), T(m_break))
    upper = TruncatedPowerLaw(T(α2), T(m_break), T(m_break), T(high))
    z1 = normalizer(lower)
    z2 = normalizer(upper)
    z = z1 + z2
    z > 0 || throw(ArgumentError("broken power-law normalizer must be positive"))
    return BrokenPowerLaw{T, typeof(lower), typeof(upper)}(z1 / z, log(z), lower, upper)
end

Base.minimum(d::BrokenPowerLaw) = d.lower.low
Base.maximum(d::BrokenPowerLaw) = d.upper.high
Base.eltype(::Type{<:BrokenPowerLaw{T}}) where {T} = T
Base.eltype(d::BrokenPowerLaw) = eltype(typeof(d))

function Distributions.insupport(d::BrokenPowerLaw, value::Real)
    return d.lower.low <= value < d.upper.high
end

function Distributions.logpdf(d::BrokenPowerLaw, value::Real)
    insupport(d, value) || return -Inf
    if value < d.lower.high
        return -d.lower.α * log(value / d.lower.pivot) - d.log_norm
    end
    return -d.upper.α * log(value / d.upper.pivot) - d.log_norm
end

function Random.rand(rng::AbstractRNG, d::BrokenPowerLaw)
    if rand(rng) <= d.low_weight
        return rand(rng, d.lower)
    end
    return rand(rng, d.upper)
end
