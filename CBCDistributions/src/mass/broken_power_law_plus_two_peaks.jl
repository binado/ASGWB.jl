using Distributions
using QuadGK
using Random

const DEFAULT_BBH_M_HIGH = 300.0
const _SQRT_EPS_FLOAT64 = sqrt(eps(Float64))
const _PLANCK_Q_GAUSS_16 = QuadGK.gauss(Float64, 16)

@inline function _power_integral(low::Real, high::Real, exponent::Real)
    high > low || return zero(promote_type(typeof(low), typeof(high), typeof(exponent)))
    T = promote_type(typeof(low), typeof(high), typeof(exponent))
    a = exponent + one(T)
    if abs(a) <= _SQRT_EPS_FLOAT64
        return log(high / low)
    end
    return (high^a - low^a) / a
end

@inline function _broken_power_integral(
        α::Real,
        low::Real,
        high::Real,
        m_break::Real
)
    high > low ||
        return zero(promote_type(typeof(α), typeof(low), typeof(high), typeof(m_break)))
    T = promote_type(typeof(α), typeof(low), typeof(high), typeof(m_break))
    a = one(T) - α
    if abs(a) <= _SQRT_EPS_FLOAT64
        return m_break * log(high / low)
    end
    return m_break * ((high / m_break)^a - (low / m_break)^a) / a
end

struct BrokenPowerLaw{T <: Real} <: ContinuousUnivariateDistribution
    α1::T          # slope below the break
    α2::T          # slope above the break
    m_break::T     # break location and shared pivot
    low::T
    high::T
    low_weight::T  # z1 / (z1 + z2): P(a draw lands below the break)
    log_norm::T    # log(z1 + z2)
end

function BrokenPowerLaw(α1::Real, α2::Real, m_break::Real, low::Real, high::Real)
    T = promote_type(
        Float64, typeof(α1), typeof(α2), typeof(m_break), typeof(low), typeof(high))
    0 < low < m_break < high ||
        throw(ArgumentError("broken power-law bounds must satisfy 0 < low < m_break < high"))
    z1 = _broken_power_integral(T(α1), T(low), T(m_break), T(m_break))
    z2 = _broken_power_integral(T(α2), T(m_break), T(high), T(m_break))
    z = z1 + z2
    z > 0 || throw(ArgumentError("broken power-law normalizer must be positive"))
    return BrokenPowerLaw{T}(T(α1), T(α2), T(m_break), T(low), T(high), z1 / z, log(z))
end

Base.minimum(d::BrokenPowerLaw) = d.low
Base.maximum(d::BrokenPowerLaw) = d.high
Base.eltype(::Type{<:BrokenPowerLaw{T}}) where {T} = T
Base.eltype(d::BrokenPowerLaw) = eltype(typeof(d))

function Distributions.insupport(d::BrokenPowerLaw, value::Real)
    return d.low <= value < d.high
end

function Distributions.logpdf(d::BrokenPowerLaw, value::Real)
    insupport(d, value) || return -Inf
    α = value < d.m_break ? d.α1 : d.α2
    return -α * log(value / d.m_break) - d.log_norm
end

function Random.rand(rng::AbstractRNG, d::BrokenPowerLaw)
    if rand(rng) <= d.low_weight
        return _rand_scaled_power(rng, d.low, d.m_break, -d.α1)
    end
    return _rand_scaled_power(rng, d.m_break, d.high, -d.α2)
end

struct DefaultBBHPrimaryMass{T <: Real, B, G, N <: Real} <:
       ContinuousUnivariateDistribution
    m1_low::T
    δm1::T
    λ0::T
    λ1::T
    λ2::T
    m_high::T
    broken::B   # untapered broken power law (weight λ0)
    peak1::G    # truncated Gaussian peak (weight λ1)
    peak2::G    # truncated Gaussian peak (weight λ2)
    log_taper_norm::N
end

function DefaultBBHPrimaryMass(;
        α1::Real,
        α2::Real,
        m_break::Real,
        μ1::Real,
        σ1::Real,
        μ2::Real,
        σ2::Real,
        m1_low::Real,
        δm1::Real,
        λ0::Real,
        λ1::Real,
        m_high::Real = DEFAULT_BBH_M_HIGH
)
    T = promote_type(
        Float64,
        typeof(α1), typeof(α2), typeof(m_break), typeof(μ1), typeof(σ1),
        typeof(μ2), typeof(σ2), typeof(m1_low), typeof(δm1), typeof(λ0),
        typeof(λ1), typeof(m_high)
    )
    α1 = T(α1)
    α2 = T(α2)
    m_break = T(m_break)
    μ1 = T(μ1)
    σ1 = T(σ1)
    μ2 = T(μ2)
    σ2 = T(σ2)
    m1_low = T(m1_low)
    δm1 = T(δm1)
    λ0 = T(λ0)
    λ1 = T(λ1)
    λ2 = one(T) - λ0 - λ1
    m_high = T(m_high)
    _validate_primary_parameters(m1_low, m_break, m_high, σ1, σ2, δm1, λ0, λ1, λ2)
    broken, peak1,
    peak2 = _primary_components(
        α1, α2, m_break, μ1, σ1, μ2, σ2, m1_low, m_high)
    primary = DefaultBBHPrimaryMass{T, typeof(broken), typeof(peak1), T}(
        m1_low, δm1, λ0, λ1, λ2, m_high, broken, peak1, peak2, zero(T))
    log_taper_norm = log(_primary_taper_normalizer(primary))
    return DefaultBBHPrimaryMass{T, typeof(broken), typeof(peak1), T}(
        m1_low, δm1, λ0, λ1, λ2, m_high, broken, peak1, peak2, log_taper_norm)
end

function _validate_primary_parameters(
        m1_low::Real,
        m_break::Real,
        m_high::Real,
        σ1::Real,
        σ2::Real,
        δm1::Real,
        λ0::Real,
        λ1::Real,
        λ2::Real
)
    0 < m1_low < m_break < m_high ||
        throw(ArgumentError("mass bounds must satisfy 0 < m1_low < m_break < m_high"))
    σ1 > 0 || throw(ArgumentError("σ1 must be positive"))
    σ2 > 0 || throw(ArgumentError("σ2 must be positive"))
    δm1 >= 0 || throw(ArgumentError("δm1 must be non-negative"))
    λ0 >= 0 || throw(ArgumentError("λ0 must be non-negative"))
    λ1 >= 0 || throw(ArgumentError("λ1 must be non-negative"))
    λ2 >= 0 || throw(ArgumentError("1 - λ0 - λ1 must be non-negative"))
    λ0 + λ1 + λ2 > 0 ||
        throw(ArgumentError("at least one mixture weight must be positive"))
    return nothing
end

Base.minimum(d::DefaultBBHPrimaryMass) = d.m1_low
Base.maximum(d::DefaultBBHPrimaryMass) = d.m_high
Base.eltype(::Type{<:DefaultBBHPrimaryMass{T}}) where {T} = T
Base.eltype(d::DefaultBBHPrimaryMass) = eltype(typeof(d))

function Distributions.insupport(d::DefaultBBHPrimaryMass, value::Real)
    return d.m1_low <= value < d.m_high
end

function _primary_components(
        α1::Real,
        α2::Real,
        m_break::Real,
        μ1::Real,
        σ1::Real,
        μ2::Real,
        σ2::Real,
        m1_low::Real,
        m_high::Real
)
    broken = BrokenPowerLaw(α1, α2, m_break, m1_low, m_high)
    peak1 = truncated(Normal(μ1, σ1), m1_low, m_high)
    peak2 = truncated(Normal(μ2, σ2), m1_low, m_high)
    return broken, peak1, peak2
end

@inline function _logsumexp3(a, b, c)
    m = max(a, b, c)
    isfinite(m) || return m
    return m + log(exp(a - m) + exp(b - m) + exp(c - m))
end

# Untapered log-density: the λ-weighted mixture of the broken power law and the two
# truncated Gaussians. Kept as an explicit logsumexp over concretely-typed component
# fields so the whole logpdf path stays type-stable and allocation-free (a heterogeneous
# `MixtureModel` would infer to `Any` and box on every call in the likelihood hot path).
@inline function _primary_log_untapered(d::DefaultBBHPrimaryMass, m::Real)
    return _logsumexp3(
        log(d.λ0) + logpdf(d.broken, m),
        log(d.λ1) + logpdf(d.peak1, m),
        log(d.λ2) + logpdf(d.peak2, m))
end

@inline function _primary_log_unnormalized(d::DefaultBBHPrimaryMass, m::Real)
    insupport(d, m) || return -Inf
    log_s = _log_planck_taper(m, d.m1_low, d.δm1)
    log_s == -Inf && return -Inf
    return _primary_log_untapered(d, m) + log_s
end

function _primary_taper_normalizer(d::DefaultBBHPrimaryMass)
    d.δm1 == 0 && return one(promote_type(typeof(d.m1_low), typeof(d.log_taper_norm)))

    f = m -> exp(_primary_log_unnormalized(d, m))
    taper_high = min(d.m1_low + d.δm1, d.m_high)
    z = zero(promote_type(typeof(d.m1_low), typeof(d.log_taper_norm)))
    if taper_high > d.m1_low
        z += first(quadgk(f, d.m1_low, taper_high))
    end
    if taper_high < d.m_high
        z += first(quadgk(f, taper_high, d.m_high))
    end
    return z
end

function Distributions.logpdf(d::DefaultBBHPrimaryMass, value::Real)
    logp = _primary_log_unnormalized(d, value)
    logp == -Inf && return -Inf
    return logp - d.log_taper_norm
end

struct DefaultBBHMassPair{P <: DefaultBBHPrimaryMass, T <: Real} <:
       SourceMassPairDistribution
    primary::P
    βq::T
    m2_low::T
    δm2::T
end

function DefaultBBHMassPair(;
        α1::Real,
        α2::Real,
        m_break::Real,
        μ1::Real,
        σ1::Real,
        μ2::Real,
        σ2::Real,
        m1_low::Real,
        δm1::Real,
        λ0::Real,
        λ1::Real,
        βq::Real,
        m2_low::Real,
        δm2::Real,
        m_high::Real = DEFAULT_BBH_M_HIGH
)
    primary = DefaultBBHPrimaryMass(;
        α1, α2, m_break, μ1, σ1, μ2, σ2, m1_low, δm1, λ0, λ1, m_high)
    T = promote_type(
        Float64, typeof(primary.m1_low), typeof(βq), typeof(m2_low), typeof(δm2))
    d = DefaultBBHMassPair(primary, T(βq), T(m2_low), T(δm2))
    _validate_mass_pair(d)
    return d
end

function _validate_mass_pair(d::DefaultBBHMassPair)
    d.m2_low > 0 || throw(ArgumentError("m2_low must be positive"))
    d.m2_low <= d.primary.m1_low ||
        throw(ArgumentError("m2_low must be less than or equal to m1_low"))
    d.δm2 >= 0 || throw(ArgumentError("δm2 must be non-negative"))
    return nothing
end

Base.length(::DefaultBBHMassPair) = 2
Base.size(::DefaultBBHMassPair) = (2,)
Base.eltype(::Type{<:DefaultBBHMassPair{P, T}}) where {P, T} = promote_type(eltype(P), T)
Base.eltype(d::DefaultBBHMassPair) = eltype(typeof(d))

function Distributions.insupport(d::DefaultBBHMassPair, value::Tuple{<:Real, <:Real})
    m1, m2 = value
    return insupport(d.primary, m1) && d.m2_low <= m2 <= m1
end

function Distributions.insupport(d::DefaultBBHMassPair, value::AbstractVector{<:Real})
    length(value) == 2 || return false
    return insupport(d, (value[1], value[2]))
end

function _q_power_integral(d::DefaultBBHMassPair, q_low::Real, q_high::Real)
    return _power_integral(q_low, q_high, d.βq)
end

@inline _gauss_rule(::Val{16}) = _PLANCK_Q_GAUSS_16
@inline _gauss_rule(::Val{N}) where {N} = QuadGK.gauss(Float64, N)

function _q_planck_taper_band_integral(
        q_low::Real,
        δq::Real,
        βq::Real,
        h::Real,
        order::Val{N} = Val(16)
) where {N}
    T = promote_type(typeof(q_low), typeof(δq), typeof(βq), typeof(h))
    h > 0 || return zero(T)
    x, w = _gauss_rule(order)
    scale = h / 2
    acc = zero(T)
    @inbounds for i in eachindex(x, w)
        t = scale * (x[i] + 1)
        acc += w[i] * (q_low + δq * t)^βq * _planck_unit_taper(t)
    end
    return δq * scale * acc
end

function _q_normalizer(d::DefaultBBHMassPair, m1::Real)
    q_low = d.m2_low / m1
    q_low < 1 || return zero(promote_type(typeof(q_low), typeof(d.βq)))
    d.δm2 == 0 && return _q_power_integral(d, q_low, one(q_low))

    # In q-space the taper has edge `q_low` and width `δq`, since
    # P(q·m₁; m2_low, δm2) = P((q - q_low) / δq). The conditional normalizer is the
    # fixed-rule tapered band integral plus the closed-form flat region above it.
    δq = d.δm2 / m1
    q_taper_high = min(q_low + δq, one(q_low))
    z = zero(promote_type(typeof(q_low), typeof(d.βq), typeof(δq)))
    if q_taper_high > q_low
        h = (q_taper_high - q_low) / δq
        z += _q_planck_taper_band_integral(q_low, δq, d.βq, h)
    end
    if q_taper_high < 1
        z += _q_power_integral(d, q_taper_high, one(q_low))
    end
    return z
end

function Distributions.logpdf(d::DefaultBBHMassPair, value::Tuple{<:Real, <:Real})
    insupport(d, value) || return -Inf
    m1, m2 = value
    q = m2 / m1
    log_s2 = _log_planck_taper(m2, d.m2_low, d.δm2)
    log_s2 == -Inf && return -Inf
    zq = _q_normalizer(d, m1)
    zq > 0 || return -Inf
    return logpdf(d.primary, m1) + d.βq * log(q) + log_s2 - log(zq) - log(m1)
end

function Distributions._logpdf(d::DefaultBBHMassPair, x::AbstractVector{<:Real})
    return logpdf(d, (x[1], x[2]))
end

function _rand_scaled_power(rng::AbstractRNG, low::Real, high::Real, exponent::Real)
    u = rand(rng)
    a = exponent + 1
    if abs(a) <= _SQRT_EPS_FLOAT64
        return low * exp(u * log(high / low))
    end
    return (low^a + u * (high^a - low^a))^(1 / a)
end

function _rand_primary_proposal(rng::AbstractRNG, d::DefaultBBHPrimaryMass)
    u = rand(rng) * (d.λ0 + d.λ1 + d.λ2)
    u < d.λ0 && return rand(rng, d.broken)
    u < d.λ0 + d.λ1 && return rand(rng, d.peak1)
    return rand(rng, d.peak2)
end

function Random.rand(rng::AbstractRNG, d::DefaultBBHPrimaryMass)
    while true
        m1 = _rand_primary_proposal(rng, d)
        m1 < d.m_high || continue
        rand(rng) <= planck_taper(m1, d.m1_low, d.δm1) && return m1
    end
end

function _rand_q_power(rng::AbstractRNG, βq::Real, q_low::Real)
    q_low < 1 || return one(q_low)
    u = rand(rng)
    a = βq + 1
    if abs(a) <= _SQRT_EPS_FLOAT64
        return q_low * exp(u * log(1 / q_low))
    end
    return (q_low^a + u * (1 - q_low^a))^(1 / a)
end

function _rand_q(rng::AbstractRNG, d::DefaultBBHMassPair, m1::Real)
    q_low = d.m2_low / m1
    while true
        q = _rand_q_power(rng, d.βq, q_low)
        rand(rng) <= planck_taper(q * m1, d.m2_low, d.δm2) && return q
    end
end

function Random.rand(rng::AbstractRNG, d::DefaultBBHMassPair)
    m1 = rand(rng, d.primary)
    q = _rand_q(rng, d, m1)
    return [m1, q * m1]
end

function Distributions._rand!(
        rng::AbstractRNG,
        d::DefaultBBHMassPair,
        x::AbstractVector{<:Real}
)
    length(x) == 2 || throw(ArgumentError("DEFAULT BBH mass pair expects length-2 output"))
    sample = rand(rng, d)
    x[1] = sample[1]
    x[2] = sample[2]
    return x
end
