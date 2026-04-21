using Distributions
using Random

const BNS_MASS_LOW = 1.1
const BNS_MASS_HIGH = 2.5
const BNS_LAMBDA_HIGH = 5000.0
const BNS_SPIN_A_MAX = 0.99

struct OrderedUniformSourceMassPair{T<:Real} <: ContinuousMultivariateDistribution
    low::T
    high::T
end

function OrderedUniformSourceMassPair(; low::Real=BNS_MASS_LOW, high::Real=BNS_MASS_HIGH)
    low < high || throw(ArgumentError("low must be smaller than high"))
    return OrderedUniformSourceMassPair(Float64(low), Float64(high))
end

Base.length(::OrderedUniformSourceMassPair) = 2
Base.size(::OrderedUniformSourceMassPair) = (2,)

function Distributions.insupport(
    d::OrderedUniformSourceMassPair,
    value::NTuple{2,<:Real},
)
    m1, m2 = value
    return m1 >= m2 && m2 >= d.low && m1 <= d.high
end

function Distributions.insupport(
    d::OrderedUniformSourceMassPair,
    value::AbstractVector{<:Real},
)
    length(value) == 2 || return false
    return insupport(d, (value[1], value[2]))
end

function Distributions.logpdf(
    d::OrderedUniformSourceMassPair,
    value::NTuple{2,<:Real},
)
    return insupport(d, value) ? log(2.0) - 2.0 * log(d.high - d.low) : -Inf
end

function Distributions.logpdf(
    d::OrderedUniformSourceMassPair,
    value::AbstractVector{<:Real},
)
    length(value) == 2 || throw(ArgumentError("ordered mass pair expects two coordinates"))
    return logpdf(d, (value[1], value[2]))
end

function Random.rand(rng::AbstractRNG, d::OrderedUniformSourceMassPair)
    span = d.high - d.low
    x = d.low + span * rand(rng)
    y = d.low + span * rand(rng)
    return x >= y ? [x, y] : [y, x]
end

struct AlignedSpinChiSimple{T<:Real} <: ContinuousUnivariateDistribution
    a_max::T
end

function AlignedSpinChiSimple(; a_max::Real=BNS_SPIN_A_MAX)
    a_max > 0 || throw(ArgumentError("a_max must be positive"))
    return AlignedSpinChiSimple(Float64(a_max))
end

Base.minimum(d::AlignedSpinChiSimple) = -d.a_max
Base.maximum(d::AlignedSpinChiSimple) = d.a_max

Distributions.insupport(d::AlignedSpinChiSimple, value::Real) = abs(value) <= d.a_max

function Distributions.logpdf(d::AlignedSpinChiSimple, value::Real)
    insupport(d, value) || return -Inf
    eps_value = eps(Float64)
    density = -log(max(abs(value), eps_value) / d.a_max) / (2.0 * d.a_max)
    return log(max(density, floatmin(Float64)))
end

function Random.rand(rng::AbstractRNG, d::AlignedSpinChiSimple)
    magnitude = d.a_max * rand(rng) * rand(rng)
    return rand(rng, Bool) ? magnitude : -magnitude
end

struct RedshiftInterpolatedDistribution{B<:RadialInterpolant} <: ContinuousUnivariateDistribution
    bundle::B
end

Base.minimum(d::RedshiftInterpolatedDistribution) = first(d.bundle.x)
Base.maximum(d::RedshiftInterpolatedDistribution) = last(d.bundle.x)

function Distributions.insupport(d::RedshiftInterpolatedDistribution, value::Real)
    return minimum(d) <= value <= maximum(d)
end

Distributions.logpdf(d::RedshiftInterpolatedDistribution, value::Real) =
    log_prob_from_bundle(value, d.bundle)

function Random.rand(rng::AbstractRNG, d::RedshiftInterpolatedDistribution)
    target = rand(rng) * d.bundle.norm
    cumulative = d.bundle.cumulative
    x = d.bundle.x
    n = length(cumulative)
    idx = searchsortedlast(cumulative, target)
    idx <= 0 && return x[1]
    idx >= n && return x[end]
    c0, c1 = cumulative[idx], cumulative[idx+1]
    x0, x1 = x[idx], x[idx+1]
    c1 > c0 || return x0
    return x0 + (target - c0) * (x1 - x0) / (c1 - c0)
end

struct IntrinsicPriorTerm{F,D}
    name::Symbol
    fields::F
    dist::D
end

IntrinsicPriorTerm(name::Symbol, field::Symbol, dist) =
    IntrinsicPriorTerm(name, (field,), dist)

function IntrinsicPriorTerm(name::Symbol, fields::Tuple{Vararg{Symbol}}, dist)
    isempty(fields) && throw(ArgumentError("IntrinsicPriorTerm fields must be non-empty"))
    return IntrinsicPriorTerm{typeof(fields),typeof(dist)}(name, fields, dist)
end

function build_uniform_priors(
    bounds::AbstractDict{<:AbstractString,<:Tuple{<:Real,<:Real}},
)
    return InferencePriors(
        Uniform(Float64(bounds["H0"][1]), Float64(bounds["H0"][2])),
        Uniform(Float64(bounds["Omega_m"][1]), Float64(bounds["Omega_m"][2])),
        Uniform(Float64(bounds["chi0"][1]), Float64(bounds["chi0"][2])),
        Uniform(Float64(bounds["chin"][1]), Float64(bounds["chin"][2])),
        Uniform(Float64(bounds["gamma"][1]), Float64(bounds["gamma"][2])),
        Uniform(Float64(bounds["kappa"][1]), Float64(bounds["kappa"][2])),
        Uniform(Float64(bounds["z_peak"][1]), Float64(bounds["z_peak"][2])),
    )
end

function logprior(h::HyperParameters, priors::InferencePriors)
    pop = h.population
    pop isa MadauDickinsonParameters || throw(
        ArgumentError("logprior with InferencePriors requires MadauDickinsonParameters"),
    )
    return (
        logpdf(priors.H0, h.cosmological.H0) +
        logpdf(priors.Omega_m, h.cosmological.Omega_m) +
        logpdf(priors.chi0, h.propagation.chi0) +
        logpdf(priors.chin, h.propagation.chin) +
        logpdf(priors.gamma, pop.gamma) +
        logpdf(priors.kappa, pop.kappa) +
        logpdf(priors.z_peak, pop.z_peak)
    )
end

function intrinsic_prior_terms(
    ::FullBNS,
    bundle::RadialInterpolant;
    mass_low::Real=BNS_MASS_LOW,
    mass_high::Real=BNS_MASS_HIGH,
    spin_a_max::Real=BNS_SPIN_A_MAX,
    lambda_high::Real=BNS_LAMBDA_HIGH,
)
    return (
        IntrinsicPriorTerm(
            :masses,
            (:mass_1_source, :mass_2_source),
            OrderedUniformSourceMassPair(; low=mass_low, high=mass_high),
        ),
        IntrinsicPriorTerm(:redshift, :redshift, RedshiftInterpolatedDistribution(bundle)),
        IntrinsicPriorTerm(:chi_1, :chi_1, AlignedSpinChiSimple(; a_max=spin_a_max)),
        IntrinsicPriorTerm(:chi_2, :chi_2, AlignedSpinChiSimple(; a_max=spin_a_max)),
        IntrinsicPriorTerm(:lambda_1, :lambda_1, Uniform(0.0, Float64(lambda_high))),
        IntrinsicPriorTerm(:lambda_2, :lambda_2, Uniform(0.0, Float64(lambda_high))),
    )
end

function _require_matching_output_length(
    out::AbstractVector,
    values::AbstractVector{<:Real}...,
)
    n = length(out)
    all(length(v) == n for v in values) || throw(
        ArgumentError(
            "output and intrinsic prior sample vectors must have matching lengths",
        ),
    )
    return n
end

function _term_result_type(
    samples,
    term::IntrinsicPriorTerm{<:Tuple{Symbol}},
)
    values = getfield(samples, term.fields[1])
    return typeof(logpdf(term.dist, first(values)))
end

function _term_result_type(
    samples,
    term::IntrinsicPriorTerm{<:Tuple{Symbol,Symbol}},
)
    values1 = getfield(samples, term.fields[1])
    values2 = getfield(samples, term.fields[2])
    return typeof(logpdf(term.dist, (first(values1), first(values2))))
end

_intrinsic_log_prob_type(samples, ::Tuple{}) = Float64

function _intrinsic_log_prob_type(samples, terms::Tuple)
    return promote_type(
        _term_result_type(samples, first(terms)),
        _intrinsic_log_prob_type(samples, Base.tail(terms)),
    )
end

function _accumulate_logpdf!(
    out::AbstractVector,
    samples,
    term::IntrinsicPriorTerm{<:Tuple{Symbol}},
)
    values = getfield(samples, term.fields[1])
    _require_matching_output_length(out, values)
    @inbounds for i in eachindex(out, values)
        out[i] += logpdf(term.dist, values[i])
    end
    return out
end

function _accumulate_logpdf!(
    out::AbstractVector,
    samples,
    term::IntrinsicPriorTerm{<:Tuple{Symbol,Symbol}},
)
    values1 = getfield(samples, term.fields[1])
    values2 = getfield(samples, term.fields[2])
    _require_matching_output_length(out, values1, values2)
    @inbounds for i in eachindex(out, values1, values2)
        out[i] += logpdf(term.dist, (values1[i], values2[i]))
    end
    return out
end

function _accumulate_logpdf!(out::AbstractVector, samples, term::IntrinsicPriorTerm)
    throw(
        ArgumentError(
            "unsupported intrinsic prior term arity $(length(term.fields)) for $(term.name)",
        ),
    )
end

_intrinsic_log_prob_samples!(out::AbstractVector, samples, ::Tuple{}) = out

function _intrinsic_log_prob_samples!(out::AbstractVector, samples, terms::Tuple)
    _accumulate_logpdf!(out, samples, first(terms))
    return _intrinsic_log_prob_samples!(out, samples, Base.tail(terms))
end

function intrinsic_log_prob_samples!(
    out::AbstractVector,
    samples,
    terms::Tuple,
)
    fill!(out, zero(eltype(out)))
    return _intrinsic_log_prob_samples!(out, samples, terms)
end

function intrinsic_log_prob_samples(samples, terms::Tuple)
    isempty(terms) && return Float64[]
    first_term = first(terms)
    first_values = getfield(samples, first(first_term.fields))
    T = _intrinsic_log_prob_type(samples, terms)
    out = similar(first_values, T)
    return intrinsic_log_prob_samples!(out, samples, terms)
end
