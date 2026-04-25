using Distributions
using Distributions: ProductNamedTupleDistribution
using Random

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
    return OrderedUniformSourceMassPair(Float64(low), Float64(high))
end

Base.length(::OrderedUniformSourceMassPair) = 2
Base.size(::OrderedUniformSourceMassPair) = (2,)
Base.eltype(::Type{<:OrderedUniformSourceMassPair}) = Float64
Base.eltype(::OrderedUniformSourceMassPair) = Float64

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
    return insupport(d, value) ? log(2.0) - 2.0 * log(d.high - d.low) : -Inf
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
    span = d.high - d.low
    x = d.low + span * rand(rng)
    y = d.low + span * rand(rng)
    return x >= y ? [x, y] : [y, x]
end

function Distributions._rand!(
        rng::AbstractRNG,
        d::OrderedUniformSourceMassPair,
        x::AbstractVector{<:Real}
)
    length(x) == 2 || throw(ArgumentError("ordered mass pair expects length-2 output"))
    span = d.high - d.low
    a = d.low + span * rand(rng)
    b = d.low + span * rand(rng)
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

struct RedshiftInterpolatedDistribution{B <: RedshiftBundle} <:
       ContinuousUnivariateDistribution
    bundle::B
end

Base.minimum(d::RedshiftInterpolatedDistribution) = first(d.bundle.pdf.x)
Base.maximum(d::RedshiftInterpolatedDistribution) = last(d.bundle.pdf.x)

function Distributions.insupport(d::RedshiftInterpolatedDistribution, value::Real)
    return minimum(d) <= value <= maximum(d)
end

function Distributions.logpdf(d::RedshiftInterpolatedDistribution, value::Real)
    insupport(d, value) || return -Inf
    return log_prob_from_bundle(value, d.bundle)
end

function Random.rand(rng::AbstractRNG, d::RedshiftInterpolatedDistribution)
    target = rand(rng) * redshift_integral(d.bundle)
    cumulative = d.bundle.pdf.cumulative
    x = d.bundle.pdf.x
    n = length(cumulative)
    idx = searchsortedlast(cumulative, target)
    idx <= 0 && return x[1]
    idx >= n && return x[end]
    c0, c1 = cumulative[idx], cumulative[idx + 1]
    x0, x1 = x[idx], x[idx + 1]
    c1 > c0 || return x0
    return x0 + (target - c0) * (x1 - x0) / (c1 - c0)
end

"""
    build_uniform_priors(bounds) -> ProductNamedTupleDistribution

Build the seven-parameter uniform hyperparameter prior as a native
[`Distributions.product_distribution`](@ref) keyed by [`DEFAULT_PARAMETER_ORDER`](@ref).
`bounds` is a dict keyed by parameter name (`"H0"`, `"Omega_m"`, `"chi0"`, `"chin"`,
`"gamma"`, `"kappa"`, `"z_peak"`) carrying `(low, high)` tuples.
"""
function build_uniform_priors(bounds::AbstractDict{
        <:AbstractString, <:Tuple{<:Real, <:Real}})
    return product_distribution((
        H0 = Uniform(Float64(bounds["H0"][1]), Float64(bounds["H0"][2])),
        Ωm = Uniform(Float64(bounds["Omega_m"][1]), Float64(bounds["Omega_m"][2])),
        Ξ₀ = Uniform(Float64(bounds["chi0"][1]), Float64(bounds["chi0"][2])),
        Ξₙ = Uniform(Float64(bounds["chin"][1]), Float64(bounds["chin"][2])),
        γ = Uniform(Float64(bounds["gamma"][1]), Float64(bounds["gamma"][2])),
        κ = Uniform(Float64(bounds["kappa"][1]), Float64(bounds["kappa"][2])),
        zpeak = Uniform(Float64(bounds["z_peak"][1]), Float64(bounds["z_peak"][2]))
    ))
end

"""
    logprior(h::HyperParameters, prior::ProductNamedTupleDistribution) -> Real

Log-prior of `h` under the seven-parameter product distribution. `h` is a flat
`NamedTuple` matching `keys(prior.dists)` (`DEFAULT_PARAMETER_ORDER`).
"""
function logprior(h::HyperParametersNT, prior::ProductNamedTupleDistribution)
    return logpdf(prior, h)
end

"""
    intrinsic_prior(::FullBNS, bundle; kwargs...) -> ProductNamedTupleDistribution

Build the intrinsic-parameter prior for the full-BNS proposal as a native
[`Distributions.product_distribution`](@ref) keyed by a `NamedTuple` with fields
`mass` (an [`OrderedUniformSourceMassPair`](@ref), a 2-vector component),
`redshift` (a [`RedshiftInterpolatedDistribution`](@ref) tied to `bundle`),
`χ₁`/`χ₂` (a shared [`AlignedSpinChiSimple`](@ref)), and `Λ₁`/`Λ₂`
(a shared [`Distributions.Uniform`](@ref)).

The returned [`ProductNamedTupleDistribution`](@ref) supports `rand`/`rand(prior, n)`
and `logpdf(prior, sample)` directly; use [`intrinsic_log_prob_samples`](@ref) for the
batched, allocation-light path on a [`FullBNSSamplesSoA`](@ref) sample container.
"""
function intrinsic_prior(
        ::FullBNS,
        bundle::RedshiftBundle;
        mass_low::Real = BNS_MASS_LOW,
        mass_high::Real = BNS_MASS_HIGH,
        spin_a_max::Real = BNS_SPIN_A_MAX,
        lambda_high::Real = BNS_LAMBDA_HIGH
)
    lambda_dist = Uniform(0.0, Float64(lambda_high))
    spin_dist = AlignedSpinChiSimple(; a_max = spin_a_max)
    return product_distribution((
        mass = OrderedUniformSourceMassPair(; low = mass_low, high = mass_high),
        redshift = RedshiftInterpolatedDistribution(bundle),
        χ₁ = spin_dist,
        χ₂ = spin_dist,
        Λ₁ = lambda_dist,
        Λ₂ = lambda_dist
    ))
end

"""
    intrinsic_log_prob_samples(prior, samples) -> Vector{Float64}

Per-sample intrinsic log-prior. Two methods:

- `samples::AbstractVector{<:NamedTuple}` (AoS fallback) broadcasts `logpdf(prior, s)`
  over the collection.
- `prior::ProductNamedTupleDistribution` + `samples::FullBNSSamplesSoA` (SoA hot path)
  evaluates each component `logpdf` over contiguous arrays and sums, with no per-sample
  heap allocation.
"""
function intrinsic_log_prob_samples(prior, samples::AbstractVector{<:NamedTuple})
    logpdf.(Ref(prior), samples)
end

function _redshift_log_prob_samples(
        dist::RedshiftInterpolatedDistribution,
        values::AbstractVector{<:Real},
        sample_interpolant::SampleInterpolant
)
    return log_prob_from_bundle(values, dist.bundle, sample_interpolant)
end

function _full_bns_pointwise_logpdf(
        prior::ProductNamedTupleDistribution,
        samples::NamedTuple,
        i::Integer
)
    return (
        logpdf(prior.dists.mass, (samples.mass[1, i], samples.mass[2, i])) +
        logpdf(prior.dists.redshift, samples.redshift[i]) +
        logpdf(prior.dists.χ₁, samples.χ₁[i]) +
        logpdf(prior.dists.χ₂, samples.χ₂[i]) +
        logpdf(prior.dists.Λ₁, samples.Λ₁[i]) +
        logpdf(prior.dists.Λ₂, samples.Λ₂[i])
    )
end

function intrinsic_log_prob_samples(
        prior::ProductNamedTupleDistribution,
        samples::NamedTuple
)
    n = _require_full_bns_soa_matching_lengths(samples)
    n == 0 && return Float64[]
    first_val = _full_bns_pointwise_logpdf(prior, samples, 1)
    out = Vector{typeof(first_val)}(undef, n)
    @inbounds out[1] = first_val
    @inbounds for i in 2:n
        out[i] = _full_bns_pointwise_logpdf(prior, samples, i)
    end
    return out
end

function intrinsic_log_prob_samples(
        prior::ProductNamedTupleDistribution,
        samples::NamedTuple,
        sample_interpolant::SampleInterpolant
)
    n = _require_full_bns_soa_matching_lengths(samples)
    n == 0 && return Float64[]
    redshift_log_prob = _redshift_log_prob_samples(
        prior.dists.redshift,
        samples.redshift,
        sample_interpolant
    )
    first_val = (
        logpdf(prior.dists.mass, (samples.mass[1, 1], samples.mass[2, 1])) +
        redshift_log_prob[1] +
        logpdf(prior.dists.χ₁, samples.χ₁[1]) +
        logpdf(prior.dists.χ₂, samples.χ₂[1]) +
        logpdf(prior.dists.Λ₁, samples.Λ₁[1]) +
        logpdf(prior.dists.Λ₂, samples.Λ₂[1])
    )
    out = Vector{typeof(first_val)}(undef, n)
    @inbounds out[1] = first_val
    @inbounds for i in 2:n
        out[i] = (
            logpdf(prior.dists.mass, (samples.mass[1, i], samples.mass[2, i])) +
            redshift_log_prob[i] +
            logpdf(prior.dists.χ₁, samples.χ₁[i]) +
            logpdf(prior.dists.χ₂, samples.χ₂[i]) +
            logpdf(prior.dists.Λ₁, samples.Λ₁[i]) +
            logpdf(prior.dists.Λ₂, samples.Λ₂[i])
        )
    end
    return out
end

"""
    intrinsic_log_prob_samples!(out, prior, samples) -> out

In-place SoA variant of [`intrinsic_log_prob_samples`](@ref). Writes per-sample
log-prior into `out`; `out` must have length equal to `length(samples.redshift)`.
"""
function intrinsic_log_prob_samples!(
        out::AbstractVector,
        prior::ProductNamedTupleDistribution,
        samples::NamedTuple
)
    n = _require_full_bns_soa_matching_lengths(samples)
    length(out) == n ||
        throw(ArgumentError("output length must match the number of samples"))
    @inbounds for i in 1:n
        out[i] = _full_bns_pointwise_logpdf(prior, samples, i)
    end
    return out
end

function intrinsic_log_prob_samples!(
        out::AbstractVector,
        prior::ProductNamedTupleDistribution,
        samples::NamedTuple,
        sample_interpolant::SampleInterpolant
)
    n = _require_full_bns_soa_matching_lengths(samples)
    length(out) == n ||
        throw(ArgumentError("output length must match the number of samples"))
    redshift_log_prob = _redshift_log_prob_samples(
        prior.dists.redshift,
        samples.redshift,
        sample_interpolant
    )
    @inbounds for i in 1:n
        out[i] = (
            logpdf(prior.dists.mass, (samples.mass[1, i], samples.mass[2, i])) +
            redshift_log_prob[i] +
            logpdf(prior.dists.χ₁, samples.χ₁[i]) +
            logpdf(prior.dists.χ₂, samples.χ₂[i]) +
            logpdf(prior.dists.Λ₁, samples.Λ₁[i]) +
            logpdf(prior.dists.Λ₂, samples.Λ₂[i])
        )
    end
    return out
end

function _require_full_bns_soa_matching_lengths(samples::NamedTuple)
    n = length(samples.redshift)
    (
        length(samples.χ₁) == n &&
        length(samples.χ₂) == n &&
        length(samples.Λ₁) == n &&
        length(samples.Λ₂) == n &&
        size(samples.mass, 2) == n
    ) || throw(ArgumentError("SoA sample vectors must all have matching lengths"))
    size(samples.mass, 1) == 2 ||
        throw(ArgumentError("SoA mass matrix must have two rows (m1, m2)"))
    return n
end
