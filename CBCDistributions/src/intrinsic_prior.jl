using Distributions
using Distributions: ProductNamedTupleDistribution

export redshift_logpdf_eltype

"""
    intrinsic_prior(::FullBNS; kwargs...) -> ProductNamedTupleDistribution

Build the intrinsic-parameter prior for the full-BNS proposal as a native
[`Distributions.product_distribution`](@ref) keyed by a `NamedTuple` with fields
`mass` (an [`OrderedUniformSourceMassPair`](@ref), a 2-vector component),
`χ₁`/`χ₂` (a shared [`AlignedSpinChiSimple`](@ref)), and `Λ₁`/`Λ₂`
(a shared [`Distributions.Uniform`](@ref)).

The returned [`ProductNamedTupleDistribution`](@ref) supports `rand`/`rand(prior, n)`
and `logpdf(prior, sample)` directly; use [`intrinsic_log_prob_samples`](@ref) for the
batched, allocation-light path on a [`FullBNSSamplesSoA`](@ref) sample container.
"""
function intrinsic_prior(
        ::FullBNS;
        mass_low::Real = BNS_MASS_LOW,
        mass_high::Real = BNS_MASS_HIGH,
        spin_a_max::Real = BNS_SPIN_A_MAX,
        lambda_high::Real = BNS_LAMBDA_HIGH
)
    lambda_dist = Uniform(0.0, Float64(lambda_high))
    spin_dist = AlignedSpinChiSimple(; a_max = spin_a_max)
    return product_distribution((
        mass = OrderedUniformSourceMassPair(; low = mass_low, high = mass_high),
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

function _full_bns_pointwise_logpdf(
        prior::ProductNamedTupleDistribution,
        samples::NamedTuple,
        i::Integer
)
    return (
        logpdf(prior.dists.mass, (samples.mass[1, i], samples.mass[2, i])) +
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

# --- Cached full-BNS intrinsic log-probability terms ---

"""
    fixed_intrinsic_log_prob(::FullBNS, samples; kwargs...) -> Vector{Float64}

Per-sample sum of mass, aligned-spin, and tidal-uniform log-pdfs for full-BNS
proposal samples. Matches the corresponding terms in [`intrinsic_prior`](@ref)
(`mass`, `χ₁`, `χ₂`, `Λ₁`, `Λ₂`); redshift is excluded because it depends on
the live [`RedshiftPrior`](@ref).
"""
function fixed_intrinsic_log_prob(
        ::FullBNS,
        samples::FullBNSSamplesSoA;
        mass_low::Real = BNS_MASS_LOW,
        mass_high::Real = BNS_MASS_HIGH,
        spin_a_max::Real = BNS_SPIN_A_MAX,
        lambda_high::Real = BNS_LAMBDA_HIGH
)
    n = _require_full_bns_soa_matching_lengths(samples)
    mass_dist = OrderedUniformSourceMassPair(; low = mass_low, high = mass_high)
    spin_dist = AlignedSpinChiSimple(; a_max = spin_a_max)
    lambda_dist = Uniform(0.0, Float64(lambda_high))
    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        out[i] = logpdf(mass_dist, (samples.mass[1, i], samples.mass[2, i])) +
                 logpdf(spin_dist, samples.χ₁[i]) +
                 logpdf(spin_dist, samples.χ₂[i]) +
                 logpdf(lambda_dist, samples.Λ₁[i]) +
                 logpdf(lambda_dist, samples.Λ₂[i])
    end
    return out
end

@inline function _redshift_logpdf(prior::RedshiftPrior, z::Real)
    x_lo = first(prior.dN_dz.x)
    x_hi = last(prior.dN_dz.x)
    (z < x_lo || z > x_hi) && return -Inf
    return redshift_log_prob(prior, z)
end

"""
    redshift_logpdf_eltype(prior::RedshiftPrior) -> Type

Element type of values returned by the redshift log-density associated with
`prior`. Useful for preallocating output vectors that promote with the
redshift contribution (for example `ForwardDiff.Dual` when `prior` was built
under AD).
"""
function redshift_logpdf_eltype(prior::RedshiftPrior)
    return promote_type(eltype(prior.dN_dz.y), typeof(redshift_integral(prior)))
end

"""
    intrinsic_log_prob_samples!(out, fixed_log_prob, prior, samples) -> out

Fill `out` with per-sample intrinsic log-prior using precomputed fixed full-BNS
terms and the live redshift density from [`RedshiftPrior`](@ref).
"""
function intrinsic_log_prob_samples!(
        out::AbstractVector,
        fixed_log_prob::AbstractVector{<:Real},
        prior::RedshiftPrior,
        samples::NamedTuple
)
    n = _require_full_bns_soa_matching_lengths(samples)
    length(out) == n ||
        throw(ArgumentError("output length must match the number of samples"))
    length(fixed_log_prob) == n ||
        throw(ArgumentError("fixed log-probability length must match the number of samples"))
    @inbounds for i in 1:n
        out[i] = fixed_log_prob[i] + _redshift_logpdf(prior, samples.redshift[i])
    end
    return out
end

"""
    intrinsic_log_prob_samples(fixed_log_prob, prior, samples) -> Vector

Allocating variant of [`intrinsic_log_prob_samples!`](@ref) for an
already-computed fixed intrinsic log-probability vector. Element type promotes with the redshift contribution
(e.g. `ForwardDiff.Dual` when `prior` was built under AD).
"""
function intrinsic_log_prob_samples(
        fixed_log_prob::AbstractVector{<:Real},
        prior::RedshiftPrior,
        samples::NamedTuple
)
    n = _require_full_bns_soa_matching_lengths(samples)
    length(fixed_log_prob) == n ||
        throw(ArgumentError("fixed log-probability length must match the number of samples"))
    if n == 0
        return Vector{promote_type(eltype(fixed_log_prob), redshift_logpdf_eltype(prior))}()
    end
    first_val = fixed_log_prob[1] + _redshift_logpdf(prior, samples.redshift[1])
    out = Vector{typeof(first_val)}(undef, n)
    intrinsic_log_prob_samples!(out, fixed_log_prob, prior, samples)
    return out
end
