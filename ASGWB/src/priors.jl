using Distributions
using Distributions: ProductNamedTupleDistribution
using Random

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
the live [`RedshiftBundle`](@ref).
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

@inline function _redshift_logpdf(bundle::RedshiftBundle, z::Real)
    x_lo = first(bundle.pdf.x)
    x_hi = last(bundle.pdf.x)
    (z < x_lo || z > x_hi) && return -Inf
    return log_prob_from_bundle(z, bundle)
end

function _redshift_logpdf_type(bundle::RedshiftBundle)
    return promote_type(eltype(bundle.pdf.y), typeof(redshift_integral(bundle)))
end

"""
    intrinsic_log_prob_samples!(out, fixed_log_prob, bundle, samples) -> out

Fill `out` with per-sample intrinsic log-prior using precomputed fixed full-BNS
terms and the live redshift density from [`RedshiftBundle`](@ref).
"""
function intrinsic_log_prob_samples!(
        out::AbstractVector,
        fixed_log_prob::AbstractVector{<:Real},
        bundle::RedshiftBundle,
        samples::NamedTuple
)
    n = _require_full_bns_soa_matching_lengths(samples)
    length(out) == n ||
        throw(ArgumentError("output length must match the number of samples"))
    length(fixed_log_prob) == n ||
        throw(ArgumentError("fixed log-probability length must match the number of samples"))
    @inbounds for i in 1:n
        out[i] = fixed_log_prob[i] + _redshift_logpdf(bundle, samples.redshift[i])
    end
    return out
end

"""
    intrinsic_log_prob_samples(fixed_log_prob, bundle, samples) -> Vector

Allocating variant of [`intrinsic_log_prob_samples!`](@ref) for an
already-computed fixed intrinsic log-probability vector. Element type promotes with the redshift contribution
(e.g. `ForwardDiff.Dual` when `bundle` was built under AD).
"""
function intrinsic_log_prob_samples(
        fixed_log_prob::AbstractVector{<:Real},
        bundle::RedshiftBundle,
        samples::NamedTuple
)
    n = _require_full_bns_soa_matching_lengths(samples)
    length(fixed_log_prob) == n ||
        throw(ArgumentError("fixed log-probability length must match the number of samples"))
    if n == 0
        return Vector{promote_type(eltype(fixed_log_prob), _redshift_logpdf_type(bundle))}()
    end
    first_val = fixed_log_prob[1] + _redshift_logpdf(bundle, samples.redshift[1])
    out = Vector{typeof(first_val)}(undef, n)
    intrinsic_log_prob_samples!(out, fixed_log_prob, bundle, samples)
    return out
end
