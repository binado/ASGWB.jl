using LinearAlgebra

# Per-sample importance weight as a product of three physically independent factors:
# the population prior ratio `exp(log_ratio)`, the FLRW background distance ratio
# `(D_L,fid/D_L,θ)²`, and the modified-propagation factor `(1/Ξ_θ)²`. The raw catalog flux
# carries `1/D_L,fid²`, so multiplying by `dl_fid_sq / (D_L,θ² · Ξ_θ²)` recovers the
# physically correct `1/D_gw,θ²` dilution.
@inline function _importance_weight_at_sample(
        target_log_prob::AbstractVector,
        proposal_log_prob::AbstractVector{<:Real},
        dl_fid_sq::AbstractVector{<:Real},
        z::AbstractVector{<:Real},
        redshift_grid::AbstractVector{<:Real},
        interp::SampleInterpolant,
        cosmology_cache::CosmologyCache,
        sample_index::Integer
)
    log_ratio = target_log_prob[sample_index] - proposal_log_prob[sample_index]
    d_l = luminosity_distance_at_sample(
        cosmology_cache, interp, redshift_grid, z, sample_index)
    Ξ_theta = gw_em_distance_ratio(z[sample_index], cosmology_cache.cosmology)
    return exp(log_ratio) * dl_fid_sq[sample_index] / (d_l^2 * Ξ_theta^2)
end

# Shared kernel for both the naive and cached `compute_importance_weights` methods. Inputs
# are explicit arrays so the only difference between the two backends is where the fiducial
# caches come from (recomputed vs read from a `ModelContext`). Built with `map` over the
# index range: this keeps the result type stable (a properly-typed empty vector for n == 0,
# rather than a `Union` with `Float64[]`) so the AD/`Dual` likelihood path stays inferrable.
function _importance_weights_core(
        target_log_prob::AbstractVector,
        proposal_log_prob::AbstractVector{<:Real},
        dl_fid_sq::AbstractVector{<:Real},
        z::AbstractVector{<:Real},
        redshift_grid::AbstractVector{<:Real},
        interp::SampleInterpolant,
        cosmology_cache::CosmologyCache
)
    length(z) == length(target_log_prob) ||
        throw(ArgumentError("population prior logpdf length must match proposal sample count"))
    return map(eachindex(z)) do i
        _importance_weight_at_sample(
            target_log_prob, proposal_log_prob, dl_fid_sq, z, redshift_grid, interp,
            cosmology_cache, i)
    end
end

"""
    compute_importance_weights(problem, C, Λ, ctx::ModelContext) -> Vector

Per-sample importance weights at hyperparameters `Λ` (cosmology family `C`), reading the
fiducial proposal caches from `ctx`. This is the hot path used by the likelihood and the
generative model.
"""
function compute_importance_weights(
        problem::ImportanceSamplingProblem,
        ::Type{C},
        Λ::NamedTuple,
        ctx::ModelContext
) where {C <: AbstractCosmology}
    cache = CosmologyCache(cosmology(C, Λ), ctx.redshift_grid)
    prior = single_event_prior(problem.population_model, cache, Λ)
    return compute_importance_weights(problem, cache, prior, ctx)
end

"""
    compute_importance_weights(problem, cache::CosmologyCache, prior, ctx::ModelContext) -> Vector

Importance weights from a prebuilt `cache` and single-event `prior`. The forward model
builds the cache once, shares it with `single_event_prior` (which uses it for the redshift
prior), and passes it here for the per-sample luminosity distance — so the cumulative
cosmology integral is computed once per evaluation rather than rebuilt in both places. This
is the bare form the forward model calls directly.
"""
function compute_importance_weights(
        problem::ImportanceSamplingProblem,
        cache::CosmologyCache,
        prior,
        ctx::ModelContext
)
    # Reuse the precomputed per-sample grid locations for the redshift component so
    # its batched logpdf skips the per-sample grid search every gradient evaluation.
    target_log_prob = batched_logpdf(
        prior, problem.samples, (; redshift = ctx.sample_interpolant))
    return _importance_weights_core(
        target_log_prob,
        ctx.proposal_log_prob,
        ctx.dl_fid_sq,
        problem.samples.redshift,
        ctx.redshift_grid,
        ctx.sample_interpolant,
        cache
    )
end

"""
    compute_importance_weights(problem, C, Λ) -> Vector

Naive importance weights: recompute the fiducial proposal caches (`proposal_log_prob`,
`dl_fid_sq`, redshift interpolant) from scratch at `problem.fiducial_hyperparameters`,
then weight at `Λ`. Slower than the `ctx` method but free of any precomputed state, so it
serves as the correctness oracle for the cached path.
"""
function compute_importance_weights(
        problem::ImportanceSamplingProblem,
        ::Type{C},
        Λ::NamedTuple
) where {C <: AbstractCosmology}
    Λ_fid = problem.fiducial_hyperparameters
    z = problem.samples.redshift
    proposal_log_prob = _reconstruct_proposal_log_prob(
        problem.samples, C, problem.population_model, Λ_fid)
    dl_fid_sq = _reconstruct_dl_fid_sq(z, C, Λ_fid)
    redshift_grid = collect(Float64, DEFAULT_Z_GRID)
    interp = SampleInterpolant(z, redshift_grid)

    cosmology_cache = CosmologyCache(cosmology(C, Λ), redshift_grid)
    prior = single_event_prior(problem.population_model, cosmology_cache, Λ)
    target_log_prob = batched_logpdf(prior, problem.samples)
    return _importance_weights_core(
        target_log_prob,
        proposal_log_prob,
        dl_fid_sq,
        z,
        redshift_grid,
        interp,
        cosmology_cache
    )
end
