using LinearAlgebra

function importance_weights(
        log_ratio::AbstractVector{<:Real},
        dgw_fid_sq::AbstractVector{<:Real},
        dgw_theta_sq::AbstractVector{<:Real}
)
    length(log_ratio) == length(dgw_fid_sq) == length(dgw_theta_sq) ||
        throw(ArgumentError("importance weight inputs must have matching lengths"))
    return exp.(log_ratio) .* dgw_fid_sq ./ dgw_theta_sq
end

@inline function _importance_weight_at_sample(
        target_log_prob::AbstractVector,
        proposal_log_prob::AbstractVector{<:Real},
        dgw_fid_sq::AbstractVector{<:Real},
        z::AbstractVector{<:Real},
        redshift_grid::AbstractVector{<:Real},
        interp::SampleInterpolant,
        cosmology_cache::CosmologyCache,
        sample_index::Integer
)
    log_ratio = target_log_prob[sample_index] - proposal_log_prob[sample_index]
    d_l = luminosity_distance_at_sample(
        cosmology_cache, interp, redshift_grid, z, sample_index)
    dgw_theta = gravitational_wave_distance(z[sample_index], d_l, cosmology_cache.cosmology)
    return exp(log_ratio) * dgw_fid_sq[sample_index] / dgw_theta^2
end

# Shared kernel for both the naive and cached `compute_importance_weights` methods. Inputs
# are explicit arrays so the only difference between the two backends is where the fiducial
# caches come from (recomputed vs read from a `ModelContext`). Built with `map` over the
# index range: this keeps the result type stable (a properly-typed empty vector for n == 0,
# rather than a `Union` with `Float64[]`) so the AD/`Dual` likelihood path stays inferrable.
function _importance_weights_core(
        target_log_prob::AbstractVector,
        proposal_log_prob::AbstractVector{<:Real},
        dgw_fid_sq::AbstractVector{<:Real},
        z::AbstractVector{<:Real},
        redshift_grid::AbstractVector{<:Real},
        interp::SampleInterpolant,
        cosmology_cache::CosmologyCache
)
    length(z) == length(target_log_prob) ||
        throw(ArgumentError("population prior logpdf length must match proposal sample count"))
    return map(eachindex(z)) do i
        _importance_weight_at_sample(
            target_log_prob, proposal_log_prob, dgw_fid_sq, z, redshift_grid, interp,
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
    c = cosmology(C, Λ)
    prior = single_event_prior(problem.population_model, c, Λ)
    return _importance_weights(problem, c, prior, ctx)
end

# Importance weights from an already-built cosmology `c` and single-event `prior`. Callers
# that also need `prior` (e.g. the likelihood, which derives the merger rate from it) build
# the prior — and its embedded redshift CosmologyCache — only once per evaluation instead of
# once per atomic call.
function _importance_weights(
        problem::ImportanceSamplingProblem,
        c::AbstractCosmology,
        prior,
        ctx::ModelContext
)
    cosmology_cache = CosmologyCache(c, ctx.redshift_grid)
    target_log_prob = batched_logpdf(prior, problem.samples)
    return _importance_weights_core(
        target_log_prob,
        ctx.proposal_log_prob,
        ctx.dgw_fid_sq,
        problem.samples.redshift,
        ctx.redshift_grid,
        ctx.sample_interpolant,
        cosmology_cache
    )
end

"""
    compute_importance_weights(problem, C, Λ) -> Vector

Naive importance weights: recompute the fiducial proposal caches (`proposal_log_prob`,
`dgw_fid_sq`, redshift interpolant) from scratch at `problem.fiducial_hyperparameters`,
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
    dgw_fid_sq = _reconstruct_dgw_fid_sq(z, C, Λ_fid)
    redshift_grid = collect(Float64, DEFAULT_Z_GRID)
    interp = SampleInterpolant(z, redshift_grid)

    c = cosmology(C, Λ)
    cosmology_cache = CosmologyCache(c, redshift_grid)
    prior = single_event_prior(problem.population_model, c, Λ)
    target_log_prob = batched_logpdf(prior, problem.samples)
    return _importance_weights_core(
        target_log_prob,
        proposal_log_prob,
        dgw_fid_sq,
        z,
        redshift_grid,
        interp,
        cosmology_cache
    )
end
