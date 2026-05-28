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

function _redshift_prior_distribution(prior)
    return prior.dists.redshift
end

function _redshift_integral_from_population(prior)
    return redshift_integral(_redshift_prior_distribution(prior).prior)
end

function _dgw_from_cached_dl(z, d_l, c::AbstractCosmology)
    return d_l
end

function _dgw_from_cached_dl(z, d_l, c::ModifiedPropagation)
    return gravitational_wave_distance(z, d_l, c.Ξ₀, c.Ξₙ)
end

@inline function _importance_terms_at_sample(
        problem::ImportanceSamplingProblem,
        cosmology_cache::CosmologyCache,
        target_log_prob::AbstractVector,
        z::AbstractVector{<:Real},
        interp::SampleInterpolant,
        sample_index::Integer
)
    log_ratio = target_log_prob[sample_index] - problem.proposal.log_prob[sample_index]
    d_l = luminosity_distance_at_sample(
        cosmology_cache,
        interp,
        problem.redshift_grid,
        z,
        sample_index
    )
    dgw_theta = _dgw_from_cached_dl(z[sample_index], d_l, cosmology_cache.cosmology)
    dgw_theta_sq = dgw_theta^2
    weight = exp(log_ratio) * problem.proposal.dgw_fid_sq[sample_index] / dgw_theta_sq
    return log_ratio, dgw_theta_sq, weight
end

function compute_importance_weights(
        problem::ImportanceSamplingProblem,
        Λ::NamedTuple,
        cosmology_cache::CosmologyCache,
        prior
)
    z = redshift(problem)
    n = length(z)
    target_log_prob = batched_logpdf(prior, problem.proposal.samples)
    n == length(target_log_prob) ||
        throw(ArgumentError("population prior logpdf length must match proposal sample count"))
    if n == 0
        return (;
            weights = Float64[],
            log_ratio = Float64[],
            target_log_prob = target_log_prob,
            dgw_theta_sq = Float64[]
        )
    end

    interp = problem.redshift_cache.sample_interpolant
    first_ratio, first_dgw_sq,
    first_weight = _importance_terms_at_sample(
        problem, cosmology_cache, target_log_prob, z, interp, 1)
    log_ratio = Vector{typeof(first_ratio)}(undef, n)
    dgw_theta_sq = Vector{typeof(first_dgw_sq)}(undef, n)
    weights = Vector{typeof(first_weight)}(undef, n)

    @inbounds begin
        log_ratio[1] = first_ratio
        dgw_theta_sq[1] = first_dgw_sq
        weights[1] = first_weight
        for i in 2:n
            ratio, dgw_sq,
            weight = _importance_terms_at_sample(
                problem, cosmology_cache, target_log_prob, z, interp, i)
            log_ratio[i] = ratio
            dgw_theta_sq[i] = dgw_sq
            weights[i] = weight
        end
    end
    return (;
        weights = weights,
        log_ratio = log_ratio,
        target_log_prob = target_log_prob,
        dgw_theta_sq = dgw_theta_sq
    )
end

function compute_importance_weights(problem::ImportanceSamplingProblem, Λ::NamedTuple)
    c = cosmology(problem.model, Λ)
    cache = CosmologyCache(c, problem.redshift_grid)
    prior = population_prior(problem.model, Λ; z_grid = problem.redshift_grid)
    return compute_importance_weights(problem, Λ, cache, prior)
end
