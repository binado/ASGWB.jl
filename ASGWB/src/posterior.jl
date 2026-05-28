function evaluate_model_terms(
        Λ::NamedTuple,
        problem::ImportanceSamplingProblem
)
    c = cosmology(problem.model, Λ)
    cosmology_cache = CosmologyCache(c, problem.redshift_grid)
    prior = population_prior(problem.model, Λ; z_grid = problem.redshift_grid)
    iw = compute_importance_weights(problem, Λ, cosmology_cache, prior)
    redshift_prior = _redshift_prior_distribution(prior).prior
    rate = merger_rate_per_sec(
        redshift_prior,
        problem.local_merger_rate,
        problem.observation.observation_time_yr,
        problem.observation.observation_time_sec
    )
    sd = spectral_density(problem.proposal.cached_flux_over_dgw2, rate; weights = iw.weights)
    return merge(iw,
        (
            redshift_integral = redshift_integral(redshift_prior),
            expected_number_of_sources = rate * problem.observation.observation_time_sec,
            spectral_density = sd,
            spectral_density_in_band = sd[problem.observation.in_band_mask]
        ))
end

function loglikelihood(
        Λ::NamedTuple,
        problem::ImportanceSamplingProblem;
        observed_spectral_density::AbstractVector{<:Real} = problem.observation.fiducial_spectral_density
)
    evaluation = evaluate_model_terms(Λ, problem)
    observed_in_band = observed_spectral_density[problem.observation.in_band_mask]
    residual = observed_in_band .- evaluation.spectral_density_in_band
    return -0.5 * sum(
        (residual ./ problem.observation.sgwb_scale_in_band) .^ 2 .+
        log.(2π .* (problem.observation.sgwb_scale_in_band .^ 2)),
    )
end

function fiducial_hyperparameters(problem::ImportanceSamplingProblem)
    problem.fiducial_hyperparameters
end

function fiducial_spectral_density(problem::ImportanceSamplingProblem)
    return evaluate_model_terms(fiducial_hyperparameters(problem), problem).spectral_density
end

function fiducial_redshift_integral(problem::ImportanceSamplingProblem)
    prior = population_prior(
        problem.model,
        problem.fiducial_hyperparameters;
        z_grid = redshift_grid(problem.redshift_grid_spec)
    )
    return Float64(redshift_integral(_redshift_prior_distribution(prior).prior))
end

function fiducial_redshift_integral(
        model::PhysicalModel,
        Λ::NamedTuple,
        spec::RedshiftPriorSpec
)
    prior = population_prior(model, Λ; z_grid = redshift_grid(spec))
    return Float64(redshift_integral(_redshift_prior_distribution(prior).prior))
end
