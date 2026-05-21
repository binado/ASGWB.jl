using Distributions: MvNormal, ProductNamedTupleDistribution
using LinearAlgebra: Diagonal
using Turing

using ASGWB: hyperparameter_order, validate_sample_only!

"""
    condition_turing_model(model, theta0, prior, sample_only::Union{Nothing,Tuple}) -> model

If `sample_only === nothing`, return `model` unchanged (all hyperparameters are sampled).

Otherwise `sample_only` lists the subset of [`hyperparameter_order`](@ref)(`prior`) that remain
stochastic; all other hyperparameters are **fixed** to the corresponding entries of `theta0`
using Turing’s conditioning operator `|` (see
[Turing docs: conditioning on data](https://turinglang.org/docs/core-functionality/#conditioning-on-data)).
"""
function condition_turing_model(
        model,
        theta0::NamedTuple,
        prior::ProductNamedTupleDistribution,
        sample_only::Union{Nothing, Tuple{Vararg{Symbol}}}
)
    sample_only === nothing && return model
    validate_sample_only!(sample_only, prior)
    order = hyperparameter_order(prior)
    fixed = Tuple(s for s in order if s ∉ sample_only)
    isempty(fixed) && return model
    return model | (; (s => theta0[s] for s in fixed)...)
end

function _turing_initial_params(
        theta0::NamedTuple,
        sample_only::Union{Nothing, Tuple{Vararg{Symbol}}}
)
    sample_only === nothing && return InitFromParams(theta0)
    return InitFromParams((; (s => theta0[s] for s in sample_only)...))
end

@model function asgwb_importance_turing_model(
        track::Bool,
        problem::ImportanceSamplingProblem,
        prior::ProductNamedTupleDistribution,
        z_grid::AbstractVector{<:Real},
        observed_in_band::AbstractVector{<:Real}
)
    d = prior.dists
    H0 ~ d.H0
    Ωm ~ d.Ωm
    Ξ₀ ~ d.Ξ₀
    Ξₙ ~ d.Ξₙ
    γ ~ d.γ
    κ ~ d.κ
    zpeak ~ d.zpeak

    h = (; H0, Ωm, Ξ₀, Ξₙ, γ, κ, zpeak)

    cosmology_cache,
    redshift_prior = cosmology_and_redshift_prior(
        h, problem.redshift_prior_spec, z_grid)
    iw = compute_importance_weights(problem, h, cosmology_cache, redshift_prior)
    rate = merger_rate_per_sec(
        redshift_prior,
        problem.local_merger_rate,
        problem.observation.observation_time_yr,
        problem.observation.observation_time_sec
    )
    sd = spectral_density(problem.proposal.cached_flux_over_dgw2, rate; weights = iw.weights)
    sd_in_band = sd[problem.observation.in_band_mask]

    observed_in_band ~
    MvNormal(sd_in_band, Diagonal(problem.observation.sgwb_scale_in_band .^ 2))

    if track
        m = problem.observation.in_band_mask
        obs = problem.observation
        df = frequency_bin_width(obs.frequencies)
        snr_sq = spectral_snr_squared(
            sd[m], obs.effective_psd[m], obs.observation_time_sec, df
        )

        return (;
            number_of_sources = rate * problem.observation.observation_time_sec,
            effective_sample_size = normalized_ess(iw.weights),
            spectral_snr_squared = snr_sq,
            spectral_snr = sqrt(snr_sq)
        )
    end

    return nothing
end

"""
    build_turing_model(problem, prior; track=false, observed_spectral_density=...) -> model

Construct a Turing `DynamicPPL.Model` for the ASGWB importance sampling likelihood.

# Arguments
- `problem::ImportanceSamplingProblem`: The pre-computed importance sampling cache.
- `prior::ProductNamedTupleDistribution`: Priors for the hyperparameters.
- `track::Bool`: If `true`, the model returns a named tuple of diagnostic quantities
  (ESS, SNR, etc.) alongside the log-joint.
- `observed_spectral_density`: The "data" to condition on. Defaults to the fiducial
  spectral density from the cache.
"""
function build_turing_model(
        problem::ImportanceSamplingProblem,
        prior::ProductNamedTupleDistribution;
        track::Bool = false,
        observed_spectral_density::AbstractVector{<:Real} = problem.observation.fiducial_spectral_density
)
    return asgwb_importance_turing_model(
        track,
        problem,
        prior,
        problem.redshift_cache.redshift_grid,
        observed_spectral_density[problem.observation.in_band_mask]
    )
end

"""
    sample_with_turing(problem, prior, theta0; kwargs...) -> (chain, model)

NUTS sampling for the importance likelihood model. Keyword `sample_only` lists which of
[`hyperparameter_order`](@ref)(`prior`) remain stochastic; all others are fixed to the corresponding entries
of `theta0` using Turing’s `|` operator (see [`condition_turing_model`](@ref)).
"""
function sample_with_turing(
        problem::ImportanceSamplingProblem,
        prior::ProductNamedTupleDistribution,
        theta0::NamedTuple;
        n_adapts::Int = 25,
        n_samples::Int = 25,
        target_acceptance::Float64 = 0.8,
        track::Bool = false,
        observed_spectral_density::AbstractVector{<:Real} = problem.observation.fiducial_spectral_density,
        sample_only::Union{Nothing, Tuple{Vararg{Symbol}}} = nothing
)
    validate_sample_only!(sample_only, prior)
    model = build_turing_model(
        problem,
        prior;
        track = track,
        observed_spectral_density = observed_spectral_density
    )
    conditioned = condition_turing_model(model, theta0, prior, sample_only)
    chain = sample(
        conditioned,
        Turing.NUTS(n_adapts, target_acceptance),
        n_samples;
        progress = false,
        initial_params = _turing_initial_params(theta0, sample_only)
    )
    return chain, conditioned
end
