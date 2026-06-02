using Distributions: MvNormal, ProductNamedTupleDistribution
using LinearAlgebra: Diagonal
using Turing
using Turing: DynamicPPL

function validate_hyperprior(order::Tuple{Vararg{Symbol}}, prior::ProductNamedTupleDistribution)
    keys(prior.dists) == order || throw(
        ArgumentError("hyperprior must match order $(order), got $(keys(prior.dists))"),
    )
    return nothing
end

function logposterior(
        Λ::NamedTuple,
        problem::ImportanceSamplingProblem,
        ::Type{C},
        ctx::ModelContext,
        prior::ProductNamedTupleDistribution;
        observed::AbstractVector{<:Real} = ctx.fiducial_spectral_density
) where {C <: AbstractCosmology}
    return logpdf(prior, Λ) +
           loglikelihood(Λ, problem, C, ctx; observed = observed)
end

function condition_turing_model(
        turing_model,
        theta0::NamedTuple,
        prior::ProductNamedTupleDistribution,
        sample_only::Union{Nothing, Tuple{Vararg{Symbol}}};
        order::Tuple{Vararg{Symbol}}
)
    validate_hyperprior(order, prior)
    ordered_theta0 = canonical_hyperparameters(order, theta0; context = "initial hyperparameters")
    sample_only === nothing && return turing_model
    isempty(sample_only) && throw(
        ArgumentError(
        "sample_only must not be empty; omit the key or use null to sample every hyperparameter",
    ),
    )
    validate_subset(sample_only, order)
    fixed = Tuple(s for s in order if s ∉ sample_only)
    isempty(fixed) && return turing_model
    return turing_model | (; (s => ordered_theta0[s] for s in fixed)...)
end

@model function sample_hyperparameters(order::Tuple{Vararg{Symbol}}, dists)
    values = map(order) do sym
        x ~ DynamicPPL.NamedDist(dists[sym], sym)
        x
    end
    return NamedTuple{order}(Tuple(values))
end

@model function asgwb_importance_turing_model(
        track::Bool,
        problem::ImportanceSamplingProblem,
        ::Val{C},
        ctx::ModelContext,
        prior::ProductNamedTupleDistribution,
        observed_in_band::AbstractVector{<:Real}
) where {C}
    order = full_hyperparameters(C, problem.population_model)
    Λ ~ to_submodel(sample_hyperparameters(order, prior.dists), false)
    Λc = canonical_hyperparameters(
        order,
        Λ;
        context = "sampled hyperparameters",
        eltype = nothing
    )

    weights = compute_importance_weights(problem, C, Λc, ctx)
    rate = merger_rate(problem, C, Λc, ctx)
    Sh = spectral_density(ctx.cached_flux_over_dgw2, rate; weights = weights)

    obs = ctx.observation
    observed_in_band ~ MvNormal(
        Sh[obs.in_band_mask],
        Diagonal(obs.sgwb_scale_in_band .^ 2)
    )

    track || return nothing
    m = obs.in_band_mask
    df = frequency_bin_width(obs.frequencies)
    snr_sq = spectral_snr_squared(
        Sh[m], obs.effective_psd[m], obs.observation_time_sec, df)
    return (;
        number_of_sources = rate * obs.observation_time_sec,
        effective_sample_size = normalized_ess(weights),
        spectral_snr_squared = snr_sq,
        spectral_snr = sqrt(snr_sq)
    )
end

function build_turing_model(
        problem::ImportanceSamplingProblem,
        ::Type{C},
        ctx::ModelContext,
        prior::ProductNamedTupleDistribution;
        track::Bool = false,
        observed::AbstractVector{<:Real} = ctx.fiducial_spectral_density
) where {C <: AbstractCosmology}
    order = full_hyperparameters(C, problem.population_model)
    validate_hyperprior(order, prior)
    return asgwb_importance_turing_model(
        track,
        problem,
        Val(C),
        ctx,
        prior,
        observed[ctx.observation.in_band_mask]
    )
end
