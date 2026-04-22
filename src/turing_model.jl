using Distributions: MvNormal, ProductNamedTupleDistribution
using LinearAlgebra: Diagonal
using Turing

"""
    condition_turing_model(model, theta0, sample_only::Union{Nothing,Tuple}) -> model

If `sample_only === nothing`, return `model` unchanged (all hyperparameters are sampled).

Otherwise `sample_only` lists the subset of [`DEFAULT_PARAMETER_ORDER`](@ref) that remain
stochastic; all other hyperparameters are **fixed** to the corresponding entries of `theta0`
using Turing’s conditioning operator `|` (see
[Turing docs: conditioning on data](https://turinglang.org/docs/core-functionality/#conditioning-on-data)).
"""
function condition_turing_model(
        model,
        theta0::HyperParameters,
        sample_only::Union{Nothing, Tuple{Vararg{Symbol}}}
)
    sample_only === nothing && return model
    fixed = Tuple(s for s in DEFAULT_PARAMETER_ORDER if s ∉ sample_only)
    isempty(fixed) && return model
    return model | (; (s => theta0[s] for s in fixed)...)
end

function _validate_sample_only(sample_only::Union{Nothing, Tuple{Vararg{Symbol}}})
    sample_only === nothing && return nothing
    isempty(sample_only) && throw(
        ArgumentError(
        "sample_only must not be empty; omit the key or use null to sample every hyperparameter",
    ),
    )
    for s in sample_only
        s in DEFAULT_PARAMETER_ORDER || throw(
            ArgumentError(
            "sample_only contains $(repr(s)); expected symbols from $(DEFAULT_PARAMETER_ORDER)",
        ),
        )
    end
    length(unique(sample_only)) == length(sample_only) ||
        throw(ArgumentError("sample_only must not repeat symbols"))
    return nothing
end

function _turing_initial_params(
        theta0::HyperParameters,
        sample_only::Union{Nothing, Tuple{Vararg{Symbol}}}
)
    sample_only === nothing && return InitFromParams(theta0)
    return InitFromParams((; (s => theta0[s] for s in sample_only)...))
end

@model function asgwb_importance_turing_model(
        problem::ImportanceSamplingProblem,
        prior::ProductNamedTupleDistribution,
        observed_in_band::AbstractVector{<:Real}
)
    d = prior.dists
    H0 ~ d.H0
    Omega_m ~ d.Omega_m
    chi0 ~ d.chi0
    chin ~ d.chin
    gamma ~ d.gamma
    kappa ~ d.kappa
    z_peak ~ d.z_peak

    h = (; H0, Omega_m, chi0, chin, gamma, kappa, z_peak)

    bundle = build_redshift_grid_bundle(h, problem.redshift_prior_spec)
    iw = compute_importance_weights(problem, h, bundle)
    rate = merger_rate_per_sec(
        bundle,
        problem.local_merger_rate,
        problem.observation.observation_time_yr,
        problem.observation.observation_time_sec
    )
    sd = spectral_density(problem.proposal.cached_flux_over_dgw2, rate; weights = iw.weights)
    sd_in_band = sd[problem.observation.in_band_mask]

    observed_in_band ~
    MvNormal(sd_in_band, Diagonal(problem.observation.sgwb_scale_in_band .^ 2))

    return (;
        number_of_sources = rate * problem.observation.observation_time_sec,
        spectral_density = sd,
        effective_sample_size = normalized_ess(iw.weights)
    )
end

function build_turing_model(
        problem::ImportanceSamplingProblem,
        prior::ProductNamedTupleDistribution;
        observed_spectral_density::AbstractVector{<:Real} = problem.observation.fiducial_spectral_density
)
    return asgwb_importance_turing_model(
        problem,
        prior,
        observed_spectral_density[problem.observation.in_band_mask]
    )
end

"""
    sample_with_turing(problem, prior, theta0; kwargs...) -> (chain, model)

NUTS sampling for the importance likelihood model. Keyword `sample_only` lists which of
`DEFAULT_PARAMETER_ORDER` remain stochastic; all others are fixed to the corresponding entries
of `theta0` using Turing’s `|` operator (see [`condition_turing_model`](@ref)).
"""
function sample_with_turing(
        problem::ImportanceSamplingProblem,
        prior::ProductNamedTupleDistribution,
        theta0::HyperParameters;
        n_adapts::Int = 25,
        n_samples::Int = 25,
        target_acceptance::Float64 = 0.8,
        observed_spectral_density::AbstractVector{<:Real} = problem.observation.fiducial_spectral_density,
        sample_only::Union{Nothing, Tuple{Vararg{Symbol}}} = nothing
)
    _validate_sample_only(sample_only)
    model = build_turing_model(
        problem,
        prior;
        observed_spectral_density = observed_spectral_density
    )
    conditioned = condition_turing_model(model, theta0, sample_only)
    chain = sample(
        conditioned,
        Turing.NUTS(n_adapts, target_acceptance),
        n_samples;
        progress = false,
        initial_params = _turing_initial_params(theta0, sample_only)
    )
    return chain, conditioned
end
