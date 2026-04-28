#!/usr/bin/env julia

import Pkg

Pkg.activate(joinpath(@__DIR__, "..", "notebooks"))
Pkg.instantiate()

module RunInferenceCLI

using ASGWB
using ASGWB:
    load_cache,
    build_turing_model,
    HyperParameters,
    Detector,
    DEFAULT_PARAMETER_ORDER

using Turing
using AdvancedHMC
using Random
using Serialization
using Logging
using MCMCChains
using ArviZ
using NCDatasets
using Distributions
using DelimitedFiles
using TOML
using Comonicon: @main


function load_observed_spectral_density(path::AbstractString, expected_len::Int)
    isfile(path) || throw(ArgumentError("observed spectrum file not found: $(repr(path))"))
    v = vec(readdlm(path, ',', Float64))
    length(v) == expected_len || throw(
        ArgumentError("observed_spectral_density_csv has length $(length(v)), expected $expected_len"),
    )
    return v
end

"""Check each `init` scalar has positive prior density under the matching `priors` entry."""
function validate_init_against_priors(priors, init)
    for (k, d) in pairs(priors)
        v = init[k]
        isfinite(logpdf(d, v)) || throw(
            ArgumentError("init.$k = $v is outside the support of the corresponding prior"),
        )
    end
    return nothing
end

function validate_sample_only(sample_only::Union{Nothing,Tuple{Vararg{Symbol}}})
    sample_only === nothing && return nothing
    isempty(sample_only) && throw(
        ArgumentError(
            "sample_only must not be empty; omit the key or use null to sample every hyperparameter",
        ),
    )
    for s in sample_only
        s in DEFAULT_PARAMETER_ORDER || throw(
            ArgumentError("sample_only contains $(repr(s)); expected symbols from $(DEFAULT_PARAMETER_ORDER)"),
        )
    end
    length(unique(sample_only)) == length(sample_only) ||
        throw(ArgumentError("sample_only must not repeat symbols"))
    return nothing
end

function _require(settings::Dict, key::AbstractString)
    haskey(settings, key) || throw(ArgumentError("missing required TOML key $(repr(key))"))
    return settings[key]
end

function _require_table(settings::Dict, key::AbstractString)
    v = _require(settings, key)
    v isa Dict || throw(ArgumentError("TOML key $(repr(key)) must be a table"))
    return v
end

function _require_string_array(settings::Dict, key::AbstractString)
    v = _require(settings, key)
    v isa Vector || throw(ArgumentError("TOML key $(repr(key)) must be an array"))
    all(x -> x isa AbstractString, v) ||
        throw(ArgumentError("TOML key $(repr(key)) must be an array of strings"))
    return Vector{String}(v)
end

function load_settings_toml(path::AbstractString)::Dict
    isfile(path) || throw(ArgumentError("settings TOML file not found: $(repr(path))"))
    settings = TOML.parsefile(path)
    settings isa Dict || throw(ArgumentError("expected TOML to parse to a Dict"))
    return settings
end


function _run(settings::Dict)
    # --- required settings ---
    cache = _require(settings, "cache_path")::String
    detectors = [Detector(n) for n in _require_string_array(settings, "detectors")]
    sample_only = Symbol.(_require_string_array(settings, "sample_only"))

    sampler_dict = _require_table(settings, "sampler")
    n_samples = _require(sampler_dict, "n_samples")::Int
    n_adapts = _require(sampler_dict, "n_adapts")::Int
    target_acceptance = Float64(_require(sampler_dict, "target_acceptance"))

    priors = (
        H0 = Uniform(20, 140),
        Ωm = Uniform(0.05, 0.95),
        Ξ₀ = Uniform(0.5, 5),
        Ξₙ = Uniform(0.05, 3),
        γ = Uniform(0.5, 10),
        κ = Uniform(0.05, 10),
        zpeak = Uniform(0.05, 10),
    )

    init = (H0 = 67.66, Ωm = 0.3096, Ξ₀ = 1.0, Ξₙ = 1.91, γ = 2.7, κ = 5.7, zpeak = 2.0)
    fixed_sites = (; (k => init[k] for k in DEFAULT_PARAMETER_ORDER if k ∉ sample_only)...)

    seed = 1
    observed_spectral_density_csv = nothing
    output_suffix = join(map(string, sample_only), "-")
    output_jls = "chains-$output_suffix.jls"
    output_netcdf = "chains-$output_suffix.nc"

    validate_init_against_priors(priors, init)
    priors_turing = product_distribution((
        H0 = priors.H0,
        Ωm = priors.Ωm,
        Ξ₀ = priors.Ξ₀,
        Ξₙ = priors.Ξₙ,
        γ = priors.γ,
        κ = priors.κ,
        zpeak = priors.zpeak,
    ))
    θ0 = HyperParameters(; init...)

    cd(pkgdir(ASGWB))

    num_threads = Base.Threads.nthreads()
    @info "starting run" threads=num_threads cache detectors=join((d.name for d in detectors), ",") sample_only

    @info "loading importance cache" path=cache
    t_cache = time()
    problem = load_cache(cache, detectors)
    @info "cache loaded" seconds=round(time() - t_cache; digits = 2) n_frequency_bins=length(problem.observation.frequencies) n_proposal_samples=length(problem.proposal.samples.redshift)

    observed = if observed_spectral_density_csv === nothing
        @info "using fiducial in-band spectrum from cache as observed data"
        problem.observation.fiducial_spectral_density
    else
        @info "loading observed spectrum from CSV" path=observed_spectral_density_csv
        load_observed_spectral_density(
            observed_spectral_density_csv,
            length(problem.observation.fiducial_spectral_density),
        )
    end

    if seed !== nothing
        @info "seeding RNG" rng_seed=seed
        Random.seed!(seed)
    else
        @info "RNG seed not set (nondeterministic run unless Julia was seeded elsewhere)"
    end

    sample_only_tup = sample_only === nothing ? nothing : Tuple(sample_only)
    validate_sample_only(sample_only_tup)

    @info "starting NUTS" n_adapts=n_adapts n_samples=n_samples target_acceptance=target_acceptance sample_only=sample_only_tup
    model = build_turing_model(problem, priors_turing; track = true, observed_spectral_density = observed)
    conditioned = model | fixed_sites
    nuts = Turing.NUTS(
        n_adapts,
        target_acceptance;
        metricT = AdvancedHMC.DenseEuclideanMetric,
    )
    chain = sample(
        conditioned,
        nuts,
        MCMCThreads(),
        n_samples,
        num_threads;
        progress = true,
        save_state = true,
    )
    @info "NUTS finished" chain_size=size(chain)

    @info "writing chain to JLS" path=output_jls
    Serialization.serialize(output_jls, chain)
    @info "wrote chain to JLS" path=output_jls

    idata = from_mcmcchains(chain; library = "Turing")
    if output_netcdf !== nothing
        @info "writing InferenceData to NetCDF" path=output_netcdf
        to_netcdf(idata, output_netcdf)
        @info "wrote InferenceData to NetCDF" path=output_netcdf
    end

    @info "done"
    return nothing
end

@main function run_inference(; settings::String = "")
    settings_path = isempty(settings) ? joinpath(@__DIR__, "run_inference.toml") : settings
    @info "loading settings" path=settings_path
    s = load_settings_toml(settings_path)
    return _run(s)
end

end # module RunInferenceCLI

Base.invokelatest(RunInferenceCLI.command_main)

