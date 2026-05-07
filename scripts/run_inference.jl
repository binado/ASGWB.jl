#!/usr/bin/env julia

import Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()

module RunInferenceCLI

using ASGWB
using ASGWB: load_cache, build_turing_model, Detector, DEFAULT_PARAMETER_ORDER

using Turing
using AdvancedHMC
using Random
using Serialization
using ArviZ
using NCDatasets
using Distributions
using TOML
using Pkg
using LinearAlgebra: BLAS
using MCMCChains: chainscat
using Dates: now, format
using Comonicon: @main


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

const PRIORS = (
    H0 = Uniform(20, 140),
    Ωm = Uniform(0.05, 0.95),
    Ξ₀ = Uniform(0.5, 5),
    Ξₙ = Uniform(0.05, 3),
    γ = Uniform(0.5, 10),
    κ = Uniform(0.05, 10),
    zpeak = Uniform(0.05, 10),
)

"""Resolve `path` relative to `base` if it is not absolute."""
resolve_path(path::AbstractString, base::AbstractString) =
    isabspath(path) ? path : normpath(joinpath(base, path))

"""
    sample_with_checkpoints(conditioned, nuts, n_samples, num_chains;
                            checkpoint_every, checkpoint_path, progress)

Sample in chunks of `checkpoint_every` iterations, serializing the cumulative chain
to `checkpoint_path` after each chunk. Uses `resume_from` so adaptation only happens
in the first chunk. If `checkpoint_every >= n_samples` (or non-positive) the run is
done in a single call.
"""
function sample_with_checkpoints(
        conditioned, nuts, n_samples::Int, num_chains::Int;
        checkpoint_every::Int, checkpoint_path::AbstractString, progress::Bool,
)
    if checkpoint_every <= 0 || checkpoint_every >= n_samples
        chain = sample(
            conditioned, nuts, MCMCThreads(), n_samples, num_chains;
            progress = progress, save_state = true,
        )
        return chain
    end

    first_chunk = min(checkpoint_every, n_samples)
    @info "sampling chunk" chunk_size=first_chunk so_far=0 target=n_samples
    chain = sample(
        conditioned, nuts, MCMCThreads(), first_chunk, num_chains;
        progress = progress, save_state = true,
    )
    Serialization.serialize(checkpoint_path, chain)
    @info "checkpoint written" path=checkpoint_path samples=size(chain, 1)

    while size(chain, 1) < n_samples
        chunk = min(checkpoint_every, n_samples - size(chain, 1))
        @info "sampling chunk" chunk_size=chunk so_far=size(chain, 1) target=n_samples
        new_chain = sample(
            conditioned, nuts, MCMCThreads(), chunk, num_chains;
            progress = progress, save_state = true, resume_from = chain,
        )
        chain = chainscat(chain, new_chain)
        Serialization.serialize(checkpoint_path, chain)
        @info "checkpoint written" path=checkpoint_path samples=size(chain, 1)
    end

    return chain
end


function _run(settings::Dict, settings_dir::AbstractString)
    cache = resolve_path(settings["cache_path"]::String, settings_dir)
    detectors = [Detector(n) for n in settings["detectors"]]
    sample_only = Tuple(Symbol(s) for s in settings["sample_only"])
    seed = settings["seed"]::Int
    init = (; (Symbol(k) => v for (k, v) in settings["init"])...)

    sampler = settings["sampler"]
    n_samples = sampler["n_samples"]::Int
    n_adapts = sampler["n_adapts"]::Int
    target_acceptance = sampler["target_acceptance"]::Float64
    num_chains = get(sampler, "num_chains", 0)::Int
    num_chains = num_chains > 0 ? num_chains : Base.Threads.nthreads()
    checkpoint_every = get(sampler, "checkpoint_every", 0)::Int

    output_dir = resolve_path(get(settings, "output_dir", ".")::String, settings_dir)
    output_prefix = get(settings, "output_prefix", "chains")::String
    mkpath(output_dir)

    # Validate sample_only
    isempty(sample_only) && throw(ArgumentError("sample_only must not be empty"))
    length(unique(sample_only)) == length(sample_only) ||
        throw(ArgumentError("sample_only must not repeat symbols"))
    for s in sample_only
        s in DEFAULT_PARAMETER_ORDER || throw(
            ArgumentError("sample_only contains $(repr(s)); expected symbols from $(DEFAULT_PARAMETER_ORDER)"),
        )
    end

    fixed_sites = (; (k => init[k] for k in DEFAULT_PARAMETER_ORDER if k ∉ sample_only)...)

    timestamp = format(now(), "yyyymmdd-HHMMSS")
    params_suffix = join(sample_only, "-")
    base = "$(output_prefix)-$(params_suffix)-seed$(seed)-$(timestamp)"
    output_jls = joinpath(output_dir, "$base.jls")
    output_netcdf = joinpath(output_dir, "$base.nc")
    checkpoint_path = joinpath(output_dir, "$base.partial.jls")

    validate_init_against_priors(PRIORS, init)
    priors_turing = product_distribution(PRIORS)

    # Cluster-friendly defaults: avoid BLAS oversubscription with MCMCThreads
    # and disable the carriage-return progress bar in non-TTY contexts.
    BLAS.set_num_threads(1)
    progress = isinteractive()

    num_threads = Base.Threads.nthreads()
    if num_chains != num_threads
        @warn "num_chains differs from Base.Threads.nthreads()" num_chains num_threads
    end

    @info "starting run" julia=VERSION threads=num_threads chains=num_chains blas_threads=BLAS.get_num_threads() cache detectors=join((d.name for d in detectors), ",") sample_only output_dir
    @info "package versions"
    Pkg.status()

    @info "loading importance cache" path=cache
    t_cache = time()
    problem = load_cache(cache, detectors)
    @info "cache loaded" seconds=round(time()-t_cache; digits = 2) n_frequency_bins=length(problem.observation.frequencies) n_proposal_samples=length(problem.proposal.samples.redshift)

    @info "using fiducial in-band spectrum from cache as observed data"
    observed = problem.observation.fiducial_spectral_density

    @info "seeding RNG" rng_seed=seed
    Random.seed!(seed)

    @info "starting NUTS" n_adapts n_samples target_acceptance sample_only checkpoint_every
    model = build_turing_model(problem, priors_turing; track = true, observed_spectral_density = observed)
    conditioned = model | fixed_sites
    nuts = Turing.NUTS(
        n_adapts,
        target_acceptance;
        metricT = AdvancedHMC.DenseEuclideanMetric,
    )
    chain = sample_with_checkpoints(
        conditioned, nuts, n_samples, num_chains;
        checkpoint_every = checkpoint_every,
        checkpoint_path = checkpoint_path,
        progress = progress,
    )
    @info "NUTS finished" chain_size=size(chain)

    @info "writing chain to JLS" path=output_jls
    Serialization.serialize(output_jls, chain)
    @info "wrote chain to JLS" path=output_jls

    @info "writing InferenceData to NetCDF" path=output_netcdf
    idata = from_mcmcchains(chain; library = "Turing")
    to_netcdf(idata, output_netcdf)
    @info "wrote InferenceData to NetCDF" path=output_netcdf

    if isfile(checkpoint_path)
        @info "removing checkpoint" path=checkpoint_path
        rm(checkpoint_path; force = true)
    end

    @info "done"
    return nothing
end

@main function run_inference(; settings::String = "")
    settings_path = isempty(settings) ? joinpath(@__DIR__, "run_inference.toml") : settings
    settings_path = abspath(settings_path)
    @info "loading settings" path=settings_path
    s = TOML.parsefile(settings_path)
    return _run(s, dirname(settings_path))
end

end # module RunInferenceCLI

Base.invokelatest(RunInferenceCLI.command_main)
