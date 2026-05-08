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
using MCMCChains: Chains
using AbstractMCMC: bundle_samples
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
    zpeak = Uniform(0.05, 10)
)

"""Resolve `path` relative to `base` if it is not absolute."""
function resolve_path(path::AbstractString, base::AbstractString)
    isabspath(path) ? path : normpath(joinpath(base, path))
end

"""
    CheckpointCallback(every, base, output_dir, model, sampler, num_chains)

AbstractMCMC callback that buffers transitions separately for each chain and
serializes a single-chain `MCMCChains.Chains` snapshot to
`base.partial.chainN.jls` every time that chain crosses a new multiple of
`every`. With `save_state = true` on the bundled samples, the snapshot retains
the matching sampler state for manual recovery/debugging.
"""
mutable struct CheckpointCallback{M, S}
    every::Int
    base::String
    output_dir::String
    model::M
    sampler::S
    transitions::Vector{Vector{Any}}
    states::Vector{Any}
    last_checkpoint_iters::Vector{Int}
    lock::ReentrantLock
end

function CheckpointCallback(
        every::Int, base::AbstractString, output_dir::AbstractString, model, sampler,
        num_chains::Int
)
    return CheckpointCallback(
        every,
        String(base),
        String(output_dir),
        model,
        sampler,
        [Vector{Any}() for _ in 1:num_chains],
        Vector{Any}(undef, num_chains),
        zeros(Int, num_chains),
        ReentrantLock()
    )
end

function checkpoint_path(cb::CheckpointCallback, chain_number::Int)
    return joinpath(cb.output_dir, "$(cb.base).partial.chain$(chain_number).jls")
end

function checkpoint_paths(cb::CheckpointCallback, num_chains::Int)
    return checkpoint_path.(Ref(cb), 1:num_chains)
end

function (cb::CheckpointCallback)(
        rng, model, sampler, transition, state, iteration;
        chain_number::Int = 1, kwargs...
)
    lock(cb.lock) do
        push!(cb.transitions[chain_number], transition)
        cb.states[chain_number] = state

        n = length(cb.transitions[chain_number])
        target = (cb.last_checkpoint_iters[chain_number] ÷ cb.every + 1) * cb.every
        n >= target || return nothing

        chain_transitions = cb.transitions[chain_number]
        typed_transitions = Vector{typeof(chain_transitions[1])}(undef, n)
        copyto!(typed_transitions, 1, chain_transitions, 1, n)
        snapshot = bundle_samples(
            typed_transitions, cb.model, cb.sampler, cb.states[chain_number],
            Chains; save_state = true
        )

        path = checkpoint_path(cb, chain_number)
        tmp = path * ".tmp"
        Serialization.serialize(tmp, snapshot)
        mv(tmp, path; force = true)
        cb.last_checkpoint_iters[chain_number] = n
        @info "checkpoint written" path=path chain=chain_number iteration=n
    end
    return nothing
end


function _run(settings::Dict, settings_dir::AbstractString; interactive::Bool = false)
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

    validate_init_against_priors(PRIORS, init)
    priors_turing = product_distribution(PRIORS)

    # Cluster-friendly defaults: avoid BLAS oversubscription with MCMCThreads.
    BLAS.set_num_threads(1)
    progress = interactive

    num_threads = Base.Threads.nthreads()
    if num_chains != num_threads
        @warn "num_chains differs from Base.Threads.nthreads()" num_chains num_threads
    end

    @info "starting run" julia=VERSION threads=num_threads chains=num_chains blas_threads=BLAS.get_num_threads() cache detectors=join(
        (d.name for d in detectors), ",") sample_only output_dir
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
        metricT = AdvancedHMC.DenseEuclideanMetric
    )
    callback = checkpoint_every > 0 ?
               CheckpointCallback(
        checkpoint_every, base, output_dir, conditioned, nuts, num_chains
    ) : nothing

    chain = if callback === nothing
        sample(
            conditioned, nuts, MCMCThreads(), n_samples, num_chains;
            progress = progress, save_state = true
        )
    else
        sample(
            conditioned, nuts, MCMCThreads(), n_samples, num_chains;
            progress = progress, save_state = true, callback = callback
        )
    end
    @info "NUTS finished" chain_size=size(chain)

    @info "writing chain to JLS" path=output_jls
    Serialization.serialize(output_jls, chain)
    @info "wrote chain to JLS" path=output_jls

    @info "writing InferenceData to NetCDF" path=output_netcdf
    idata = from_mcmcchains(chain; library = "Turing")
    to_netcdf(idata, output_netcdf)
    @info "wrote InferenceData to NetCDF" path=output_netcdf

    if callback !== nothing
        retained_checkpoint_paths = filter(isfile, checkpoint_paths(callback, num_chains))
        if !isempty(retained_checkpoint_paths)
            @info "retaining partial checkpoint files" count=length(retained_checkpoint_paths) output_dir base
        end
    end

    @info "done"
    return nothing
end

function resolve_config_path(config::AbstractString)
    default_path = joinpath(@__DIR__, "run_inference.toml")
    settings_path = isempty(config) ? get(ENV, "MCMC_CONFIG_FILEPATH", default_path) :
                    config
    return abspath(settings_path)
end

recursive_merge(x::AbstractDict...) = merge(recursive_merge, x...)
recursive_merge(_, x) = x

function cli_overrides(;
        seed::Int,
        output_dir::AbstractString,
        output_prefix::AbstractString,
        num_chains::Int,
        n_samples::Int,
        n_adapts::Int,
        checkpoint_every::Int
)
    overrides = Dict{String, Any}("seed" => seed)
    isempty(output_dir) || (overrides["output_dir"] = String(output_dir))
    isempty(output_prefix) || (overrides["output_prefix"] = String(output_prefix))

    sampler = Dict{String, Any}()
    num_chains >= 0 && (sampler["num_chains"] = num_chains)
    n_samples > 0 && (sampler["n_samples"] = n_samples)
    n_adapts > 0 && (sampler["n_adapts"] = n_adapts)
    checkpoint_every >= 0 && (sampler["checkpoint_every"] = checkpoint_every)
    isempty(sampler) || (overrides["sampler"] = sampler)

    return overrides
end

"""
Run ASGWB inference from a TOML configuration file.

# Options

- `--config=<path>`: TOML settings file. Falls back to `MCMC_CONFIG_FILEPATH`,
  then `scripts/run_inference.toml`.

- `--seed=<int>`: RNG seed.

- `--output-dir=<path>`: override `output_dir` from the TOML settings.

- `--output-prefix=<name>`: override `output_prefix` from the TOML settings.

- `--num-chains=<int>`: override `sampler.num_chains` from the TOML settings.

- `--n-samples=<int>`: override `sampler.n_samples` from the TOML settings.

- `--n-adapts=<int>`: override `sampler.n_adapts` from the TOML settings.

- `--checkpoint-every=<int>`: override `sampler.checkpoint_every` from the TOML settings.

- `--interactive`: enable Turing's sampling progress bar.
"""
@main function run_inference(;
        config::String = "",
        seed::Int = 42,
        output_dir::String = "",
        output_prefix::String = "",
        num_chains::Int = -1,
        n_samples::Int = 0,
        n_adapts::Int = 0,
        checkpoint_every::Int = -1,
        interactive::Bool = false
)
    settings_path = resolve_config_path(config)
    @info "loading settings" path=settings_path
    s = TOML.parsefile(settings_path)

    s = recursive_merge(
        s,
        cli_overrides(;
            seed,
            output_dir,
            output_prefix,
            num_chains,
            n_samples,
            n_adapts,
            checkpoint_every
        )
    )

    return _run(s, dirname(settings_path); interactive)
end

end # module RunInferenceCLI

Base.invokelatest(RunInferenceCLI.command_main)
