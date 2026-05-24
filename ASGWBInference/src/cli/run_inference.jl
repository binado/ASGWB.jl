module RunInferenceCLI

using ASGWB
using ASGWB:
             load_cache,
             Detector,
             MadauDickinsonModifiedPropagation,
             cosmology_type,
             propagation_model,
             canonical_hyperparameters,
             hyperparameters,
             validate_hyperparameters,
             validate_prior,
             validate_subset
using ..ChainIO: atomic_save_chain
using ..InferenceImpl: build_turing_model, condition_turing_model

using Turing
using AdvancedHMC
using Random
using ADTypes
using JLD2
using Distributions
using TOML
using Pkg
using LinearAlgebra: BLAS
using AbstractMCMC: bundle_samples
using FlexiChains: VNChain
using Dates: now, format

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
    w0 = Uniform(-3.0, 0.0),
    wa = Uniform(-3.0, 3.0),
    Ξ₀ = Uniform(0.5, 5),
    Ξₙ = Uniform(0.05, 3),
    γ = Uniform(0.5, 10),
    κ = Uniform(0.05, 10),
    zpeak = Uniform(0.05, 10)
)

"""Select the subset of `priors` corresponding to `names`."""
function select_priors(priors::NamedTuple, names::Tuple{Vararg{Symbol}})
    return NamedTuple{names}(priors)
end

function _resolve_inference_model(settings::Dict)
    cosmology_name = get(get(settings, "model", Dict()), "cosmology", "LambdaCDM")::String
    C = cosmology_type(cosmology_name)
    return MadauDickinsonModifiedPropagation{C}()
end

"""Resolve `path` relative to `base` if it is not absolute."""
function resolve_path(path::AbstractString, base::AbstractString)
    isabspath(path) ? path : normpath(joinpath(base, path))
end

const DEFAULT_CONFIG_RELATIVE_PATH = joinpath("config", "run_inference.toml")

function _has_repo_markers(path::AbstractString)
    return isfile(joinpath(path, "Project.toml")) &&
           isdir(joinpath(path, "ASGWB")) &&
           isdir(joinpath(path, "ASGWBInference"))
end

function repo_root(start::AbstractString = pwd())
    env_root = get(ENV, "ASGWB_REPO_ROOT", "")
    if !isempty(env_root)
        return normpath(abspath(env_root))
    end

    current = normpath(abspath(start))
    while true
        _has_repo_markers(current) && return current
        parent = dirname(current)
        parent == current && break
        current = parent
    end

    throw(ArgumentError(
        "could not find ASGWB.jl repository root from $(repr(start)); " *
        "set ASGWB_REPO_ROOT to the repository root"
    ))
end

function default_config_path()
    return joinpath(repo_root(), DEFAULT_CONFIG_RELATIVE_PATH)
end

function resolve_config_path(config::AbstractString)
    root = repo_root()
    raw_path = isempty(config) ?
               get(ENV, "MCMC_CONFIG_FILEPATH", DEFAULT_CONFIG_RELATIVE_PATH) :
               config
    return isabspath(raw_path) ? normpath(raw_path) : normpath(joinpath(root, raw_path))
end

"""
    CheckpointCallback(every, base, output_dir, model, sampler, num_chains)

AbstractMCMC callback that buffers transitions separately for each chain and
saves a single-chain `FlexiChains.VNChain` snapshot to
`base.partial.chainN.jld2` every time that chain crosses a new multiple of
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
    save_state::Bool
end

function CheckpointCallback(
        every::Int, base::AbstractString, output_dir::AbstractString, model, sampler,
        num_chains::Int; save_state::Bool = true
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
        save_state
    )
end

function checkpoint_path(cb::CheckpointCallback, chain_number::Int)
    return joinpath(cb.output_dir, "$(cb.base).partial.chain$(chain_number).jld2")
end

function checkpoint_paths(cb::CheckpointCallback, num_chains::Int)
    return checkpoint_path.(Ref(cb), 1:num_chains)
end

function (cb::CheckpointCallback)(
        rng, model, sampler, transition, state, iteration;
        chain_number::Int = 1, kwargs...
)
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
        VNChain; save_state = cb.save_state
    )

    path = checkpoint_path(cb, chain_number)
    tmp = path * ".tmp"
    jldsave(tmp; snapshot)
    mv(tmp, path; force = true)
    cb.last_checkpoint_iters[chain_number] = n
    @info "checkpoint written" path=path chain=chain_number iteration=n
    return nothing
end


"""Map a config-string to an `ADTypes.AbstractADType` for Turing's NUTS."""
function resolve_adtype(name::AbstractString)
    if name == "ForwardDiff"
        return ADTypes.AutoForwardDiff()
    else
        throw(ArgumentError(
            "this inference CLI supports only ad_backend = \"ForwardDiff\"; got $(repr(name))"
        ))
    end
end

function parse_sample_only(settings::Dict)
    raw = get(settings, "sample_only", nothing)
    raw === nothing && return nothing
    raw isa Vector || throw(
        ArgumentError("sample_only must be null, omitted, or an array of strings"),
    )
    all(x -> x isa AbstractString, raw) ||
        throw(ArgumentError("sample_only must be an array of strings"))
    return Tuple(Symbol(s) for s in raw)
end

function _run(settings::Dict, settings_dir::AbstractString; interactive::Bool = false)
    cache = resolve_path(settings["cache_path"]::String, settings_dir)
    detectors = [Detector(n) for n in settings["detectors"]]
    sample_only = parse_sample_only(settings)
    seed = settings["seed"]::Int
    inference_model = _resolve_inference_model(settings)
    model_params = hyperparameters(inference_model)
    all_init = (; (Symbol(k) => v for (k, v) in settings["init"])...)
    init = canonical_hyperparameters(
        inference_model,
        NamedTuple{model_params}(all_init);
        context = "init hyperparameters"
    )

    sampler = settings["sampler"]
    n_samples = sampler["n_samples"]::Int
    n_adapts = sampler["n_adapts"]::Int
    target_acceptance = sampler["target_acceptance"]::Float64
    ad_backend_name = get(sampler, "ad_backend", "ForwardDiff")::String
    adtype = resolve_adtype(ad_backend_name)
    num_chains = get(sampler, "num_chains", 0)::Int
    num_chains = num_chains > 0 ? num_chains : Base.Threads.nthreads()
    checkpoint_every = get(sampler, "checkpoint_every", 0)::Int

    output_dir = resolve_path(get(settings, "output_dir", ".")::String, settings_dir)
    output_prefix = get(settings, "output_prefix", "chains")::String
    mkpath(output_dir)

    timestamp = format(now(), "yyyymmdd-HHMMSS")
    params_suffix = sample_only === nothing ? "all" : join(sample_only, "-")
    base = "$(output_prefix)-$(params_suffix)-seed$(seed)-$(timestamp)"
    output_jld2 = joinpath(output_dir, "$base.jld2")

    selected_priors = select_priors(PRIORS, model_params)
    validate_init_against_priors(selected_priors, init)
    priors_turing = product_distribution(selected_priors)
    validate_prior(inference_model, priors_turing)
    validate_hyperparameters(inference_model, init; context = "init hyperparameters")
    if sample_only !== nothing
        isempty(sample_only) && throw(
            ArgumentError(
            "sample_only must not be empty; omit the key or use null to sample every hyperparameter",
        ),
        )
        validate_subset(sample_only, inference_model)
    end

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
    problem = load_cache(cache, detectors)
    cache_cosmology = cosmology_type(problem.fiducial_parameters)
    infer_cosmology = cosmology_type(inference_model)
    cache_cosmology === infer_cosmology || throw(
        ArgumentError(
        "cache cosmology $(cache_cosmology) does not match [model].cosmology " *
        "($(infer_cosmology)); rebuild the cache or fix config",
    ),
    )
    @info "cache loaded" n_frequency_bins=length(problem.observation.frequencies) n_proposal_samples=length(problem.proposal.samples.redshift)

    @info "using fiducial in-band spectrum from cache as observed data"
    observed = problem.observation.fiducial_spectral_density

    @info "seeding RNG" rng_seed=seed
    Random.seed!(seed)

    final_save_state = false
    checkpoint_save_state = true

    @info "starting NUTS" n_adapts n_samples target_acceptance ad_backend=ad_backend_name sample_only checkpoint_every
    turing_model = build_turing_model(
        problem,
        priors_turing;
        model = inference_model,
        track = true,
        observed_spectral_density = observed
    )
    conditioned = condition_turing_model(
        turing_model,
        init,
        priors_turing,
        sample_only;
        model = inference_model
    )
    nuts = Turing.NUTS(
        n_adapts,
        target_acceptance;
        metricT = AdvancedHMC.DenseEuclideanMetric,
        adtype = adtype
    )
    callback = checkpoint_every > 0 ?
               CheckpointCallback(
        checkpoint_every, base, output_dir, conditioned, nuts, num_chains;
        save_state = checkpoint_save_state
    ) : nothing

    chain = if callback === nothing
        sample(
            conditioned, nuts, MCMCThreads(), n_samples, num_chains;
            progress = progress, save_state = final_save_state, chain_type = VNChain
        )
    else
        sample(
            conditioned, nuts, MCMCThreads(), n_samples, num_chains;
            progress = progress, save_state = final_save_state, callback = callback,
            chain_type = VNChain
        )
    end
    @info "NUTS finished" chain_size=size(chain)

    @info "writing chain to JLD2" path=output_jld2
    atomic_save_chain(output_jld2, chain)

    if callback !== nothing
        paths = filter(isfile, checkpoint_paths(callback, num_chains))
        if !isempty(paths)
            rm.(paths; force = true)
            @info "removed partial checkpoint files" count = length(paths) output_dir base
        end
    end

    @info "done"
    return nothing
end

"""
Run ASGWB inference from a TOML configuration file.

Configuration is selected with `MCMC_CONFIG_FILEPATH`, or defaults to
`config/run_inference.toml` relative to the repository root. A relative
`MCMC_CONFIG_FILEPATH` is also resolved against the repository root.
"""
function run(config::AbstractString)
    settings_path = resolve_config_path(config)
    @info "loading settings" path=settings_path
    s = TOML.parsefile(settings_path)

    return _run(s, dirname(settings_path); interactive = false)
end

function run_from_env()
    return run("")
end

end # module RunInferenceCLI
