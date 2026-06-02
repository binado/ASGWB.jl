module RunInferenceCLI

using ASGWB
using ASGWB:
             Detector,
             validate_hyperparameters,
             validate_subset,
             full_hyperparameters,
             full_hyperprior
using ..ChainIO: atomic_save_chain
using ..InferenceImpl: build_turing_model, condition_turing_model, validate_hyperprior,
                       load_problem_context, POPULATION_REGISTRY

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

"""Check each fiducial scalar has positive prior density under the matching `priors` entry."""
function validate_init_against_priors(priors, fiducial)
    for (k, d) in pairs(priors)
        v = fiducial[k]
        isfinite(logpdf(d, v)) || throw(
            ArgumentError(
            "fiducial.$k = $v is outside the support of the corresponding prior",
        ),
        )
    end
    return nothing
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
`every`. With `save_state = true` on the saved samples, the snapshot retains
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

function parse_sample_only(settings::Dict, order::Tuple{Vararg{Symbol}})
    raw = get(settings, "sample_only", nothing)
    raw === nothing && return nothing
    raw isa Vector || throw(
        ArgumentError("sample_only must be null, omitted, or an array of strings"),
    )
    all(x -> x isa AbstractString, raw) ||
        throw(ArgumentError("sample_only must be an array of strings"))
    return Tuple(
        let s = Symbol(x)
            s in order ||
                throw(ArgumentError("sample_only contains unknown parameter $(repr(String(s)))"))
            s
        end
    for x in raw
    )
end

function _run(settings::Dict, settings_dir::AbstractString; interactive::Bool = false)
    catalog_path = resolve_path(settings["catalog_path"]::String, settings_dir)
    model_path = resolve_path(settings["model_path"]::String, settings_dir)
    detectors = [Detector(n) for n in settings["detectors"]]
    seed = settings["seed"]::Int

    local_merger_rate = Float64(settings["local_merger_rate"])
    observation_time_yr = Float64(settings["observation_time_yr"])

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

    @info "loading catalog" catalog_path model_path
    loaded = load_problem_context(
        catalog_path,
        model_path,
        detectors,
        POPULATION_REGISTRY;
        local_merger_rate = local_merger_rate,
        observation_time_yr = observation_time_yr
    )
    problem = loaded.problem
    C = loaded.cosmology_type
    ctx = loaded.ctx
    @info "catalog loaded" n_frequency_bins=length(ctx.observation.frequencies) n_proposal_samples=length(problem.samples.redshift)

    pop = problem.population_model
    order = full_hyperparameters(C, pop)
    hyperprior = full_hyperprior(C, pop)

    fiducial = problem.fiducial_hyperparameters
    initial_params = fill(InitFromPrior(), num_chains)

    sample_only = parse_sample_only(settings, order)

    validate_init_against_priors(hyperprior.dists, fiducial)
    validate_hyperprior(order, hyperprior)
    validate_hyperparameters(order, fiducial; context = "fiducial hyperparameters")
    if sample_only !== nothing
        isempty(sample_only) && throw(
            ArgumentError(
            "sample_only must not be empty; omit the key or use null to sample every hyperparameter",
        ),
        )
        validate_subset(sample_only, order)
    end

    # Cluster-friendly defaults: avoid BLAS oversubscription with MCMCThreads.
    BLAS.set_num_threads(1)
    progress = interactive

    num_threads = Base.Threads.nthreads()
    if num_chains != num_threads
        @warn "num_chains differs from Base.Threads.nthreads()" num_chains num_threads
    end

    timestamp = format(now(), "yyyymmdd-HHMMSS")
    det_suffix = join((d.name for d in detectors), ",")
    params_suffix = sample_only === nothing ? "all" : join(sample_only, "-")
    base = "$(output_prefix)-$(params_suffix)-det=$(det_suffix)-seed$(seed)-$(timestamp)"
    output_jld2 = joinpath(output_dir, "$base.jld2")

    @info "starting run" julia=VERSION threads=num_threads chains=num_chains blas_threads=BLAS.get_num_threads() catalog_path model_path detectors=join(
        (d.name for d in detectors), ",") sample_only output_dir
    @info "package versions"
    Pkg.status()

    @info "using fiducial in-band spectrum from cache as observed data"
    observed = ctx.fiducial_spectral_density

    @info "seeding RNG" rng_seed=seed
    Random.seed!(seed)

    final_save_state = false
    checkpoint_save_state = true

    @info "starting NUTS" n_adapts n_samples target_acceptance ad_backend=ad_backend_name sample_only checkpoint_every
    turing_model = build_turing_model(
        problem,
        C,
        ctx,
        hyperprior;
        track = true,
        observed = observed
    )
    conditioned = condition_turing_model(
        turing_model,
        fiducial,
        hyperprior,
        sample_only;
        order = order
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
            progress = progress, save_state = final_save_state, chain_type = VNChain,
            initial_params = initial_params
        )
    else
        sample(
            conditioned, nuts, MCMCThreads(), n_samples, num_chains;
            progress = progress, save_state = final_save_state, callback = callback,
            chain_type = VNChain, initial_params = initial_params
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
