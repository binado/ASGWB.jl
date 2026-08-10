# Headless, config-driven NUTS runner for the AstroSGWB importance-sampling model.
#
# This mirrors the sampling cells of notebooks/mcmc.jl but takes run-specific
# settings (catalog, detectors, fiducials, sampler, etc.) from a TOML config,
# parsed and validated via AstroSGWBInference.MCMCConfig. Hyperprior bounds, the
# cosmology family, and the population model are fixed here, matching the notebook.
#
# Run from the repository root, for example:
#   julia --project=scripts/run -t auto scripts/run_mcmc.jl config/mcmc/example.toml

module AstroSGWBRunMCMC

const _REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

using AstroSGWB
using AstroSGWB:
                 load_catalog,
                 average_mode,
                 AnalyticInclination,
                 CatalogInclination,
                 effective_psd,
                 ModifiedPropagation,
                 W0CDM,
                 Detector
using AstroSGWBImportanceModels:
                                 prepare_bns_madau_dickinson_model
using AstroSGWBInference:
                          build_turing_model,
                          MCMCConfig,
                          load_config,
                          save_config
using InferenceObjects: InferenceObjects
# `to_netcdf` lives in InferenceObjects' NCDatasets extension, which only
# activates when NCDatasets is loaded; it's an explicit dep of this project.
using NCDatasets: NCDatasets
using ADTypes: AutoForwardDiff
using AdvancedHMC: DenseEuclideanMetric
using Distributions: Uniform
using FlexiChains: VNChain
using Turing
using Random
using Logging
using LinearAlgebra: BLAS
using Dates: now, format

# Fixed model selection (see notebooks/mcmc.jl): background cosmology `C` and GW
# propagation `P` are now orthogonal axes.
const C = W0CDM
const P = ModifiedPropagation

# Inclination-averaging convention of the catalog. `nothing` derives it from the
# catalog's own `inclination` column via `AstroSGWB.average_mode`: an all-zero
# column means face-on waveforms and the analytic 2/5 average, anything else
# means the catalog already averages over ι. Set this to `AnalyticInclination()`
# or `CatalogInclination()` to override the derived value.
#
# A catalog with no `inclination` column at all falls back to
# `AnalyticInclination()`. Every gwmock-pop catalog emits the column, so that
# fallback only bites on hand-built or pre-gwmock files -- the resolved value is
# logged below so a surprising fallback is visible in the run log.
const AVERAGE_MODE = nothing

# Analysis band (Hz). The catalog carries no band information: `frequencies` and the
# rows of `polarization_power` are sliced with this cut before the effective PSD is computed, so
# every bin handed to the model is scored. Matches the generator band of the production
# catalog.
const MINIMUM_FREQUENCY = 2.0
const MAXIMUM_FREQUENCY = 4096.0

# Hard-coded hyperprior bounds (matching notebooks/mcmc.jl).
const HYPERPRIOR = (
    H0 = Uniform(20.0, 140.0),
    Ωm = Uniform(0.05, 0.95),
    w0 = Uniform(-3, 1),
    Ξ₀ = Uniform(0.5, 5.0),
    Ξₙ = Uniform(0.3, 3.0),
    γ = Uniform(0.5, 10.0),
    κ = Uniform(0.05, 10.0),
    zpeak = Uniform(0.05, 10.0)
)

# --------------------------------------------------------------------------
# Materialization helpers
# --------------------------------------------------------------------------

function _resolve_catalog_path(catalog_path::AbstractString, base::AbstractString)
    return isabspath(catalog_path) ? String(catalog_path) :
           normpath(joinpath(base, catalog_path))
end

function _resolve_adtype(name::AbstractString)
    name == "ForwardDiff" && return AutoForwardDiff()
    throw(ArgumentError("unsupported ad_backend $(repr(name)) (use \"ForwardDiff\")"))
end

"""
Materialize the config's fiducial map as a `NamedTuple`.

This is the **full** point (`prior ∪ constants`) at which `observed` is synthesized, so
it is built from the config's own keys rather than checked against a model-declared
order -- there is no longer such an order to check against. A key the model reads but
the config omits throws a `KeyError` on `Λ.name` at prepare time, before NUTS starts.
Keys are sorted for a deterministic `NamedTuple` type.
"""
function _fiducials_namedtuple(cfg::MCMCConfig)
    names = Tuple(sort!(collect(keys(cfg.fiducials)); by = string))
    return NamedTuple{names}(Tuple(cfg.fiducials[sym] for sym in names))
end

"""
Restrict `prior` to the hyperparameters named in `sample_only`.

`NamedTuple{names}(prior)` alone would report an unknown symbol as
`type NamedTuple has no field :Xyz`, which does not say where the name came from.
"""
function _restrict_prior(prior::NamedTuple, sample_only)
    (sample_only === nothing || isempty(sample_only)) && return prior
    names = Tuple(Symbol.(sample_only))
    for sym in names
        haskey(prior, sym) || throw(ArgumentError(
            "unknown hyperparameter $(repr(sym)) in sample_only; " *
            "expected one of $(keys(prior))",
        ))
    end
    return NamedTuple{names}(prior)
end

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

function run_mcmc(config_file::String)
    BLAS.set_num_threads(1)
    num_threads = Base.Threads.nthreads()

    @info "loading config" path = config_file
    cfg = load_config(config_file)

    catalog_path = _resolve_catalog_path(cfg.catalog_path, _REPO_ROOT)
    detectors = [Detector(n) for n in cfg.detectors]
    output_dir = joinpath(_REPO_ROOT, cfg.output_dir)
    output_prefix = cfg.output_prefix

    cfg_nchains = cfg.sampler.nchains
    nchains = cfg_nchains > 0 ? cfg_nchains : num_threads
    nchains == num_threads || throw(ArgumentError(
        "sampler.nchains must equal Base.Threads.nthreads() for MCMCThreads() " *
        "(got nchains=$nchains, nthreads()=$num_threads); " *
        "set nchains = 0 or match -t / SLURM_CPUS_PER_TASK",
    ))

    @info "model" cosmology=string(C) propagation=string(P) sampleable=keys(HYPERPRIOR)
    fiducials = _fiducials_namedtuple(cfg)
    # S3: the prior declares what is sampled, `constants` what is held fixed. The chain
    # then carries exactly the sampled variables by construction -- no DynamicPPL
    # conditioning, no complement computation, no subset validation.
    prior = _restrict_prior(HYPERPRIOR, cfg.sample_only)
    constants = Base.structdiff(fiducials, prior)
    sample_only = keys(prior) == keys(HYPERPRIOR) ? nothing : keys(prior)

    @info "seeding RNG" seed = cfg.seed
    Random.seed!(cfg.seed)

    @info "loading catalog" catalog_path detectors=join((d.name for d in detectors), ",")
    catalog = load_catalog(catalog_path)
    resolved_average_mode = AVERAGE_MODE === nothing ? average_mode(catalog) : AVERAGE_MODE
    @info "average mode" mode=string(resolved_average_mode) derived=(AVERAGE_MODE===nothing) has_inclination_column=haskey(
        catalog.samples, :inclination)
    samples = catalog.samples
    # Re-reference the stored EM-distance polarization power to the fiducial GW distance, matching the
    # `+2 log Ξ_fid` term the prepared model's log-weights carry. No-op under Ξ₀ = 1.
    apply_gw_distance_correction!(catalog, propagation(P, fiducials))
    # Band selection is the caller's job: restrict to the analysis band before
    # computing the effective PSD, so every bin handed to the model is scored.
    band = (catalog.frequencies .>= MINIMUM_FREQUENCY) .&
           (catalog.frequencies .<= MAXIMUM_FREQUENCY)
    polarization_power = catalog.polarization_power[band, :]
    frequencies = catalog.frequencies[band]
    model = prepare_bns_madau_dickinson_model(samples, fiducials, C, P)
    eff_psd = effective_psd(frequencies, detectors)
    @info "catalog loaded" n_frequency_bins=length(frequencies) n_proposal_samples=length(
        samples.redshift,
    )

    mkpath(output_dir)
    timestamp = format(now(), "yyyymmdd-HHMMSS")
    config_stem = splitext(basename(config_file))[1]
    det_suffix = join((d.name for d in detectors), ",")
    params_suffix = sample_only === nothing ? "all" : join(sample_only, "-")
    base = "$(output_prefix)-$(config_stem)-$(params_suffix)-det=$(det_suffix)-seed$(cfg.seed)-$(timestamp)"
    output_nc = joinpath(output_dir, "$base.nc")
    output_toml = joinpath(output_dir, "$base.toml")

    adtype = _resolve_adtype(cfg.sampler.ad_backend)
    @info "starting NUTS" nadapts=cfg.sampler.nadapts nsamples=cfg.sampler.nsamples target_acceptance=cfg.sampler.target_acceptance ad_backend=cfg.sampler.ad_backend sampled=keys(prior) fixed=keys(constants) nchains
    turing_model = build_turing_model(
        model,
        polarization_power,
        samples,
        fiducials,
        frequencies,
        eff_psd,
        cfg.observation_time,
        prior;
        constants = constants,
        track = true,
        average_mode = resolved_average_mode
    )
    nuts = Turing.NUTS(
        cfg.sampler.nadapts,
        cfg.sampler.target_acceptance;
        metricT = DenseEuclideanMetric,
        adtype = adtype
    )
    initial_params = fill(InitFromPrior(), nchains)
    chain = sample(
        turing_model,
        nuts,
        MCMCThreads(),
        cfg.sampler.nsamples,
        nchains;
        progress = true,
        save_state = false,
        chain_type = VNChain,
        initial_params = initial_params
    )
    @info "NUTS finished" chain_size = size(chain)

    @info "writing chain to netCDF" path = output_nc
    idata = InferenceObjects.convert_to_inference_data(chain)
    InferenceObjects.to_netcdf(idata, output_nc)
    @info "writing run config to TOML" path = output_toml
    save_config(cfg, output_toml)
    @info "done" output_nc output_toml
    return output_nc
end

end # module AstroSGWBRunMCMC

function (@main)(args::Vector{String})
    length(args) == 1 || throw(ArgumentError("usage: run_mcmc.jl <config.toml>"))
    AstroSGWBRunMCMC.run_mcmc(args[1])
    return 0
end
