# Headless, config-driven NUTS runner for the GWBackground importance-sampling model.
#
# This mirrors the sampling cells of notebooks/mcmc.jl but takes run-specific
# settings (catalog, detectors, fiducials, sampler, etc.) from a TOML config,
# parsed and validated via GWBackgroundInference.MCMCConfig. Hyperprior bounds, the
# cosmology family, and the population model are fixed here, matching the notebook.
#
# Run from the repository root, for example:
#   julia --project=scripts/run -t auto scripts/run_mcmc.jl config/mcmc/example.toml

module GWBackgroundRunMCMC

const _REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

using GWBackground
using GWBackground:
                 load_catalog,
                 average_mode,
                 AnalyticInclination,
                 CatalogInclination,
                 effective_psd,
                 ModifiedPropagation,
                 LambdaCDM,
                 Detector
using GWBackgroundImportanceModels:
                                 prepare_bns_madau_dickinson_model,
                                 bns_amplitude_scalings,
                                 bns_hyperprior,
                                 bns_hyperprior_amplitude_marginalized
using GWBackgroundInference:
                          gwbackground_importance_turing_model,
                          gwbackground_amplitude_marginalized_turing_model,
                          forward_model,
                          quadrature_grid,
                          reconstruct_amplitude,
                          rename_posterior_for_netcdf,
                          merge_into_posterior,
                          MCMCConfig,
                          load_config,
                          save_config,
                          posterior_params
using InferenceObjects: InferenceObjects
# `to_netcdf` lives in InferenceObjects' NCDatasets extension, which only
# activates when NCDatasets is loaded; it's an explicit dep of this project.
using NCDatasets: NCDatasets
using ADTypes: AutoForwardDiff, AutoEnzyme
using Enzyme
using AdvancedHMC: DenseEuclideanMetric
using Distributions: Uniform
using FlexiChains: VNChain, Parameter, @varname
using Turing
using Random
using Logging
using LinearAlgebra: BLAS
using Dates: now, format

# Fixed model selection (see notebooks/mcmc.jl): background cosmology `C` and GW
# propagation `P` are orthogonal axes.
const C = LambdaCDM
const P = ModifiedPropagation

# Inclination-averaging convention of the catalog. `nothing` derives it from the
# catalog's own `inclination` column via `GWBackground.average_mode`: an all-zero
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

# Hard-coded hyperprior bounds (matching notebooks/mcmc.jl). The NamedTuple holds
# distributions; `bns_hyperprior` declares the Turing `~` layout. The `R₀` entry is the
# nominal distribution its conditioning (`model | (; R₀ = …)`) pins against.
const HYPERPRIOR = (
    H0 = Uniform(20.0, 140.0),
    Ωm = Uniform(0.05, 0.95),
    Ξ₀ = Uniform(0.5, 5.0),
    Ξₙ = Uniform(0.3, 3.0),
    γ = Uniform(0.5, 10.0),
    κ = Uniform(0.05, 10.0),
    zpeak = Uniform(0.05, 10.0),
    R₀ = Uniform(10.0, 1000.0)
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
    name == "Enzyme" &&
        return AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
    throw(ArgumentError(
        "unsupported ad_backend $(repr(name)); supported: \"ForwardDiff\", \"Enzyme\"",
    ))
end

"""
Materialize the config's fiducial map as a `NamedTuple`.

This is the **full** point (sampled plus conditioned) at which `observed` is
synthesized, so it is built from the config's own keys rather than checked against a
model-declared order -- there is no longer such an order to check against. A key the
model reads but the config omits throws a `KeyError` on `Λ.name` at prepare time, before
NUTS starts. Keys are sorted for a deterministic `NamedTuple` type.
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

"""Keep only keys that the production prior model declares (drop e.g. leftover `w0`)."""
function _only_hyperprior_keys(nt::NamedTuple)
    names = Tuple(n for n in keys(nt) if haskey(HYPERPRIOR, n))
    return NamedTuple{names}(nt)
end

"""
Build the amplitude marginalization for a `likelihood = "amplitude_marginalized"` config.

Everything here is *live*: the prior distribution and the two scalings are objects, not
derived numbers, so nothing can go stale against the config it came from. The one array,
`grid`, is a quadrature scheme rather than a tabulation of the density.

The prior, fiducial, and scalings must reach both the model and `reconstruct_amplitude`
unchanged -- a mismatch is silent, because the sufficient statistics stay finite and
plausible whatever conditional you pair them with. The `grid` is only quadrature accuracy
(the reconstruction re-meshes from it internally), but building everything once here keeps
it that way by construction.
"""
function _amplitude_marginalization(cfg::MCMCConfig, prior::NamedTuple, fiducials)
    name = cfg.amplitude_parameter
    haskey(prior, name) || throw(ArgumentError(
        "amplitude_parameter $(repr(name)) is not in HYPERPRIOR; it needs a prior to be " *
        "marginalized under",
    ))
    scalings = bns_amplitude_scalings(name)
    amplitude_prior = prior[name]
    return (;
        name,
        fiducial = NamedTuple{(name,)}((fiducials[name],)),
        prior = amplitude_prior,
        scalings.amplitude_fn,
        scalings.merger_rate_fn,
        grid = quadrature_grid(amplitude_prior;
            num_nodes = cfg.amplitude_num_nodes,
            span_sigma = cfg.amplitude_prior_span_sigma)
    )
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
    detectors = Detector.(cfg.detectors)
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
    # S3: the prior declares every hyperparameter name; fixing one is conditioning
    # (`model | fixed`), so the chain carries exactly the sampled variables by
    # construction -- no complement computation, no subset validation. `R₀` is pinned at
    # its fiducial unless named in `sample_only`.
    sample_only = cfg.sample_only === nothing || isempty(cfg.sample_only) ? nothing :
                  cfg.sample_only
    sampleable = sample_only === nothing ?
                 Base.structdiff(HYPERPRIOR, (; R₀ = HYPERPRIOR.R₀)) : HYPERPRIOR
    prior = _restrict_prior(sampleable, cfg.sample_only)

    # Under the marginalized likelihood the amplitude parameter is neither a latent nor a
    # conditioned value: the model pins it internally to build the template. The config
    # already rejects it in `sample_only`, but `sample_only = nothing` means "sample
    # everything", so it is dropped from the prior here as well.
    amplitude = cfg.likelihood == "amplitude_marginalized" ?
                _amplitude_marginalization(cfg, HYPERPRIOR, fiducials) : nothing
    if amplitude === nothing
        model_prior = HYPERPRIOR
        prior_model = bns_hyperprior(model_prior)
        fixed = _only_hyperprior_keys(Base.structdiff(fiducials, prior))
    else
        prior = Base.structdiff(prior, amplitude.fiducial)
        model_prior = Base.structdiff(HYPERPRIOR, amplitude.fiducial)
        prior_model = bns_hyperprior_amplitude_marginalized(
            model_prior, Val(amplitude.name))
        fixed = _only_hyperprior_keys(
            Base.structdiff(Base.structdiff(fiducials, prior), amplitude.fiducial))
        @info "amplitude marginalization" parameter=amplitude.name fiducial=only(amplitude.fiducial) prior=amplitude.prior num_nodes=length(amplitude.grid) grid=extrema(amplitude.grid)
    end

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
    # The *saved* parameters, not the sampler's latents: under marginalization the chain
    # carries one parameter NUTS never proposed.
    saved_params = posterior_params(cfg)
    params_suffix = isempty(saved_params) ? "all" : join(saved_params, "-")
    base = "$(output_prefix)-$(config_stem)-$(params_suffix)-det=$(det_suffix)-seed$(cfg.seed)-$(timestamp)"
    output_nc = joinpath(output_dir, "$base.nc")
    output_toml = joinpath(output_dir, "$base.toml")

    adtype = _resolve_adtype(cfg.sampler.ad_backend)
    @info "starting NUTS" nadapts=cfg.sampler.nadapts nsamples=cfg.sampler.nsamples target_acceptance=cfg.sampler.target_acceptance ad_backend=cfg.sampler.ad_backend sampled=keys(prior) fixed=keys(fixed) nchains
    # No external spectrum to fit: synthesize `observed` at the fiducials. One
    # `resolved_average_mode` reaches both this call and the model that scores it.
    observed = forward_model(
        model, polarization_power, samples, fiducials;
        average_mode = resolved_average_mode).spectral_density
    unconditioned = if amplitude === nothing
        gwbackground_importance_turing_model(
            model,
            polarization_power,
            samples,
            prior_model,
            observed,
            frequencies,
            eff_psd,
            cfg.observation_time,
            resolved_average_mode
        )
    else
        gwbackground_amplitude_marginalized_turing_model(
            model,
            polarization_power,
            samples,
            prior_model,
            observed,
            frequencies,
            eff_psd,
            cfg.observation_time,
            resolved_average_mode,
            amplitude.fiducial,
            amplitude.amplitude_fn,
            amplitude.prior,
            amplitude.grid
        )
    end
    turing_model = unconditioned | fixed
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

    if amplitude !== nothing
        # Post-processing, against the saved chain alone: no catalog, no (nfreq, nsamples)
        # contraction, O(length(grid)) per draw. The RNG is seeded distinctly from the
        # sampler because these are fresh draws from the conditional, not a deterministic
        # function of the chain.
        @info "reconstructing marginalized parameter" parameter = amplitude.name
        reconstruction = reconstruct_amplitude(
            Random.Xoshiro(cfg.seed + 1),
            Array(chain[Parameter(@varname(amplitude_mle))]),
            Array(chain[Parameter(@varname(template_optimal_snr))]),
            Array(chain[Parameter(@varname(template_merger_rate))]);
            amplitude.amplitude_fn,
            amplitude.merger_rate_fn,
            prior = amplitude.prior,
            fiducial = only(amplitude.fiducial),
            grid = amplitude.grid
        )
        min_nodes = minimum(reconstruction.quadrature_effective_nodes)
        # "Exact up to quadrature error" only holds if the grid resolves the conditional
        # posterior. A handful of nodes under the bump still yields a finite, plausible
        # log evidence, so this is the only symptom there is.
        min_nodes < 30 &&
            @warn "quadrature grid may not resolve the conditional posterior; increase amplitude_num_nodes" min_effective_nodes=min_nodes threshold=30
        chain = merge_into_posterior(
            chain,
            merge(
                NamedTuple{(amplitude.name,)}((reconstruction.parameter,)),
                (;
                    reconstruction.total_merger_rate,
                    reconstruction.quadrature_effective_nodes
                )
            )
        )
        @info "reconstruction done" min_effective_nodes=min_nodes reconstructed=extrema(reconstruction.parameter)
    end

    @info "writing chain to netCDF" path = output_nc
    # Unicode hyperparameter names become ASCII at the file boundary only, so this netCDF
    # and a Python `astrogwb` one carry the same variable names.
    idata = InferenceObjects.convert_to_inference_data(rename_posterior_for_netcdf(chain))
    InferenceObjects.to_netcdf(idata, output_nc)
    @info "writing run config to TOML" path = output_toml
    save_config(cfg, output_toml)
    @info "done" output_nc output_toml
    return output_nc
end

end # module GWBackgroundRunMCMC

function (@main)(args::Vector{String})
    length(args) == 1 || throw(ArgumentError("usage: run_mcmc.jl <config.toml>"))
    GWBackgroundRunMCMC.run_mcmc(args[1])
    return 0
end
