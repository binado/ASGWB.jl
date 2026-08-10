### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ 8f3a2c1d-4e5b-4a6c-9d0e-1f2a3b4c5d6e
md"""
# Cosmological parameter inference with the astrophysical GWB

In this notebook, we perform Bayesian inference on the cosmological and astrophysical parameters that play into the gravitational-wave background of stellar-mass compact binary coalescences (CBCs) such as neutron stars or black holes.

To properly run the notebook, you must specify a path to a catalog HDF5 file containing the intrinsic parameter samples of the CBC population as well as the associated waveforms.
"""

# ╔═╡ 9a4b3c2d-5f6e-4b7a-8c1d-2e3f4a5b6c7d
begin
    num_threads = Base.Threads.nthreads()
    print(num_threads)
end

# ╔═╡ a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d
md"""
## Importance model

Inference requires an importance adapter, parametrized by a vector ``\Lambda``, which
characterizes the distribution of the intrinsic parameters ``p(\theta | \Lambda)``.

The canonical adapter is `BNSMadauDickinsonImportanceModel{C, P}` from
`AstroSGWBImportanceModels`. The entire inference contract is that the prepared model is
**callable**:

- **`model(Λ, samples) -> (rate, log_weights)`** — inlines the redshift log-ratio, importance weights, and rate normalization. For this BNS population the Λ-independent mass/spin/tidal priors cancel exactly, so only the redshift + distance/propagation terms survive.

Hyperparameter *names* are declared by the hyperprior below, not by the model: a name the
model reads but the prior omits throws a `KeyError` on `Λ.name` at the first evaluation.
"""

# ╔═╡ b2c3d4e5-f6a7-4b8c-9d0e-1f2a3b4c5d6e
md"""
## Configuration

Edit runtime settings here: `catalog_path`, detectors, observation time, merger rate, fiducials, `hyperprior_dists` / `hyperprior`, sampler (`nsamples`, `nadapts`, `ad_backend`, `nchains`), output paths, and `DEBUG`.
"""

# ╔═╡ c3d4e5f6-a7b8-4c9d-0e1f-2a3b4c5d6e7f
begin
    DEBUG = false
    @info debug = DEBUG

    function resolve_adtype(name::AbstractString)
        if name == "ForwardDiff"
            return ADTypes.AutoForwardDiff()
        else
            throw(ArgumentError(
                "this notebook supports only ad_backend = \"ForwardDiff\"; got $(repr(name))",
            ))
        end
    end

    _repo_root = normpath(joinpath(@__DIR__, ".."))

    catalog_path = joinpath(_repo_root, "catalog.h5")
    detnames = [:S1, :R1, :C1]
    detectors = map(Detector ∘ string, detnames)
    sample_only = (:H0,)

    seed = 42
    @info "seeding RNG" rng_seed = seed
    Random.seed!(seed)

    local_merger_rate = 161.0 # Matches COBA simulations
    observation_time = 1.0

    # Analysis band (Hz). The catalog carries no band information: slice
    # `frequencies` and the rows of `polarization_power` with this cut before computing the
    # effective PSD. Matches the generator band of the production catalog.
    minimum_frequency = 2.0
    maximum_frequency = 4096.0

    output_dir = joinpath(_repo_root, "chains")
    output_prefix = "chains"

    sampler = (
        nsamples = 3000,
        nadapts = 3000,
        target_acceptance = 0.9,
        ad_backend = "ForwardDiff",
        nchains = 0
    )

    cosmology_parameters = (;
        H0 = 67.66,
        Ωm = 0.3096,
        w0 = -1,
        Ξ₀ = 1.0,
        Ξₙ = 1.91
    )
    fiducials = (;
        cosmology_parameters...,
        γ = 2.7,
        κ = 3.0,
        zpeak = 2.0,
        # S7: the local merger rate (Gpc^-3 yr^-1) is an ordinary hyperparameter read as
        # `Λ.R₀`, not a prepare-time keyword. The MCMC cell pins it at this fiducial by
        # conditioning; name it in `sample_only` to sample it.
        R₀ = local_merger_rate
    )

    # Edit hyperprior bounds here (order: cosmology, then population). The prior declares
    # every name, sampled or pinned; the `R₀` entry is the nominal distribution its
    # conditioning pins against.
    hyperprior_dists = (
        H0 = Uniform(20.0, 140.0),
        Ωm = Uniform(0.05, 0.95),
        w0 = Uniform(-3, 1),
        Ξ₀ = Uniform(0.5, 5.0),
        Ξₙ = Uniform(0.3, 3.0),
        γ = Uniform(0.5, 10.0),
        κ = Uniform(0.05, 10.0),
        zpeak = Uniform(0.05, 10.0),
        R₀ = Uniform(10.0, 1000.0)
    )
    hyperprior = hyperprior_dists

    # Defining cosmology and propagation. Background expansion `C` and GW propagation `P`
    # are orthogonal axes (use `GR` for standard propagation).
    C = W0CDM
    P = ModifiedPropagation

    nchains = sampler.nchains > 0 ? sampler.nchains : num_threads
end

# ╔═╡ 3d7e6f5a-8c9b-4e0d-1f4a-5b6c7d8e9f0a
begin
    if nchains != num_threads
        @warn "nchains differs from Base.Threads.nthreads()" nchains num_threads
    end

    @info "loading catalog" catalog_path detectors = join((d.name for d in detectors), ",")
    catalog = load_catalog(catalog_path)

    # Inclination-averaging convention derived from the catalog's `inclination`
    # column: all-zero means face-on waveforms (analytic 2/5 average), anything
    # else means the catalog already averages over ι. A catalog with no such
    # column falls back to `AnalyticInclination()`. Replace with an explicit
    # `AnalyticInclination()` / `CatalogInclination()` to override.
    resolved_average_mode = average_mode(catalog)
    @info "average mode" mode=string(resolved_average_mode) has_inclination_column=haskey(
        catalog.samples, :inclination)

    samples = catalog.samples

    # Re-reference the stored EM-distance polarization power to the fiducial GW distance, matching the
    # `+2 log Ξ_fid` term the prepared model's log-weights carry. No-op under Ξ₀ = 1.
    # The out-of-place form is deliberate: Pluto re-runs cells reactively and the
    # correction is not idempotent, so mutating `catalog.polarization_power` here would compound to
    # Ξ⁻⁴, Ξ⁻⁶, ... on every re-execution. Downstream cells use `polarization_power`, not
    # `catalog.polarization_power`.
    polarization_power = apply_gw_distance_correction(
        catalog.polarization_power, catalog.samples.redshift, propagation(P, fiducials))

    # Band selection is the caller's job: restrict to the analysis band before
    # computing the effective PSD, so every bin handed to the model is scored.
    band = (catalog.frequencies .>= minimum_frequency) .&
           (catalog.frequencies .<= maximum_frequency)
    polarization_power = polarization_power[band, :]
    frequencies = catalog.frequencies[band]

    model = prepare_bns_madau_dickinson_model(samples, fiducials, C, P)
    eff_psd = effective_psd(frequencies, detectors)
    # S2: the prior declares the hyperparameter names; there is no model to ask.
    order = keys(hyperprior_dists)
    @info order
    sample_only_tup = sample_only === nothing ? nothing : Tuple(sample_only)

    @info "catalog loaded" n_frequency_bins=length(frequencies) n_proposal_samples=length(
        samples.redshift,
    )

    mkpath(output_dir)
    timestamp = format(now(), "yyyymmdd-HHMMSS")
    det_suffix = join((d.name for d in detectors), ",")
    params_suffix = sample_only === nothing ? "all" : join(sample_only, "-")
    base = "$(output_prefix)-$(params_suffix)-det=$(det_suffix)-seed$(seed)-$(timestamp)"
    output_nc = joinpath(output_dir, "$base.nc")
    output_toml = joinpath(output_dir, "$base.toml")

    # Reproducible record of this run's settings, dumped on a successful run.
    run_config = MCMCConfig(
        2,
        catalog_path,
        string.(detnames),
        seed,
        observation_time,
        SamplerConfig(
            sampler.nsamples,
            sampler.nadapts,
            sampler.target_acceptance,
            sampler.ad_backend,
            sampler.nchains
        ),
        Dict{Symbol, Float64}(k => Float64(v) for (k, v) in pairs(fiducials)),
        sample_only_tup === nothing ? nothing : collect(Symbol, sample_only_tup),
        output_dir,
        output_prefix
    )

    nothing
end

# ╔═╡ c2627b5e-b9f4-4535-b0c3-69ce8b2a696c
md"""
## Visualizing ``\Omega_{\mathrm{GW}}``

In the cells below, we plot ``\Omega_{\mathrm{GW}}(f)`` as a function of the frequency ``f`` for the fiducial values of the parameters ``\Lambda``.
"""

# ╔═╡ d4e5f6a7-b8c9-4d0e-1f2a-3b4c5d6e7f8a
function plot_fiducial_omega_gw(
        model, polarization_power, samples, fiducials, frequencies, eff_psd, observation_time)
    forward = forward_model(model, polarization_power, samples, fiducials)
    rate0, Sh0 = forward.rate, forward.spectral_density
    f = frequencies
    df = frequency_bin_width(f)
    snr = spectral_snr(
        Sh0,
        eff_psd,
        year_to_second(observation_time),
        df
    )

    Ωgw_plot = Ωgw(Sh0, f, fiducials.H0)
    mask = Ωgw_plot .> 0.0
    fm = f[mask]
    Ωgw_pos = Ωgw_plot[mask]
    fig = Figure(size = (900, 450))
    ax = Axis(
        fig[1, 1];
        xlabel = L"$f~\mathrm{(Hz)}$",
        ylabel = L"$\Omega_{\mathrm{GW}}(f)$",
        xscale = log10,
        yscale = log10,
        limits = (nothing, nothing, 1e-15, nothing)
    )
    if !isempty(Ωgw_pos)
        label = @sprintf "SNR = %.1f" snr
        lines!(ax, fm, Ωgw_pos; label = label)
        axislegend(ax; position = :rt)
    end
    return fig
end

# ╔═╡ 5f9a8b7c-0e1d-4a2f-3b6c-7d8e9f0a1b2c
plot_fiducial_omega_gw(
    model, polarization_power, samples, fiducials, frequencies, eff_psd, observation_time)

# ╔═╡ ccf43d43-7f31-41e9-85db-12842561973c
md"""
## Running the MCMC
"""

# ╔═╡ 7b1c0d9e-2f3a-4c4b-5d6e-7f8a9b0c1d2e
begin
    initial_params = fill(InitFromPrior(), nchains)
    adtype = resolve_adtype(sampler.ad_backend)

    @info "starting NUTS" nadapts=sampler.nadapts nsamples=sampler.nsamples target_acceptance=sampler.target_acceptance ad_backend=sampler.ad_backend sample_only=sample_only_tup
    # S3: the prior declares every hyperparameter name; fixing one is conditioning
    # (`model | fixed`), so the chain carries exactly the sampled variables by
    # construction. `R₀` is pinned at its fiducial unless named in `sample_only`.
    sampled_prior = sample_only_tup === nothing ?
                    Base.structdiff(hyperprior, (; R₀ = hyperprior.R₀)) :
                    NamedTuple{sample_only_tup}(hyperprior)
    fixed = Base.structdiff(fiducials, sampled_prior)
    # No external spectrum to fit: synthesize `observed` at the fiducials. One
    # `resolved_average_mode` reaches both this call and the model that scores it.
    observed = forward_model(
        model, polarization_power, samples, fiducials;
        average_mode = resolved_average_mode).spectral_density
    turing_model = astrosgwb_importance_turing_model(
        false,
        resolved_average_mode,
        model,
        polarization_power,
        samples,
        frequencies,
        eff_psd,
        observation_time,
        hyperprior,
        observed
    ) | fixed
    nuts = Turing.NUTS(
        sampler.nadapts,
        sampler.target_acceptance;
        metricT = AdvancedHMC.DenseEuclideanMetric,
        adtype = adtype
    )
    chain = if DEBUG
        @info "MCMC skipped for debugging"
        nothing
    else
        sampled_chain = sample(
            turing_model,
            nuts,
            MCMCThreads(),
            sampler.nsamples,
            nchains;
            progress = true,
            save_state = false,
            chain_type = VNChain,
            initial_params = initial_params
        )
        @info "NUTS finished" chain_size = size(sampled_chain)
        sampled_chain
    end
    chain
end

# ╔═╡ 8c2d1e0f-3a4b-4c5d-6e7f-8a9b0c1d2e3f
md"""
## Saving the chains to an output file
"""

# ╔═╡ 9d3e2f1a-4b5c-4d6e-7f8a-9b0c1d2e3f4a
begin
    if chain !== nothing
        @info "writing chain to netCDF" path = output_nc
        idata = InferenceObjects.convert_to_inference_data(chain)
        InferenceObjects.to_netcdf(idata, output_nc)
        @info "writing run config to TOML" path = output_toml
        save_config(run_config, output_toml)
        @info "done"
    else
        @info "skipping netCDF save (no chain; DEBUG mode)"
    end
end

# ╔═╡ 0e4f3a2b-5c6d-4e7f-8a9b-0c1d2e3f4a5b
md"""
## Diagnostic plots
"""

# ╔═╡ 1f5a4b3c-6d7e-4f8a-9b0c-1d2e3f4a5b6c
summarystats(chain)

# ╔═╡ 2a6b5c4d-7e8f-4a9b-0c1d-2e3f4a5b6c7d
FlexiChains.mtraceplot(chain)

# ╔═╡ 3b7c6d5e-8f9a-4b0c-1d2e-3f4a5b6c7d8e
begin
    n_draws = size(chain, 1)
    autocor_maxlag = min(100, max(1, n_draws - 1))
    FlexiChains.mautocorplot(chain; lags = 1:autocor_maxlag)
end

# ╔═╡ 4c8d7e6f-9a0b-4c1d-2e3f-4a5b6c7d8e9f
begin
    chain_params = FlexiChains.parameters(chain)
    fig = if length(chain_params) >= 2
        pairplot(chain)
    else
        Makie.density(chain)
    end
    fig
end

# ╔═╡ 2c6d5e4f-7b8a-4d9c-0e3f-4a5b6c7d8e9f
nothing

# ╔═╡ 1b5c4d3e-6a7f-4c8b-9d2e-3f4a5b6c7d8e
begin
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
    using AstroSGWB
    using AstroSGWB:
                     Detector,
                     effective_psd,
                     load_catalog,
                     average_mode,
                     AnalyticInclination,
                     CatalogInclination,
                     W0CDM,
                     ModifiedPropagation,
                     spectral_density,
                     year_to_second,
                     Ωgw
    using AstroSGWBImportanceModels:
                                     prepare_bns_madau_dickinson_model
    using AstroSGWBInference: astrosgwb_importance_turing_model, forward_model
    using AstroSGWBInference: MCMCConfig, SamplerConfig, save_config
    using Distributions: Uniform
    using InferenceObjects: InferenceObjects
    # `to_netcdf` lives in InferenceObjects' NCDatasets extension, which only
    # activates when NCDatasets is loaded.
    using NCDatasets: NCDatasets
    using Turing
    using AdvancedHMC
    using ADTypes
    using Random
    using Logging
    using FlexiChains
    using FlexiChains: VNChain
    using PairPlots
    using CairoMakie
    using LaTeXStrings
    using Dates: now, format
    using Printf
    using LinearAlgebra: BLAS
    BLAS.set_num_threads(1)
end

# ╔═╡ Cell order:
# ╠═8f3a2c1d-4e5b-4a6c-9d0e-1f2a3b4c5d6e
# ╠═9a4b3c2d-5f6e-4b7a-8c1d-2e3f4a5b6c7d
# ╠═1b5c4d3e-6a7f-4c8b-9d2e-3f4a5b6c7d8e
# ╠═a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d
# ╠═2c6d5e4f-7b8a-4d9c-0e3f-4a5b6c7d8e9f
# ╟─b2c3d4e5-f6a7-4b8c-9d0e-1f2a3b4c5d6e
# ╠═c3d4e5f6-a7b8-4c9d-0e1f-2a3b4c5d6e7f
# ╠═3d7e6f5a-8c9b-4e0d-1f4a-5b6c7d8e9f0a
# ╠═c2627b5e-b9f4-4535-b0c3-69ce8b2a696c
# ╠═d4e5f6a7-b8c9-4d0e-1f2a-3b4c5d6e7f8a
# ╠═5f9a8b7c-0e1d-4a2f-3b6c-7d8e9f0a1b2c
# ╠═ccf43d43-7f31-41e9-85db-12842561973c
# ╠═7b1c0d9e-2f3a-4c4b-5d6e-7f8a9b0c1d2e
# ╠═8c2d1e0f-3a4b-4c5d-6e7f-8a9b0c1d2e3f
# ╠═9d3e2f1a-4b5c-4d6e-7f8a-9b0c1d2e3f4a
# ╟─0e4f3a2b-5c6d-4e7f-8a9b-0c1d2e3f4a5b
# ╠═1f5a4b3c-6d7e-4f8a-9b0c-1d2e3f4a5b6c
# ╠═2a6b5c4d-7e8f-4a9b-0c1d-2e3f4a5b6c7d
# ╠═3b7c6d5e-8f9a-4b0c-1d2e-3f4a5b6c7d8e
# ╠═4c8d7e6f-9a0b-4c1d-2e3f-4a5b6c7d8e9f
