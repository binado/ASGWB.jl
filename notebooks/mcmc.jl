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
`AstroSGWBImportanceModels`. It implements the two-method inference contract:

- **`hyperparameters(model)`** — declares the joint hyperparameter names: cosmology (`C`), propagation (`P`), and the Madau–Dickinson redshift parameters `:γ`, `:κ`, `:zpeak`.
- **`merger_rate_and_log_weights(model, Λ, samples)`** — inlines the redshift log-ratio, importance weights, and rate normalization. For this BNS population the Λ-independent mass/spin/tidal priors cancel exactly, so only the redshift + distance/propagation terms survive.

`bns_samples_from_catalog` keeps only the catalog columns the weight loop reads (`redshift` and `luminosity_distance`); when the catalog omits `luminosity_distance` it is generated once from redshift at the fiducial cosmology, so the `samples` NamedTuple stays the single source of truth for the EM distance.
"""

# ╔═╡ b2c3d4e5-f6a7-4b8c-9d0e-1f2a3b4c5d6e
md"""
## Configuration

Edit runtime settings here: `catalog_path`, detectors, observation time, merger rate, fiducials, `hyperprior_dists` / `hyperprior`, sampler (`nsamples`, `nadapts`, `ad_backend`, `nchains`), output paths, `chain_input_jld2`, and `DEBUG`.
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
        zpeak = 2.0
    )

    # Edit hyperprior bounds here (order: cosmology, then population).
    hyperprior_dists = (
        H0 = Uniform(20.0, 140.0),
        Ωm = Uniform(0.05, 0.95),
        w0 = Uniform(-3, 1),
        Ξ₀ = Uniform(0.5, 5.0),
        Ξₙ = Uniform(0.3, 3.0),
        γ = Uniform(0.5, 10.0),
        κ = Uniform(0.05, 10.0),
        zpeak = Uniform(0.05, 10.0)
    )
    hyperprior = product_distribution(hyperprior_dists)

    # Defining cosmology and propagation. Background expansion `C` and GW propagation `P`
    # are orthogonal axes (use `GR` for standard propagation).
    C = W0CDM
    P = ModifiedPropagation

    chain_input_jld2 = nothing

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

    samples = bns_samples_from_catalog(catalog.samples, C, fiducials)

    # Re-reference the stored EM-distance fluxes to the fiducial GW distance, matching the
    # `+2 log Ξ_fid` term the prepared model's log-weights carry. No-op under Ξ₀ = 1.
    # The out-of-place form is deliberate: Pluto re-runs cells reactively and the
    # correction is not idempotent, so mutating `catalog.fluxes` here would compound to
    # Ξ⁻⁴, Ξ⁻⁶, ... on every re-execution. Downstream cells use `fluxes`, not
    # `catalog.fluxes`.
    fluxes = apply_gw_distance_correction(
        catalog.fluxes, catalog.samples.redshift, propagation(P, fiducials))

    model = prepare_bns_madau_dickinson_model(
        samples,
        fiducials,
        C,
        P;
        observation_time = observation_time,
        local_merger_rate = local_merger_rate
    )
    observation = build_observation_context(
        catalog.frequencies, detectors, catalog.in_band_mask, observation_time)
    order = hyperparameters(model)
    @info order
    sample_only_tup = sample_only === nothing ? nothing : Tuple(sample_only)

    @info "catalog loaded" n_frequency_bins=length(observation.frequencies) n_proposal_samples=length(
        samples.redshift,
    )

    mkpath(output_dir)
    timestamp = format(now(), "yyyymmdd-HHMMSS")
    det_suffix = join((d.name for d in detectors), ",")
    params_suffix = sample_only === nothing ? "all" : join(sample_only, "-")
    base = "$(output_prefix)-$(params_suffix)-det=$(det_suffix)-seed$(seed)-$(timestamp)"
    output_jld2 = joinpath(output_dir, "$base.jld2")
    output_toml = joinpath(output_dir, "$base.toml")

    # Reproducible record of this run's settings, dumped on a successful run.
    run_config = MCMCConfig(
        1,
        catalog_path,
        string.(detnames),
        seed,
        observation_time,
        local_merger_rate,
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
function plot_fiducial_omega_gw(model, fluxes, samples, fiducials, observation)
    rate0, log_weights0 = merger_rate_and_log_weights(model, fiducials, samples)
    Sh0 = spectral_density(fluxes, rate0; weights = exp.(log_weights0))
    f = observation.frequencies
    df = frequency_bin_width(f)
    snr = spectral_snr(
        Sh0,
        observation.effective_psd,
        year_to_second(observation.observation_time),
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
plot_fiducial_omega_gw(model, fluxes, samples, fiducials, observation)

# ╔═╡ ccf43d43-7f31-41e9-85db-12842561973c
md"""
## Running the MCMC
"""

# ╔═╡ 7b1c0d9e-2f3a-4c4b-5d6e-7f8a9b0c1d2e
begin
    if chain_input_jld2 !== nothing
        chain_path = isabspath(chain_input_jld2) ? String(chain_input_jld2) :
                     normpath(joinpath(_repo_root, chain_input_jld2))
        isfile(chain_path) ||
            throw(ArgumentError("JLD2 chain file not found: $(repr(chain_path))"))
        @info "loading chain from JLD2" path = chain_path
        chain = load(chain_path)["chain"]
        @info "chain loaded" chain_size = size(chain)
    else
        initial_params = fill(InitFromPrior(), nchains)
        adtype = resolve_adtype(sampler.ad_backend)

        @info "starting NUTS" nadapts=sampler.nadapts nsamples=sampler.nsamples target_acceptance=sampler.target_acceptance ad_backend=sampler.ad_backend sample_only=sample_only_tup
        turing_model = build_turing_model(
            model,
            fluxes,
            samples,
            fiducials,
            observation,
            hyperprior;
            track = false,
            average_mode = resolved_average_mode
        )
        conditioned = condition_turing_model(
            turing_model,
            fiducials,
            hyperprior,
            sample_only_tup
        )
        nuts = Turing.NUTS(
            sampler.nadapts,
            sampler.target_acceptance;
            metricT = AdvancedHMC.DenseEuclideanMetric,
            adtype = adtype
        )
        if DEBUG
            @info "MCMC skipped for debugging"
            chain = nothing
        else
            chain = sample(
                conditioned,
                nuts,
                MCMCThreads(),
                sampler.nsamples,
                nchains;
                progress = true,
                save_state = false,
                chain_type = VNChain,
                initial_params = initial_params
            )
            @info "NUTS finished" chain_size = size(chain)
        end
    end
    chain
end

# ╔═╡ 8c2d1e0f-3a4b-4c5d-6e7f-8a9b0c1d2e3f
md"""
## Saving the chains to an output file
"""

# ╔═╡ 9d3e2f1a-4b5c-4d6e-7f8a-9b0c1d2e3f4a
begin
    if chain_input_jld2 === nothing && chain != nothing
        @info "writing chain to JLD2" path = output_jld2
        atomic_save_chain(output_jld2, chain)
        @info "writing run config to TOML" path = output_toml
        save_config(run_config, output_toml)
        @info "done"
    else
        @info "skipping JLD2 save (chain was loaded from disk)"
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
                     build_observation_context,
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
                                     bns_samples_from_catalog,
                                     prepare_bns_madau_dickinson_model
    using AstroSGWBInference: build_turing_model, condition_turing_model
    using AstroSGWBInference: hyperparameters, merger_rate_and_log_weights
    using AstroSGWBInference: MCMCConfig, SamplerConfig, save_config
    using AstroSGWBInference.ChainIO: atomic_save_chain
    using Distributions: Uniform, product_distribution
    using Turing
    using AdvancedHMC
    using ADTypes
    using Random
    using JLD2: load
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
