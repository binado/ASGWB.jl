### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ 1b5c4d3e-6a7f-4c8b-9d2e-3f4a5b6c7d8e
begin
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
    using ASGWB
    using ASGWB:
                 build_model_context,
                 canonical_hyperparameters,
                 compute_importance_weights,
                 cosmology_type,
                 Detector,
                 full_hyperparameters,
                 PopulationModel,
                 AbstractCosmology,
                 ImportanceSamplingProblem,
                 load_catalog,
                 merger_rate,
                 OrderedUniformSourceMassPair,
                 AlignedSpinChiSimple,
                 redshift_prior,
                 MadauDickinsonSourceFrame,
                 BNS_LAMBDA_HIGH,
                 stack_source_masses,
                 spectral_density,
                 Ωgw
    using ASGWBInference: build_turing_model, condition_turing_model
    using ASGWBInference.ChainIO: atomic_save_chain
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
    using LinearAlgebra: BLAS
    BLAS.set_num_threads(1)
end

# ╔═╡ 8f3a2c1d-4e5b-4a6c-9d0e-1f2a3b4c5d6e
md"""
# MCMC (Pluto)

Canonical notebook-first MCMC workflow: inline model definition, inline fiducials and hyperprior bounds, `load_catalog`, explicit `ImportanceSamplingProblem`, **NUTS** via `build_turing_model` / `condition_turing_model` / `sample`, and **`atomic_save_chain`** output. No config TOML, no `model.toml`, no checkpoint callbacks.

Plots **Ω_GW(f)** at the fiducial point and optional **FlexiChains** / **PairPlots** diagnostics. Set **`chain_input_jld2`** to reload an existing chain without sampling.

## Environment

The first code cell runs `Pkg.activate(@__DIR__)`, so this notebook uses [notebooks/Project.toml](Project.toml). From the repository root:

```bash
julia -e 'using Pluto; Pluto.run(notebook="notebooks/mcmc_pluto.jl")'
```

Provide **`catalog.h5`** at the repo root (or change `catalog_path` in the config cell). **`ASGWB`** / **`ASGWBInference`** are path deps.
"""

# ╔═╡ 9a4b3c2d-5f6e-4b7a-8c1d-2e3f4a5b6c7d
begin
    num_threads = Base.Threads.nthreads()
    print(num_threads)
end

# ╔═╡ 2c6d5e4f-7b8a-4d9c-0e3f-4a5b6c7d8e9f
begin
    import ASGWB: hyperparameters, single_event_prior

    DEBUG = false
    @info debug = DEBUG

    struct BNSPopulationModel <: PopulationModel end

    hyperparameters(::BNSPopulationModel) = (:γ, :κ, :zpeak)

    # function hyperprior(::BNSPopulationModel)
    #     return product_distribution((
    #         γ = Uniform(0.5, 10.0),
    #         κ = Uniform(0.05, 10.0),
    #         zpeak = Uniform(0.05, 10.0)
    #     ))
    # end

    function single_event_prior(
            ::BNSPopulationModel, cache::CosmologyCache, Λ::NamedTuple)
        z_d = redshift_prior(MadauDickinsonSourceFrame(), cache, Λ)
        spin = AlignedSpinChiSimple()
        return product_distribution((
            mass = OrderedUniformSourceMassPair(),
            redshift = z_d,
            χ₁ = spin,
            χ₂ = spin,
            Λ₁ = Uniform(0.0, BNS_LAMBDA_HIGH),
            Λ₂ = Uniform(0.0, BNS_LAMBDA_HIGH)
        ))
    end

    function bns_samples_from_catalog(catalog_samples::NamedTuple)
        return (
            mass = stack_source_masses(
                catalog_samples.mass_1_source, catalog_samples.mass_2_source),
            redshift = copy(catalog_samples.redshift),
            χ₁ = copy(catalog_samples.chi_1),
            χ₂ = copy(catalog_samples.chi_2),
            Λ₁ = copy(catalog_samples.lambda_1),
            Λ₂ = copy(catalog_samples.lambda_2)
        )
    end

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
    detectors = [Detector("S1"), Detector("R1")]
    sample_only = (:H0, :w0)

    seed = 42
    @info "seeding RNG" rng_seed = seed
    Random.seed!(seed)

    local_merger_rate = 161.0 # Matches COBA simulations
    observation_time_yr = 1.0

    output_dir = joinpath(_repo_root, "chains")
    output_prefix = "chains"

    sampler = (
        n_samples = 3000,
        n_adapts = 3000,
        target_acceptance = 0.9,
        ad_backend = "ForwardDiff",
        num_chains = 0
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
        κ = 5.7,
        zpeak = 2.0
    )

    # Edit hyperprior bounds here (order: cosmology, then population).
    hyperprior_dists = (
        H0 = Uniform(20.0, 140.0),
        Ωm = Uniform(0.05, 0.95),
        w0 = Uniform(-3, 1),
        Ξ₀ = Uniform(0.5, 5.0),
        Ξₙ = Uniform(0.05, 3.0),
        γ = Uniform(0.5, 10.0),
        κ = Uniform(0.05, 10.0),
        zpeak = Uniform(0.05, 10.0)
    )
    hyperprior = product_distribution(hyperprior_dists)

    chain_input_jld2 = nothing

    num_chains = sampler.num_chains > 0 ? sampler.num_chains : num_threads
end

# ╔═╡ 3d7e6f5a-8c9b-4e0d-1f4a-5b6c7d8e9f0a
begin
    if num_chains != num_threads
        @warn "num_chains differs from Base.Threads.nthreads()" num_chains num_threads
    end

    @info "loading catalog" catalog_path detectors = join((d.name for d in detectors), ",")
    loaded = load_catalog(catalog_path)
    catalog = loaded.catalog
    C = ModifiedPropagation{W0CDM}
    @info hyperparameters(C)
    pop = BNSPopulationModel()
    samples = bns_samples_from_catalog(catalog.samples)
    problem = ImportanceSamplingProblem(pop, catalog.fluxes, samples, fiducials)
    ctx = build_model_context(
        problem,
        C,
        loaded.metadata.grid,
        detectors,
        observation_time_yr,
        local_merger_rate
    )
    order = full_hyperparameters(C, pop)
    @info order
    sample_only_tup = sample_only === nothing ? nothing : Tuple(sample_only)

    @info "catalog loaded" n_frequency_bins=length(ctx.observation.frequencies) n_proposal_samples=length(
        problem.samples.redshift,
    )

    @info "using fiducial in-band spectrum from cache as observed data"
    observed = ctx.fiducial_spectral_density


    mkpath(output_dir)
    timestamp = format(now(), "yyyymmdd-HHMMSS")
    det_suffix = join((d.name for d in detectors), ",")
    params_suffix = sample_only === nothing ? "all" : join(sample_only, "-")
    base = "$(output_prefix)-$(params_suffix)-det=$(det_suffix)-seed$(seed)-$(timestamp)"
    output_jld2 = joinpath(output_dir, "$base.jld2")

    nothing
end

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
        initial_params = fill(InitFromPrior(), num_chains)
        adtype = resolve_adtype(sampler.ad_backend)

        @info "starting NUTS" n_adapts=sampler.n_adapts n_samples=sampler.n_samples target_acceptance=sampler.target_acceptance ad_backend=sampler.ad_backend sample_only=sample_only_tup
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
            fiducials,
            hyperprior,
            sample_only_tup;
            order = order
        )
        nuts = Turing.NUTS(
            sampler.n_adapts,
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
                sampler.n_samples,
                num_chains;
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
## Storing the chains
"""

# ╔═╡ 9d3e2f1a-4b5c-4d6e-7f8a-9b0c1d2e3f4a
begin
    if chain_input_jld2 === nothing && chain != nothing
        @info "writing chain to JLD2" path = output_jld2
        atomic_save_chain(output_jld2, chain)
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

# ╔═╡ 5f9a8b7c-0e1d-4a2f-3b6c-7d8e9f0a1b2c
begin
    weights0 = compute_importance_weights(problem, C, fiducials, ctx)
    rate0 = merger_rate(problem, C, fiducials, ctx)
    Sh0 = spectral_density(problem.fluxes, rate0; weights = weights0)
    f = ctx.observation.frequencies
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
        lines!(ax, fm, Ωgw_pos; label = L"$\mathrm{model~at~fiducial}$")
        axislegend(ax; position = :rt)
    end
    fig
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

# ╔═╡ 3b7c6d5e-8f9a-4b0c-1d2e-3f4a5b6c7d8e
begin
    n_draws = size(chain, 1)
    autocor_maxlag = min(100, max(1, n_draws - 1))
    fig = FlexiChains.mautocorplot(chain; lags = 1:autocor_maxlag)
    fig
end

# ╔═╡ 2a6b5c4d-7e8f-4a9b-0c1d-2e3f4a5b6c7d
begin
    fig = FlexiChains.mtraceplot(chain)
    fig
end

# ╔═╡ Cell order:
# ╟─8f3a2c1d-4e5b-4a6c-9d0e-1f2a3b4c5d6e
# ╠═9a4b3c2d-5f6e-4b7a-8c1d-2e3f4a5b6c7d
# ╠═1b5c4d3e-6a7f-4c8b-9d2e-3f4a5b6c7d8e
# ╠═2c6d5e4f-7b8a-4d9c-0e3f-4a5b6c7d8e9f
# ╠═3d7e6f5a-8c9b-4e0d-1f4a-5b6c7d8e9f0a
# ╠═5f9a8b7c-0e1d-4a2f-3b6c-7d8e9f0a1b2c
# ╠═7b1c0d9e-2f3a-4c4b-5d6e-7f8a9b0c1d2e
# ╟─8c2d1e0f-3a4b-4c5d-6e7f-8a9b0c1d2e3f
# ╠═9d3e2f1a-4b5c-4d6e-7f8a-9b0c1d2e3f4a
# ╟─0e4f3a2b-5c6d-4e7f-8a9b-0c1d2e3f4a5b
# ╠═1f5a4b3c-6d7e-4f8a-9b0c-1d2e3f4a5b6c
# ╠═2a6b5c4d-7e8f-4a9b-0c1d-2e3f4a5b6c7d
# ╠═3b7c6d5e-8f9a-4b0c-1d2e-3f4a5b6c7d8e
# ╠═4c8d7e6f-9a0b-4c1d-2e3f-4a5b6c7d8e9f
