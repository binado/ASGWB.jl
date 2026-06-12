# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Julia 1.12
#     language: julia
#     name: julia-1.12
# ---

# %% [markdown]
# # MCMC
#
# Notebook-first MCMC workflow matching `mcmc_pluto.jl`: inline population model,
# inline fiducials and hyperprior bounds, `load_catalog`, explicit
# `ImportanceSamplingProblem`, Turing NUTS sampling, JLD2 chain output, and plots.

# %%
begin
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()

    using ASGWB
    using ASGWB:
                 AbstractCosmology,
                 AlignedSpinChiSimple,
                 BNS_LAMBDA_HIGH,
                 Detector,
                 ImportanceSamplingProblem,
                 MadauDickinsonSourceFrame,
                 OrderedUniformSourceMassPair,
                 PopulationModel,
                 build_model_context,
                 compute_importance_weights,
                 full_hyperparameters,
                 load_catalog,
                 merger_rate,
                 redshift_prior,
                 spectral_density,
                 stack_source_masses,
                 Ωgw
    using ASGWBInference: atomic_save_chain, build_turing_model, condition_turing_model
    import ASGWB: hyperparameters, single_event_prior
    using AdvancedHMC
    using CairoMakie
    using Dates: format, now
    using Distributions: Uniform, product_distribution
    using FlexiChains
    using FlexiChains: VNChain
    using JLD2: load
    using LaTeXStrings
    using LinearAlgebra: BLAS
    using PairPlots
    using Random
    using Turing

    BLAS.set_num_threads(1)
    num_threads = Base.Threads.nthreads()
end

# %%
begin
    struct BNSPopulationModel <: PopulationModel end

    hyperparameters(::BNSPopulationModel) = (:γ, :κ, :zpeak)

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
end

# %%
begin
    _repo_root = normpath(joinpath(@__DIR__, ".."))

    catalog_path = joinpath(_repo_root, "catalog.h5")
    detectors = [Detector("E1"), Detector("E2"), Detector("E3")]
    sample_only = (:Ξ₀,)

    seed = 42
    Random.seed!(seed)

    local_merger_rate = 161.0
    observation_time_yr = 1.0

    output_dir = joinpath(_repo_root, "chains")
    output_prefix = "chains"

    sampler = (
        n_samples = 3000,
        n_adapts = 3000,
        target_acceptance = 0.9,
        num_chains = 0
    )

    cosmology_parameters = (;
        H0 = 67.66,
        Ωm = 0.3096,
        Ξ₀ = 1.0,
        Ξₙ = 1.91
    )
    cosmology_model = LambdaCDM(cosmology_parameters.H0, cosmology_parameters.Ωm)
    cosmology = ModifiedPropagation(
        cosmology_model,
        cosmology_parameters.Ξ₀,
        cosmology_parameters.Ξₙ
    )
    C = typeof(cosmology)

    fiducials = (;
        cosmology_parameters...,
        γ = 2.7,
        κ = 5.7,
        zpeak = 2.0
    )

    hyperprior = product_distribution((
        H0 = Uniform(20.0, 140.0),
        Ωm = Uniform(0.05, 0.95),
        Ξ₀ = Uniform(0.5, 5.0),
        Ξₙ = Uniform(0.05, 3.0),
        γ = Uniform(0.5, 10.0),
        κ = Uniform(0.05, 10.0),
        zpeak = Uniform(0.05, 10.0)
    ))

    chain_input_jld2 = nothing
    num_chains = sampler.num_chains > 0 ? sampler.num_chains : num_threads
end

# %%
begin
    loaded = load_catalog(catalog_path)
    pop = BNSPopulationModel()
    samples = bns_samples_from_catalog(loaded.catalog.samples)
    problem = ImportanceSamplingProblem(pop, loaded.catalog.fluxes, samples, fiducials)
    ctx = build_model_context(
        problem,
        C,
        loaded.metadata.grid,
        detectors,
        observation_time_yr,
        local_merger_rate
    )
    order = full_hyperparameters(C, pop)
    sample_only_tup = sample_only === nothing ? nothing : Tuple(sample_only)
    observed = ctx.fiducial_spectral_density

    mkpath(output_dir)
    timestamp = format(now(), "yyyymmdd-HHMMSS")
    det_suffix = join((d.name for d in detectors), ",")
    params_suffix = sample_only === nothing ? "all" : join(sample_only, "-")
    base = "$(output_prefix)-$(params_suffix)-det=$(det_suffix)-seed$(seed)-$(timestamp)"
    output_jld2 = joinpath(output_dir, "$base.jld2")
end

# %%
begin
    weights0 = compute_importance_weights(problem, C, fiducials, ctx)
    rate0 = merger_rate(problem, C, fiducials, ctx)
    Sh0 = spectral_density(problem.fluxes, rate0; weights = weights0)
    f = ctx.observation.frequencies
    Ωgw_plot = Ωgw(Sh0, f, fiducials.H0)
    mask = Ωgw_plot .> 0.0

    fig = Figure(size = (900, 450))
    ax = Axis(
        fig[1, 1];
        xlabel = L"$f~\mathrm{(Hz)}$",
        ylabel = L"$\Omega_{\mathrm{GW}}(f)$",
        xscale = log10,
        yscale = log10,
        limits = (nothing, nothing, 1e-15, nothing)
    )
    if any(mask)
        lines!(ax, f[mask], Ωgw_plot[mask]; label = L"$\mathrm{model~at~fiducial}$")
        axislegend(ax; position = :rt)
    end
    fig
end

# %%
begin
    if chain_input_jld2 !== nothing
        chain_path = isabspath(chain_input_jld2) ? String(chain_input_jld2) :
                     normpath(joinpath(_repo_root, chain_input_jld2))
        isfile(chain_path) ||
            throw(ArgumentError("JLD2 chain file not found: $(repr(chain_path))"))
        chain = load(chain_path)["chain"]
    else
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
            metricT = AdvancedHMC.DenseEuclideanMetric
        )
        chain = sample(
            conditioned,
            nuts,
            MCMCThreads(),
            sampler.n_samples,
            num_chains;
            progress = true,
            save_state = false,
            chain_type = VNChain,
            initial_params = fill(InitFromPrior(), num_chains)
        )
    end
    chain
end

# %% [markdown]
# ## Storing The Chains

# %%
begin
    if chain_input_jld2 === nothing
        atomic_save_chain(output_jld2, chain)
    end
end

# %% [markdown]
# ## Diagnostic Plots

# %%
summarystats(chain)

# %%
FlexiChains.mtraceplot(chain)

# %%
begin
    n_draws = size(chain, 1)
    autocor_maxlag = min(100, max(1, n_draws - 1))
    FlexiChains.mautocorplot(chain; lags = 1:autocor_maxlag)
end

# %%
begin
    chain_params = FlexiChains.parameters(chain)
    if length(chain_params) >= 2
        pairplot(chain)
    else
        Makie.density(chain)
    end
end
