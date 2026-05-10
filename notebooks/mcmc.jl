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
#     display_name: Julia 1.12.6
#     language: julia
#     name: julia-1.12
# ---

# %% [markdown]
# # MCMC
#
# Same overall flow as `scripts/run_turing.jl`, but this notebook uses **unicode-key named tuples** (`Ωm`, `Ξ₀`, …) aligned with `HyperParameters` and the Turing `product_distribution` prior. On-disk JSON for the CLI still uses ASCII keys (`Omega_m`, …). After **`load_cache`**, it plots **Ω_GW(f)** at the initial `θ0` (via `evaluate_importance_terms` and `omegagw`) with **CairoMakie**, then runs **NUTS** in a dedicated cell with the same steps as `sample_with_turing` (`build_turing_model`, `condition_turing_model`, `InitFromParams`, `sample`).
#
# On-disk chains use **JLD2** with the top-level key **`chain`**, matching **`scripts/run_inference.jl`**. Set **`chain_input_jld2`** to a path (absolute or relative to the package root, like the cache HDF5 path) to skip sampling and load an existing run for diagnostics only.
#
# The first cell activates the **workspace subproject** `Project.toml` under `notebooks/` (Pkg **workspace** with the package root: one shared `Manifest.toml` at the repo root). Notebook-only packages (**`CairoMakie`**, **`LaTeXStrings`**, **`StatsPlots`**, **`Plots`**, **`MCMCChains`**, **`ArviZ`**, **`NCDatasets`**) live there; **`ASGWB`** is a path dev of the parent package. **`CairoMakie`** with **`LaTeXStrings`** (`L"..."`) draws Ω_GW; **`StatsPlots`** covers MCMC diagnostics; **`ArviZ`** (with **`NCDatasets`**) can convert samples to `InferenceData` and write NetCDF. **`Turing`** and the core **`ASGWB`** stack come from the devved package.

# %%
begin
    num_threads = Base.Threads.nthreads()
    print(num_threads)
end

# %%
begin
    import Pkg
    # Activates the environment in the directory where the notebook lives
    Pkg.activate(@__DIR__)
    # Ensure dependencies are installed for fresh clones or clean depots
    Pkg.instantiate()
    using ASGWB
    using ASGWB:
                 load_cache,
                 build_turing_model,
                 evaluate_importance_terms,
                 omegagw,
                 HyperParameters,
                 Detector,
                 DEFAULT_PARAMETER_ORDER
    using Turing
    using AdvancedHMC
    using Random
    using JLD2
    using Logging
    using MCMCChains
    using ArviZ
    using NCDatasets
    using StatsPlots
    using Plots
    using CairoMakie
    using LaTeXStrings
    using Distributions
    default(size = (900, 450))
end

# %%
begin
    using DelimitedFiles

    function load_observed_spectral_density(path::AbstractString, expected_len::Int)
        isfile(path) ||
            throw(ArgumentError("observed spectrum file not found: $(repr(path))"))
        v = vec(readdlm(path, ',', Float64))
        length(v) == expected_len || throw(
            ArgumentError(
            "observed_spectral_density_csv has length $(length(v)), expected $expected_len",
        ),
        )
        return v
    end

    """Check each `init` scalar has positive prior density under the matching `priors` entry."""
    function validate_init_against_priors(priors, init)
        for (k, d) in pairs(priors)
            v = init[k]
            isfinite(logpdf(d, v)) || throw(
                ArgumentError(
                "init.$k = $v is outside the support of the corresponding prior",
            ),
            )
        end
        return nothing
    end

    # --- edit everything below (same role as the JSON used by `scripts/run_turing.jl`) ---

    cache = "analysis_numpyro_julia_cache.h5"
    detectors = [Detector("S1"), Detector("R1")]
    sample_only = [:H0,]

    priors = (
        H0 = Uniform(20, 140),
        Ωm = Uniform(0.05, 0.95),
        Ξ₀ = Uniform(0.5, 5),
        Ξₙ = Uniform(0.05, 3),
        γ = Uniform(0.5, 10),
        κ = Uniform(0.05, 10),
        zpeak = Uniform(0.05, 10)
    )

    init = (H0 = 67.66, Ωm = 0.3096, Ξ₀ = 1.0, Ξₙ = 1.91, γ = 2.7, κ = 5.7, zpeak = 2.0)
    fixed_sites = (; (k => init[k] for k in DEFAULT_PARAMETER_ORDER if k ∉ sample_only)...)

    sampler = (n_samples = 2000, n_adapts = 2000, target_acceptance = 0.9)

    seed = 1
    observed_spectral_density_csv = nothing
    output_suffix = join(map(string, sample_only), "-")
    output_jld2 = "chains-$output_suffix.jld2"
    output_netcdf = "chains-$output_suffix.nc"
    chain_input_jld2 = nothing

    validate_init_against_priors(priors, init)
    priors_turing = product_distribution((
        H0 = priors.H0,
        Ωm = priors.Ωm,
        Ξ₀ = priors.Ξ₀,
        Ξₙ = priors.Ξₙ,
        γ = priors.γ,
        κ = priors.κ,
        zpeak = priors.zpeak
    ))
    θ0 = HyperParameters(; init...)
end

# %%
fixed_sites

# %%
begin
    function validate_sample_only(sample_only::Union{Nothing, Tuple{Vararg{Symbol}}})
        sample_only === nothing && return nothing
        isempty(sample_only) && throw(
            ArgumentError(
            "sample_only must not be empty; omit the key or use null to sample every hyperparameter",
        ),
        )
        for s in sample_only
            s in DEFAULT_PARAMETER_ORDER || throw(
                ArgumentError(
                "sample_only contains $(repr(s)); expected symbols from $(DEFAULT_PARAMETER_ORDER)",
            ),
            )
        end
        length(unique(sample_only)) == length(sample_only) ||
            throw(ArgumentError("sample_only must not repeat symbols"))
        return nothing
    end

    function turing_initial_params(
            theta0::HyperParameters,
            sample_only::Union{Nothing, Tuple{Vararg{Symbol}}}
    )
        sample_only === nothing && return InitFromParams(theta0)
        return InitFromParams((; (s => theta0[s] for s in sample_only)...))
    end

    cd(pkgdir(ASGWB))
    @info "loading importance cache" path=cache detectors=join((d.name for d in detectors), ",")
    t_cache = time()
    problem = load_cache(cache, detectors)
    @info "cache loaded" seconds=round(time()-t_cache; digits = 2) n_frequency_bins=length(problem.observation.frequencies) n_proposal_samples=length(problem.proposal.samples.redshift)

    observed = if observed_spectral_density_csv === nothing
        @info "using fiducial in-band spectrum from cache as observed data"
        problem.observation.fiducial_spectral_density
    else
        @info "loading observed spectrum from CSV" path = observed_spectral_density_csv
        load_observed_spectral_density(
            observed_spectral_density_csv,
            length(problem.observation.fiducial_spectral_density)
        )
    end

    if seed !== nothing
        @info "seeding RNG" rng_seed = seed
        Random.seed!(seed)
    else
        @info "RNG seed not set (nondeterministic run unless Julia was seeded elsewhere)"
    end

    sample_only_tup = if sample_only === nothing
        nothing
    else
        Tuple(sample_only)
    end
    #validate_sample_only(sample_only_tup)
    sam = sampler
    nothing
end

# %%
begin
    ev = evaluate_importance_terms(θ0, problem)
    f = problem.observation.frequencies
    Ωgw = omegagw(ev.spectral_density, f, θ0)
    mask = Ωgw .> 0.0
    fm = f[mask]
    Ωm = Ωgw[mask]
    fig = Figure(size = (900, 450))
    ax = Axis(
        fig[1, 1];
        xlabel = L"$f~\mathrm{(Hz)}$",
        ylabel = L"$\Omega_{\mathrm{GW}}(f)$",
        xscale = log10,
        yscale = log10,
        limits = (nothing, nothing, 1e-15, nothing)
    )
    if !isempty(Ωm)
        lines!(ax, fm, Ωm; label = L"$\mathrm{model~at}~\theta_0$")
        axislegend(ax; position = :rt)
    end
    fig
end

# %%
Ωgw

# %%
begin
    if chain_input_jld2 !== nothing
        chain_path = isabspath(chain_input_jld2) ? String(chain_input_jld2) :
            normpath(joinpath(pkgdir(ASGWB), chain_input_jld2))
        isfile(chain_path) ||
            throw(ArgumentError("JLD2 chain file not found: $(repr(chain_path))"))
        @info "loading chain from JLD2" path = chain_path
        chain = load(chain_path)["chain"]
        @info "chain loaded" chain_size = size(chain)
    else
        @info "starting NUTS" n_adapts = sam.n_adapts n_samples = sam.n_samples target_acceptance = sam.target_acceptance sample_only = sample_only_tup
        model = build_turing_model(
            problem, priors_turing; track = true, observed_spectral_density = observed
        )
        conditioned = model | fixed_sites
        nuts = Turing.NUTS(
            sam.n_adapts,
            sam.target_acceptance;
            metricT = AdvancedHMC.DenseEuclideanMetric
        )
        chain = sample(
            conditioned,
            nuts,
            MCMCThreads(),
            sam.n_samples,
            num_threads;
            progress = true,
            save_state = true
        )
        @info "NUTS finished" chain_size = size(chain)
    end
    chain
end

# %% [markdown]
# ## Storing the chains

# %%
begin
    idata = from_mcmcchains(chain; library = "Turing")
    if output_netcdf !== nothing
        @info "Writing InferenceData to NetCDF" path = output_netcdf
        to_netcdf(idata, output_netcdf)
        @info "Done"
    end
    if chain_input_jld2 === nothing
        @info "Saving chain object to JLD2" path = output_jld2
        jldsave(output_jld2; chain)
        @info "Done"
    else
        @info "Skipping JLD2 save (chain was loaded from disk)"
    end
end

# %% [markdown]
# ## Diagnostic plots

# %%
describe(chain)

# %%
traceplot(chain)

# %%
autocorplot(chain)

# %%
let pnames = names(chain, :parameters)
    if length(pnames) >= 2
        MCMCChains.corner(chain)
    else
        StatsPlots.density(chain)
    end
end
