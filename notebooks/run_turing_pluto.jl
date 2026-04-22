### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ bb8c74e6-36b3-11f1-84a9-df5091ee4210
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
    using Random
    using Serialization
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


# ╔═╡ aa7c7524-36b3-11f1-bd4e-1121e886c676
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
    detectors = [Detector("E1"), Detector("E2"), Detector("E3")]
    sample_only = [:Ξ0]

    priors = (
        H0 = Uniform(20, 140),
        Ωm = Uniform(0.05, 0.95),
        Ξ0 = Uniform(0.5, 5),
        Ξn = Uniform(0.05, 3),
        γ = Uniform(0.5, 10),
        zp = Uniform(0.05, 10),
        κ = Uniform(0.05, 10)
    )

    init = (H0 = 67.66, Ωm = 0.3096, Ξ0 = 1.0, Ξn = 1.91, γ = 2.7, κ = 5.7, zp = 2.0)
    to_ascii = (
        H0 = :H0,
        Ωm = :Omega_m,
        Ξ0 = :chi0,
        Ξn = :chin,
        γ = :gamma,
        κ = :kappa,
        zp = :z_peak
    )
    fixed_sites = (; (to_ascii[k] => v for (k, v) in pairs(init) if k ∉ sample_only)...)

    sampler = (n_samples = 2000, n_adapts = 2000, target_acceptance = 0.9)

    seed = 1
    observed_spectral_density_csv = nothing
    output_jls = nothing
    output_netcdf = nothing

    validate_init_against_priors(priors, init)
    priors_turing = product_distribution((
        H0 = priors.H0,
        Omega_m = priors.Ωm,
        chi0 = priors.Ξ0,
        chin = priors.Ξn,
        gamma = priors.γ,
        kappa = priors.κ,
        z_peak = priors.zp
    ))
    θ0 = HyperParameters(;
        H0 = init.H0,
        Omega_m = init.Ωm,
        chi0 = init.Ξ0,
        chin = init.Ξn,
        gamma = init.γ,
        kappa = init.κ,
        z_peak = init.zp
    )
end


# ╔═╡ aa7c72a2-36b3-11f1-a5e9-17b92e804e41
md"""
# ASGWB Turing sampling

Same overall flow as [`scripts/run_turing.jl`](../scripts/run_turing.jl), but this notebook keeps **human-facing** settings as **unicode-key named tuples** (`Ωm`, `Ξ0`, …), maps once into the package’s ASCII product-distribution prior and `HyperParameters` NamedTuple (what the Turing `@model` expects). After **`load_cache`**, it plots **Ω_GW(f)** at the initial `θ0` (via `evaluate_importance_terms` and `omegagw`) with **CairoMakie**, then runs **NUTS** in a dedicated cell with the same steps as `sample_with_turing` (`build_turing_model`, `condition_turing_model`, `InitFromParams`, `sample`).

The first cell activates the **workspace subproject** [`Project.toml`](./Project.toml) under `notebooks/` (Pkg **workspace** with the package root: one shared [`Manifest.toml`](../Manifest.toml) at the repo root). Notebook-only packages (**`CairoMakie`**, **`LaTeXStrings`**, **`StatsPlots`**, **`Plots`**, **`Pluto`**, **`MCMCChains`**, **`ArviZ`**, **`NCDatasets`**) live there; **`ASGWB`** is a path dev of the parent package. **`CairoMakie`** with **`LaTeXStrings`** (`L"..."`) draws Ω_GW; **`StatsPlots`** covers MCMC diagnostics; **`ArviZ`** (with **`NCDatasets`**) can convert samples to [**`InferenceData`**](https://arviz-devs.github.io/ArviZ.jl/stable/api/inference_data/) and write NetCDF. **`Turing`** and the core **`ASGWB`** stack come from the devved package.
"""


# ╔═╡ a9aeb877-2396-49b6-856e-c719be5db6d7
begin
    num_threads = Base.Threads.nthreads()
    print(num_threads)
end

# ╔═╡ aa7c7572-36b3-11f1-a66c-f1c2e7b4f465
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


# ╔═╡ 954323e3-b79e-4b1e-9200-dcf074777345
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


# ╔═╡ 949e4bd9-1ce8-44f0-a8d6-04c8e966641a
Ωgw

# ╔═╡ 23f963ee-0675-4f7f-875d-a8afd443e166
begin
    @info "starting NUTS" n_adapts=sam.n_adapts n_samples=sam.n_samples target_acceptance=sam.target_acceptance sample_only=sample_only_tup

    t_sample = time()
    model = build_turing_model(problem, priors_turing; track = false, observed_spectral_density = observed)
    conditioned = model | fixed_sites
    nuts = NUTS(sam.n_adapts, sam.target_acceptance)
    chain = sample(
        conditioned,
        nuts,
        MCMCThreads(),
        sam.n_samples,
        num_threads;
        progress = true
    )
    @info "NUTS finished" seconds=round(time()-t_sample; digits = 2) chain_size=size(chain)

    if output_jls !== nothing
        @info "serializing chain" path = output_jls
        open(output_jls, "w") do io
            serialize(io, chain)
        end
        @info "wrote chain to disk" path = output_jls
    end

    idata = from_mcmcchains(chain; library = "Turing")
    model_track = build_turing_model(problem, priors_turing; track = true, observed_spectral_density = observed)
    conditioned_track = model_track | fixed_sites
    extras = Turing.returned(conditioned_track, chain)
    idata.posterior["spectral_snr"] = map(x -> x.spectral_snr, extras)
    idata.posterior["effective_sample_size"] = map(x -> x.effective_sample_size, extras)
    idata.posterior["number_of_sources"] = map(x -> x.number_of_sources, extras)
    if output_netcdf !== nothing
        @info "writing InferenceData to NetCDF" path = output_netcdf
        to_netcdf(idata, output_netcdf)
        @info "wrote InferenceData to NetCDF" path = output_netcdf
    end

    @info "sampling cell complete" chain_size = size(chain)
    chain
end


# ╔═╡ e4fcf73c-7193-445b-8d55-1ad9a9645caa
describe(chain)

# ╔═╡ aa7c7584-36b3-11f1-9894-cd9b12e6b4fe
traceplot(chain)


# ╔═╡ 622dc36b-a0f2-482e-b562-70ee63d5904a
autocorplot(chain)

# ╔═╡ aa7c759a-36b3-11f1-af66-b5a6a831a0c8
let pnames = names(chain, :parameters)
    if length(pnames) >= 2
        StatsPlots.corner(chain)
    else
        StatsPlots.density(chain)
    end
end


# ╔═╡ Cell order:
# ╠═aa7c72a2-36b3-11f1-a5e9-17b92e804e41
# ╠═a9aeb877-2396-49b6-856e-c719be5db6d7
# ╠═bb8c74e6-36b3-11f1-84a9-df5091ee4210
# ╠═aa7c7524-36b3-11f1-bd4e-1121e886c676
# ╠═aa7c7572-36b3-11f1-a66c-f1c2e7b4f465
# ╠═954323e3-b79e-4b1e-9200-dcf074777345
# ╠═949e4bd9-1ce8-44f0-a8d6-04c8e966641a
# ╠═23f963ee-0675-4f7f-875d-a8afd443e166
# ╠═e4fcf73c-7193-445b-8d55-1ad9a9645caa
# ╠═aa7c7584-36b3-11f1-9894-cd9b12e6b4fe
# ╠═622dc36b-a0f2-482e-b562-70ee63d5904a
# ╠═aa7c759a-36b3-11f1-af66-b5a6a831a0c8
