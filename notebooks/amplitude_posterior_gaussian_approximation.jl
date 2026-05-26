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
# # Posterior vs. Fisher-Gaussian approximation
#
# Overlays `Normal(μ_fid, 1/SNR)` on the empirical 1D posterior density.
# Requires a **single-parameter chain** (e.g. produced with `sample_only = (:H0,)`)
# and the importance-sampling **cache HDF5** that the run was built from.
#
# - `μ_fid` — fiducial value of the parameter (read from the cache via `fiducial_hyperparameters`)
# - `SNR` — matched-filter SNR of the fiducial spectral density against the network PSD
# - `σ_Fisher = 1/SNR` — the Cramér-Rao lower bound for a linear amplitude parameter
#
# This is intentionally an amplitude/SNR approximation with all other parameters fixed. It does
# not compute the local curvature for the sampled hyperparameter or apply parameter Jacobians such
# as `A = 1 / H0`.

# %%
begin
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()

    include(joinpath(@__DIR__, "plotting", "Plotting.jl"))
    using .Plotting

    using CairoMakie
    using Distributions
    using Statistics
    using FlexiChains
    using FlexiChains: Extra, FlexiChain, Parameter
    using ASGWB
    using ASGWB: Detector
end

# %% [markdown]
# ## Configuration

# %%
begin
    chain_filepath = get(
        ENV, "ASGWB_CHAIN_FILE",
        "chains/chains-H0-seed13-20260508-183716-slim-flexi.jld2"
    )
    chain_path = (realpath ∘ joinpath)(@__DIR__, "..", chain_filepath)

    cache_filepath = get(ENV, "ASGWB_CACHE_FILE", "analysis_numpyro_julia_cache.h5")
    cache_path = (realpath ∘ joinpath)(@__DIR__, "..", cache_filepath)

    # Detector network used to build the cache — changing this changes the effective PSD
    # and therefore the SNR, so it is kept explicit rather than driven by ENV.
    detectors = [Detector("S1"), Detector("R1")]
end

# %% [markdown]
# ## Loading the chain

# %%
chain = load_chain(chain_path)

# %% [markdown]
# ## Validate single-parameter chain

# %%
begin
    chain_params = FlexiChains.parameters(chain)
    length(chain_params) == 1 || throw(ArgumentError(
        "expected single-parameter chain, got $chain_params — " *
        "this notebook is purpose-built for 1D posterior overlays"
    ))
    param_name = only(chain_params)
    param_sym = Symbol(string(param_name))   # VarName → Symbol for NamedTuple lookup
    @info "single-parameter chain validated" param = param_name
end

# %% [markdown]
# ## Load cache and derive fiducial value + Fisher σ

# %%
begin
    @info "loading importance cache" path=cache_path detectors=join(
        (d.name for d in detectors), ",")
    problem = load_cache(cache_path, detectors)

    fid_params = fiducial_hyperparameters(problem)
    haskey(fid_params, param_sym) || throw(ArgumentError(
        "parameter $param_sym not found in fiducial_hyperparameters; " *
        "available keys: $(keys(fid_params))"
    ))
    μ_fid = fid_params[param_sym]

    sd_fid = problem.observation.fiducial_spectral_density
    m = problem.observation.in_band_mask
    psd = problem.observation.effective_psd
    T = problem.observation.observation_time_sec
    df = frequency_bin_width(problem.observation.frequencies)

    snr = spectral_snr(sd_fid[m], psd[m], T, df)
    σ_fisher = 1 / snr

    @info "Fisher-Gaussian parameters" μ_fid snr σ_fisher
end

# %% [markdown]
# ## Posterior density + Fisher-Gaussian overlay

# %%
begin
    samples = vec(Array(chain[Parameter(param_name)]))
    lo, hi = extrema(samples)
    pad = 0.05 * (hi - lo)
    xs = range(
        min(lo - pad, μ_fid - 4σ_fisher),
        max(hi + pad, μ_fid + 4σ_fisher);
        length = 400
    )

    fig = Figure(size = (800, 500))
    ax = Axis(fig[1, 1]; xlabel = string(param_name), ylabel = "density")

    Makie.density!(ax, chain, Parameter(param_name);
        pool_chains = true, label = "posterior")
    Makie.lines!(ax, collect(xs), pdf.(Normal(μ_fid, σ_fisher), xs);
        color = :black, linewidth = 2,
        label = "Normal(μ_fid, 1/SNR)")
    Makie.vlines!(ax, [μ_fid];
        color = (:black, 0.5), linestyle = :dash, label = "fiducial")

    axislegend(ax; position = :rt, framevisible = false)

    save_figure(fig, "amplitude_posterior_gaussian_overlay")
    fig
end
