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
# # MCMC post-processing

# %%
begin
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()

    include(joinpath(@__DIR__, "plotting", "Plotting.jl"))
    using .Plotting

    using CairoMakie
    using PairPlots
    using Statistics
    using Turing
    using FlexiChains
    using FlexiChains: Extra, FlexiChain
    using ASGWB
end

# %%
const FIDUCIALS = (;
    H0 = 67.66,
    Ωm = 0.3096,
    w0 = -1.0,
    wa = 0.0,
    Ξ₀ = 1.0,
    Ξₙ = 1.91,
    γ = 2.7,
    κ = 5.7,
    zpeak = 2.0
)

# %% [markdown]
# ## Loading chains

# %%
filepath = get(ENV, "ASGWB_CHAIN_FILE", "chains/chains-H0-seed13-20260508-183716-slim-flexi.jld2")

chain_path = (realpath ∘ joinpath)(@__DIR__, "..", filepath)

# %%
chain = load_chain(chain_path)

# %%
chain_params = FlexiChains.parameters(chain)

# %% [markdown]
# ## Data

# %%
chain_params

# %% [markdown]
# ## Diagnostics

# %%
summarystats(chain)

# %% [markdown]
# ## Chain diagnostics

# %%
begin
    fig = chain_diagnostics_grid(chain)
    save_figure(fig, "chain_diagnostics")
    fig
end

# %% [markdown]
# ## Posterior distributions

# %%
begin
    chn = FlexiChains.subset_parameters(chain)
    fig = if length(chain_params) >= 2
        truths = PairPlots.Truth(
            (; (k => FIDUCIALS[k] for k in chain_params)...);
            color = :black
        )
        viz = (
            PairPlots.Scatter(filtersigma = 2, color = PLOT_CONFIG.primary_color),
            PairPlots.Contour(color = PLOT_CONFIG.primary_color),
            PairPlots.Contourf(color = (PLOT_CONFIG.primary_color, PLOT_CONFIG.alpha)),
            PairPlots.MarginDensity(;
                color = (PLOT_CONFIG.primary_color, 1),
                strokecolor = PLOT_CONFIG.primary_color,
                strokewidth = PLOT_CONFIG.strokewidth
            ),
            PairPlots.MarginQuantileText(margin_quantile_latex_formatter; color = :black)
        )
        fig = pairplot(chn => viz, truths; labels = PARAM_LABELS)
    else
        fap = Makie.density(chn;
            pool_chains = true,
            legend_position = :none,
            color = (PLOT_CONFIG.primary_color, PLOT_CONFIG.alpha),
            strokecolor = PLOT_CONFIG.primary_color,
            strokewidth = PLOT_CONFIG.strokewidth
        )
        relabel_axes!(fap, PARAM_LABELS)
    end
    save_figure(fig, length(chain_params) >= 2 ? "pairplot" : "posterior_density")
    fig
end
