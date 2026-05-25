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

    using CairoMakie
    using JLD2
    using LaTeXStrings
    using PairPlots
    using Statistics
    using Turing
    using FlexiChains
    using FlexiChains: Extra, FlexiChain
    using ASGWB
end

# %%
output_dir = joinpath(@__DIR__, "..", get(ENV, "ASGWB_FIGURES_DIR", "output-test-figures"))
const FIGURE_DPI = 300

# Makie figure sizes are CSS pixels; 1 in = 96 CSS px (see Makie docs).
const MAKIE_CSS_PX_PER_INCH = 96
const MAKIE_DEFAULT_PT_PER_UNIT = 0.75

function _makie_save_kwargs(dpi::Int)
    scale = dpi / MAKIE_CSS_PX_PER_INCH
    return (; px_per_unit = scale, pt_per_unit = MAKIE_DEFAULT_PT_PER_UNIT * scale)
end

function _save_plot_object!(obj, path::AbstractString; dpi::Int)
    return save(path, obj; _makie_save_kwargs(dpi)...)
end

function save_figure(
        fig,
        name::AbstractString;
        output_dir::Union{Nothing, AbstractString} = output_dir,
        dpi::Int = FIGURE_DPI
)
    output_dir === nothing && return fig
    mkpath(output_dir)
    stem = joinpath(output_dir, name)
    try
        _save_plot_object!(fig, stem * ".pdf"; dpi)
    catch err
        @warn "PDF export failed; saving PNG instead" name exception = err
        _save_plot_object!(fig, stem * ".png"; dpi)
    end
    return fig
end

# %%
Base.@kwdef struct PlotConfig
    palette::Vector{Makie.RGBAf} = Makie.wong_colors()
    primary_color::Makie.RGBAf = Makie.wong_colors()[1]
    alpha::Float64 = 0.5
    linewidth::Float64 = 1.2
    strokewidth::Float64 = 2.0
    smooth_window::Int = 25
    draw_stride::Int = 5
end

const PLOT_CONFIG = PlotConfig()

function chain_colors(cfg::PlotConfig, n::Integer; alpha::Real = cfg.alpha)
    return [(cfg.palette[mod1(i, length(cfg.palette))], alpha) for i in 1:n]
end

# %%
const PARAM_LABELS = Dict{Symbol, LaTeXString}(
    :H0 => L"H_0",
    :Ωm => L"\Omega_m",
    :w0 => L"w_0",
    :wa => L"w_a",
    :Ξ₀ => L"\Xi_0",
    :Ξₙ => L"\Xi_n",
    :γ => L"\gamma",
    :κ => L"\kappa",
    :zpeak => L"z_{\mathrm{peak}}"
)

param_label(sym::Symbol) = get(PARAM_LABELS, sym, LaTeXString(string(sym)))

function relabel_axes!(fap, lookup::Dict{Symbol, LaTeXString})
    fig = fap isa Makie.Figure ? fap : fap.figure
    for elem in fig.content
        elem isa Makie.Axis || continue
        sym = Symbol(string(elem.title[]))
        haskey(lookup, sym) && (elem.title[] = lookup[sym])
    end
    return fap
end

# %% [markdown]
# ## Loading chains

# %%
filepath = get(ENV, "ASGWB_CHAIN_FILE", "chains/chains-H0-seed13-20260508-183716-slim-flexi.jld2")

chain_path = (realpath ∘ joinpath)(@__DIR__, "..", filepath)

# %%
function _load_chain(path::AbstractString)
    isfile(path) || throw(ArgumentError("JLD2 file not found: $(repr(path))"))
    data = load(path)
    if haskey(data, "chain")
        return data["chain"]
    elseif haskey(data, "snapshot")
        return data["snapshot"]
    else
        throw(ArgumentError(
            "JLD2 file contains neither 'chain' nor 'snapshot' key: $(repr(path))",
        ))
    end
end

# %%
chain = _load_chain(chain_path)

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
# ## Trace and autocorrelation plots

# %%
begin
    fig = FlexiChains.mtraceplot(
        chain; color = chain_colors(PLOT_CONFIG, size(chain, 2)), legend_position = :right)
    relabel_axes!(fig, PARAM_LABELS)
    save_figure(fig, "traceplot")
    fig
end

# %%
begin
    n_draws = size(chain, 1)
    autocor_maxlag = min(100, max(1, n_draws - 1))
    fig = FlexiChains.mautocorplot(
        chain;
        lags = 1:autocor_maxlag,
        color = chain_colors(PLOT_CONFIG, size(chain, 2)),
        legend_position = :right
    )
    relabel_axes!(fig, PARAM_LABELS)
    save_figure(fig, "autocorplot")
    fig
end

# %%
begin
    fig = FlexiChains.mmeanplot(
        chain; color = chain_colors(PLOT_CONFIG, size(chain, 2)), legend_position = :right)
    relabel_axes!(fig, PARAM_LABELS)
    save_figure(fig, "meanplot")
    fig
end

# %% [markdown]
# ## Posterior distributions

# %%
begin
    fig = if length(chain_params) >= 2
        pairplot(
            PairPlots.Series(chain; label = "posterior",
                color = PLOT_CONFIG.primary_color) => (
                PairPlots.Contour(color = PLOT_CONFIG.primary_color),
                PairPlots.Contourf(color = (PLOT_CONFIG.primary_color, PLOT_CONFIG.alpha)),
                PairPlots.MarginDensity(
                    color = (PLOT_CONFIG.primary_color, PLOT_CONFIG.alpha),
                    strokecolor = PLOT_CONFIG.primary_color,
                    strokewidth = PLOT_CONFIG.strokewidth
                )
            );
            pool_chains = true,
            labels = PARAM_LABELS
        )
    else
        fap = Makie.density(chain;
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
