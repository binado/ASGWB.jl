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
    :Ξₙ => L"n",
    :γ => L"\gamma",
    :κ => L"\kappa",
    :zpeak => L"z_{\mathrm{peak}}"
)

param_label(sym::Symbol) = get(PARAM_LABELS, sym, LaTeXString(string(sym)))

function richtext_to_latex(x::AbstractString)
    x == " ± " && return " \\pm "
    x == "( " && return "("
    x == ") × 10" && return ") \\times 10"
    occursin('×', x) && return replace(x, "×" => "\\times")
    return x
end

function richtext_to_latex(rt::Makie.RichText)
    if rt.type === :leftsubsup
        low, high = rt.children
        return "_{$(low)}^{$(high)}"
    elseif rt.type === :sup
        return "^{$(only(rt.children))}"
    else
        return join(richtext_to_latex(c) for c in rt.children)
    end
end

richtext_to_latex(x) = string(x)

margin_quantile_latex_formatter(low, mid, high) =
    latexstring(richtext_to_latex(PairPlots.margin_confidence_default_formatter(low, mid, high)))

# PairPlots passes LaTeX formatters through `Makie.rich`, which only accepts String/RichText.
function Makie.rich(prev, title::LaTeXString; kwargs...)
    return title
end

function relabel_axes!(fap, lookup::Dict{Symbol, LaTeXString})
    fig = fap isa Makie.Figure ? fap : fap.figure
    for elem in fig.content
        elem isa Makie.Axis || continue
        sym = Symbol(string(elem.title[]))
        haskey(lookup, sym) && (elem.title[] = lookup[sym])
    end
    return fap
end

function chain_diagnostics_grid(chain; cfg::PlotConfig = PLOT_CONFIG)
    params = collect(FlexiChains.parameters(chain))
    n_params = length(params)
    n_params >= 1 || throw(ArgumentError("chain has no parameters to plot"))

    n_chains = size(chain, 2)
    n_draws = size(chain, 1)
    colors = chain_colors(cfg, n_chains)
    maxlag = min(100, max(1, n_draws - 1))

    panel_w, panel_h = 360, 180
    fig = Makie.Figure(size = (3 * panel_w + 80, n_params * panel_h + 80))

    Makie.Label(fig[0, 1], "Trace"; font = :bold, tellwidth = false)
    Makie.Label(fig[0, 2], "Running mean"; font = :bold, tellwidth = false)
    Makie.Label(fig[0, 3], "Autocorrelation"; font = :bold, tellwidth = false)

    trace_axes = Makie.Axis[]
    mean_axes = Makie.Axis[]
    autocor_axes = Makie.Axis[]

    for (i, p) in enumerate(params)
        Makie.Label(fig[i, 0], param_label(p); rotation = π / 2, tellheight = false)

        ax_trace = Makie.Axis(fig[i, 1])
        FlexiChains.mtraceplot!(ax_trace, chain, p; color = colors)
        push!(trace_axes, ax_trace)

        ax_mean = Makie.Axis(fig[i, 2])
        FlexiChains.mmeanplot!(ax_mean, chain, p; color = colors)
        push!(mean_axes, ax_mean)

        ax_ac = Makie.Axis(fig[i, 3])
        FlexiChains.mautocorplot!(ax_ac, chain, p; lags = 1:maxlag, color = colors)
        push!(autocor_axes, ax_ac)
    end

    Makie.linkxaxes!(trace_axes...)
    Makie.linkxaxes!(mean_axes...)

    for axes_col in (trace_axes, mean_axes, autocor_axes)
        for ax in axes_col[1:(end - 1)]
            Makie.hidexdecorations!(ax; grid = false)
        end
    end
    trace_axes[end].xlabel = "iteration"
    mean_axes[end].xlabel = "iteration"
    autocor_axes[end].xlabel = "lag"

    return fig
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
            PairPlots.MarginQuantileText(margin_quantile_latex_formatter; color = :black),
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
