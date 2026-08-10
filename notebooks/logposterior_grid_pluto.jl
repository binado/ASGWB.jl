### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ 7e3f2a1b-4c5d-4e6f-8a7b-9c8d0e1f2a3b
md"""
# Logposterior Grid (Pluto)

Sampler-free posterior visualization: builds the conditioned Turing model with the same canonical adapter as `mcmc.jl`, wraps it with `DynamicPPL.LogDensityFunction` (non-linked ⇒ physical-space logposterior, no Jacobian), and evaluates the **logposterior on a regular grid** over the free (sampled) parameters. Plots the result as a 1-D line or 2-D heatmap with CairoMakie.

With `observed` synthesized by `forward_model` at the fiducial, the likelihood is largest at the fiducial
point (flat priors); the full log-posterior also includes the prior. Useful for checking
posterior geometry and identifiability before or without running HMC.

## Environment

The first code cell runs `Pkg.activate(@__DIR__)`, so this notebook uses [notebooks/Project.toml](Project.toml). From the repository root:

```bash
julia --project=notebooks -e 'using Pkg; Pkg.instantiate(); using Pluto; Pluto.run(notebook="notebooks/logposterior_grid_pluto.jl")'
```

Provide **`catalog.h5`** at the repo root (or change `catalog_path` in the config cell). Set **`sample_only`** to a 1- or 2-element `Tuple` of hyperparameter symbols (e.g. `(:H0,)` or `(:H0, :w0)`).
"""

# ╔═╡ 8f4a3b2c-5d6e-4f7a-9b8c-0d1e2f3a4b5c
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
                     ModifiedPropagation
    using AstroSGWBInference: astrosgwb_importance_turing_model, forward_model
    using AstroSGWBImportanceModels:
                                     prepare_bns_madau_dickinson_model
    using Distributions: Uniform
    using Turing
    using Turing: DynamicPPL
    using Random
    using CairoMakie
    using LaTeXStrings
    using LinearAlgebra: BLAS
    BLAS.set_num_threads(1)
end

# ╔═╡ 9a5b4c3d-6e7f-4a8b-ac9d-1e2f3a4b5c6d
begin
    num_threads = Base.Threads.nthreads()
    print(num_threads)
end

# ╔═╡ ab6c5d4e-7f8a-4b9c-bd0e-2f3a4b5c6d7e
begin
    _repo_root = normpath(joinpath(@__DIR__, ".."))

    catalog_path = joinpath(_repo_root, "catalog.h5")
    detectors = [Detector("S1"), Detector("R1")]

    # Change to (:H0,) for a 1-parameter line plot
    sample_only = (:H0, :w0)

    seed = 42
    @info "seeding RNG" rng_seed = seed
    Random.seed!(seed)

    local_merger_rate = 161.0
    observation_time_yr = 1.0

    # Analysis band (Hz). The catalog carries no band information: slice
    # `frequencies` and the rows of `polarization_power` with this cut before computing the
    # effective PSD. Matches the generator band of the production catalog.
    minimum_frequency = 2.0
    maximum_frequency = 4096.0

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
        # `Λ.R₀`, not a prepare-time keyword. The grid cell pins it at this fiducial by
        # conditioning; name it in `sample_only` to free it.
        R₀ = local_merger_rate
    )

    # The prior declares every name, sampled or pinned; the `R₀` entry is the nominal
    # distribution its conditioning pins against.
    hyperprior_dists = (
        H0 = Uniform(20.0, 140.0),
        Ωm = Uniform(0.05, 0.95),
        w0 = Uniform(-3, 1),
        Ξ₀ = Uniform(0.5, 5.0),
        Ξₙ = Uniform(0.05, 3.0),
        γ = Uniform(0.5, 10.0),
        κ = Uniform(0.05, 10.0),
        zpeak = Uniform(0.05, 10.0),
        R₀ = Uniform(10.0, 1000.0)
    )
    hyperprior = hyperprior_dists
end

# ╔═╡ bc7d6e5f-8a9b-4c0d-8e1f-3a4b5c6d7e8f
begin
    @info "loading catalog" catalog_path detectors = join((d.name for d in detectors), ",")
    catalog = load_catalog(catalog_path)
    # Derived from the catalog's `inclination` column; override with an explicit
    # `AnalyticInclination()` / `CatalogInclination()` if needed.
    resolved_average_mode = average_mode(catalog)
    @info "average mode" mode = string(resolved_average_mode)
    C = W0CDM
    P = ModifiedPropagation
    samples = catalog.samples

    # Re-reference the stored EM-distance polarization power to the fiducial GW distance, matching the
    # `+2 log Ξ_fid` term the prepared model's log-weights carry. No-op under Ξ₀ = 1.
    # Out-of-place on purpose: the correction is not idempotent and Pluto re-runs cells.
    polarization_power = apply_gw_distance_correction(
        catalog.polarization_power, catalog.samples.redshift, propagation(P, fiducials))

    # Band selection is the caller's job: restrict to the analysis band before
    # computing the effective PSD, so every bin handed to the model is scored.
    band = (catalog.frequencies .>= minimum_frequency) .&
           (catalog.frequencies .<= maximum_frequency)
    polarization_power = polarization_power[band, :]
    frequencies = catalog.frequencies[band]

    prepared_model = prepare_bns_madau_dickinson_model(samples, fiducials, C, P)
    eff_psd = effective_psd(frequencies, detectors)
    # S2: the prior declares the hyperparameter names; there is no model to ask.
    order = keys(hyperprior_dists)
    @info order
    sample_only_tup = sample_only === nothing ? nothing : Tuple(sample_only)

    @info "catalog loaded" n_frequency_bins=length(frequencies) n_proposal_samples=length(
        samples.redshift,
    )

    @info "using fiducial spectrum from cache as observed data"
    observed = forward_model(
        prepared_model, polarization_power, samples, fiducials;
        average_mode = resolved_average_mode).spectral_density

    nothing
end

# ╔═╡ cd8e7f6a-9b0c-4d1e-9f20-4b5c6d7e8f9a
md"""
## Logposterior grid

Wrap the Turing model with `DynamicPPL.LogDensityFunction` (non-linked ⇒ physical-space logposterior, no Jacobian) and evaluate it on a regular grid over the free-parameter axes defined by `sample_only`.
"""

# ╔═╡ de9f8a7b-0c1d-4e2f-8031-5c6d7e8f9a0b
begin
    # S3: the prior declares every hyperparameter name; conditioning on the complement
    # pins the rest, so the model carries exactly the free parameters by construction.
    # `R₀` is pinned at its fiducial unless named in `sample_only`.
    sampled_prior = sample_only_tup === nothing ?
                    Base.structdiff(hyperprior, (; R₀ = hyperprior.R₀)) :
                    NamedTuple{sample_only_tup}(hyperprior)
    fixed = Base.structdiff(fiducials, sampled_prior)
    model = astrosgwb_importance_turing_model(
        false, resolved_average_mode, prepared_model, polarization_power, samples,
        frequencies, eff_psd, observation_time_yr, hyperprior, observed) | fixed
    lf = DynamicPPL.LogDensityFunction(model)

    free_order = keys(sampled_prior)

    z0 = convert(Vector{Float64}, DynamicPPL.VarInfo(model)[:])
    length(z0) == length(free_order) || error(
        "VarInfo free vector has length $(length(z0)) but free_order has length $(length(free_order)). " *
        "The model's variable layout does not match the expected free_order."
    )
    (1 <= length(free_order) <= 2) || error(
        "this notebook supports 1 or 2 free parameters; got $(length(free_order)). " *
        "Set sample_only to a 1- or 2-element Tuple of symbols in the config cell."
    )

    @info "free parameter space" free_order n_free = length(free_order)

    function loglik_at(values::NamedTuple)
        z = [values[s] for s in free_order]
        return DynamicPPL.LogDensityProblems.logdensity(lf, z)
    end

    # JIT warmup + fiducial sanity check
    let fid_nt = NamedTuple{free_order}(Tuple(fiducials[s] for s in free_order))
        logp_fid = loglik_at(fid_nt)
        @info "logposterior at fiducial" logp_fid
    end
end

# ╔═╡ ef0a9b8c-1d2e-4f3a-9142-6d7e8f9a0b1c
begin
    # Default: full prior range. Override per-parameter to zoom in.
    # Example: grid_bounds[:H0] = (60.0, 80.0)
    grid_bounds = Dict{Symbol, Tuple{Float64, Float64}}()

    n_grid = length(free_order) == 1 ? 121 : 81

    axes_ranges = [let s = free_order[i]
                       lo,
                       hi = get(grid_bounds, s, (
                           hyperprior_dists[s].a, hyperprior_dists[s].b))
                       range(lo, hi; length = n_grid)
                   end
                   for i in eachindex(free_order)]

    @info "grid axes" [free_order[i] => (
                           first(axes_ranges[i]),
                           last(axes_ranges[i]),
                           length(axes_ranges[i])
                       )
                       for i in eachindex(free_order)]
end

# ╔═╡ f0ab0c9d-2e3f-4a4b-a253-7e8f9a0b1c2d
grid_result = let
    n_pts = prod(length(r) for r in axes_ranges)
    @info "evaluating logposterior grid" n_pts
    t_eval = @elapsed result = if length(free_order) == 1
        xs = collect(axes_ranges[1])
        param = free_order[1]
        logp = [loglik_at(NamedTuple{free_order}((x,))) for x in xs]
        ix = argmax(logp)
        @info "1-D grid result" param argmax_x=xs[ix] fiducial_x=fiducials[param]
        (; dim = 1, xs, ys = nothing, logp,
            fid = (fiducials[param],), argmax_pt = (xs[ix],))
    else
        xs = collect(axes_ranges[1])
        ys = collect(axes_ranges[2])
        px, py = free_order[1], free_order[2]
        logp = [loglik_at(NamedTuple{free_order}((x, y))) for x in xs, y in ys]
        imax = argmax(logp)
        ax_val, ay_val = xs[imax[1]], ys[imax[2]]
        @info "2-D grid result" px argmax_x=ax_val fiducial_x=fiducials[px] py argmax_y=ay_val fiducial_y=fiducials[py]
        (; dim = 2, xs, ys, logp, fid = (fiducials[px], fiducials[py]),
            argmax_pt = (ax_val, ay_val))
    end
    @info "grid evaluation complete" elapsed_s = round(t_eval; digits = 1)
    result
end

# ╔═╡ a1bc1d0e-3f4a-4b5c-b364-8f9a0b1c2d3e
md"""
## Plot
"""

# ╔═╡ b2cd2e1f-4a5b-4c6d-8475-9a0b1c2d3e4f
begin
    PARAM_LABELS = Dict{Symbol, LaTeXString}(
        :H0 => L"H_0~[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]",
        :Ωm => L"\Omega_m",
        :w0 => L"w_0",
        :Ξ₀ => L"\Xi_0",
        :Ξₙ => L"\Xi_n",
        :γ => L"\gamma",
        :κ => L"\kappa",
        :zpeak => L"z_\mathrm{peak}"
    )
    param_label(sym) = get(PARAM_LABELS, sym, LaTeXString(string(sym)))

    fig = Figure(size = (720, 520))

    if grid_result.dim == 1
        ax = Axis(
            fig[1, 1];
            xlabel = param_label(free_order[1]),
            ylabel = L"\log p(\theta \mid d)"
        )
        lines!(ax, grid_result.xs, grid_result.logp)
        vlines!(ax, [grid_result.fid[1]];
            color = :red, linestyle = :dash, label = "fiducial")
        axislegend(ax; position = :rt)
    else
        ax = Axis(
            fig[1, 1];
            xlabel = param_label(free_order[1]),
            ylabel = param_label(free_order[2])
        )
        hm = heatmap!(ax, grid_result.xs, grid_result.ys, grid_result.logp)
        Colorbar(fig[1, 2], hm; label = L"\log p(\theta \mid d)")
        contour!(ax, grid_result.xs, grid_result.ys, grid_result.logp;
            levels = 10, color = :white, linewidth = 0.5)
        scatter!(ax, [grid_result.fid[1]], [grid_result.fid[2]];
            color = :white, marker = :star5, markersize = 20,
            strokecolor = :black, strokewidth = 1, label = "fiducial")
        scatter!(ax, [grid_result.argmax_pt[1]], [grid_result.argmax_pt[2]];
            color = :yellow, marker = :circle, markersize = 15,
            strokecolor = :black, strokewidth = 1, label = "grid argmax")
        axislegend(ax; position = :rt)
    end

    fig
end

# ╔═╡ Cell order:
# ╟─7e3f2a1b-4c5d-4e6f-8a7b-9c8d0e1f2a3b
# ╠═9a5b4c3d-6e7f-4a8b-ac9d-1e2f3a4b5c6d
# ╠═8f4a3b2c-5d6e-4f7a-9b8c-0d1e2f3a4b5c
# ╠═ab6c5d4e-7f8a-4b9c-bd0e-2f3a4b5c6d7e
# ╠═bc7d6e5f-8a9b-4c0d-8e1f-3a4b5c6d7e8f
# ╟─cd8e7f6a-9b0c-4d1e-9f20-4b5c6d7e8f9a
# ╠═de9f8a7b-0c1d-4e2f-8031-5c6d7e8f9a0b
# ╠═ef0a9b8c-1d2e-4f3a-9142-6d7e8f9a0b1c
# ╠═f0ab0c9d-2e3f-4a4b-a253-7e8f9a0b1c2d
# ╟─a1bc1d0e-3f4a-4b5c-b364-8f9a0b1c2d3e
# ╠═b2cd2e1f-4a5b-4c6d-8475-9a0b1c2d3e4f
