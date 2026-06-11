### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ a1b2c3d4-0001-4e5f-9a0b-1c2d3e4f5a6b
begin
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
    using ASGWB:
                 PopulationModel,
                 CosmologyCache,
                 LambdaCDM,
                 cosmology,
                 OrderedUniformSourceMassPair,
                 AlignedSpinChiSimple,
                 MadauDickinsonSourceFrame,
                 redshift_prior,
                 luminosity_distance
    using Distributions: Uniform, product_distribution, ProductNamedTupleDistribution
    using DataFrames
    using CSV
    using CairoMakie
    using Random
    using Dates: now, format
end

# ╔═╡ a1b2c3d4-0002-4e5f-9a0b-1c2d3e4f5a6b
md"""
# Drawing samples for a population model

In this notebook, we define a population model for binary neutron star (BNS) assuming
- Uniform prior on the masses (with mass ordering constraint)
- Aligned spin prior
- Uniform prior on tidal deformabilities
- Redshift prior proportional to SFR for Madau-Dickinson like curve
"""

# ╔═╡ 3bdc27d4-f478-45df-951d-67d4deae22a7
md"""
## Configuration cell

Change the cosmology or sampling settings here.
"""

# ╔═╡ a1b2c3d4-0003-4e5f-9a0b-1c2d3e4f5a6b
begin
    # ---- All configuration is inlined here, immediately after the imports ----

    seed = 42
    n_samples = 100_000

    # Cosmology family used to build the detector-frame redshift prior.
    C = LambdaCDM

    # Full hyperparameter vector. Order: cosmology, then population.
    Λ = (;
        # Cosmology (LambdaCDM reads H0, Ωm)
        H0 = 67.66,
        Ωm = 0.3096,
        # Madau–Dickinson source-frame SFR
        γ = 2.7,
        κ = 5.7,
        zpeak = 2.0,
        # Promoted population bounds
        m_low = 1.1,        # BNS_MASS_LOW
        m_high = 2.5,       # BNS_MASS_HIGH
        a_max = 0.99,       # BNS_SPIN_A_MAX
        lambda_max = 5000.0 # BNS_LAMBDA_HIGH
    )

    # Internal sample key → bilby injection-file column name. Also fixes the
    # column/panel order. χ/Λ symbols are renamed to bilby's ASCII conventions.
    bilby_names = (
        mass_1_source = :mass_1_source,
        mass_2_source = :mass_2_source,
        redshift = :redshift,
        luminosity_distance = :luminosity_distance,
        χ₁ = :chi_1,
        χ₂ = :chi_2,
        Λ₁ = :lambda_1,
        Λ₂ = :lambda_2
    )

    # Per-parameter Makie x-axis scale; absent keys default to `identity`.
    xscales = (luminosity_distance = log10,)

    output_dir = joinpath(@__DIR__, "output")
    timestamp = format(now(), "yyyymmdd-HHMMSS")
    output_file = "bns-prior-injections-seed$(seed)-$(timestamp).dat"
end

# ╔═╡ 607800d6-5c09-43f4-b268-8e56192173c1
md"""
## Defining the model

All we need to do to define a population model is to create a struct which subtypes `PopulationModel` and defines two methods:
- a `hyperparameters(model)` method which returns a tuple of symbols of the hyperparameters of the model
- a `single_event_prior(model, cosmology, Λ)` method which, for a given hyperparameter vector Λ, returns p(θ | Λ, cosmology)
"""

# ╔═╡ a1b2c3d4-0004-4e5f-9a0b-1c2d3e4f5a6b
begin
    import ASGWB: single_event_prior, hyperparameters

    struct BNSUniformMassAlignedSpinTidalSFR <: PopulationModel end

    hyperparameters(::BNSUniformMassAlignedSpinTidalSFR) =
        (:γ, :κ, :zpeak, :m_low, :m_high, :a_max, :lambda_max)

    function single_event_prior(
            ::BNSUniformMassAlignedSpinTidalSFR,
            cache::CosmologyCache,
            Λ::NamedTuple
    )
        z_d = redshift_prior(MadauDickinsonSourceFrame(), cache, Λ)
        spin = AlignedSpinChiSimple(a_max = Λ.a_max)
        return product_distribution((
            mass = OrderedUniformSourceMassPair(low = Λ.m_low, high = Λ.m_high),
            redshift = z_d,
            χ₁ = spin,
            χ₂ = spin,
            Λ₁ = Uniform(0.0, Λ.lambda_max),
            Λ₂ = Uniform(0.0, Λ.lambda_max)
        ))
    end

    pop = BNSUniformMassAlignedSpinTidalSFR()
end

# ╔═╡ a1b2c3d4-0008-4e5f-9a0b-1c2d3e4f5a6b
"""
    sample_columns(d::ProductNamedTupleDistribution, n) -> NamedTuple of arrays

Vectorized columnar draw: sample each component of `d` as a batch via Distributions'
`rand(component, n)`, returning a `NamedTuple` of column arrays keyed by component name.
Multivariate components (e.g. `mass`) come back as `(length, n)` matrices.

This is the fast path for bulk sampling: it writes directly into typed contiguous arrays
and avoids the 100k per-event `NamedTuple`s (and per-event `mass` vectors) that
`rand(d, (n,))` allocates.
"""
sample_columns(d::ProductNamedTupleDistribution, n::Integer) =
    map(component -> rand(component, n), d.dists)

# ╔═╡ 0feb8a65-c358-4449-828d-78198f26d18d
md"""
## Drawing samples from the population model
"""

# ╔═╡ a1b2c3d4-0005-4e5f-9a0b-1c2d3e4f5a6b
begin
    Random.seed!(seed)
    cosmo = cosmology(C, Λ)
    prior = single_event_prior(pop, cosmo, Λ)

    @info "drawing prior samples" n_samples seed model = nameof(typeof(pop))
    # `cols` is a NamedTuple keyed like the prior: `mass` is a (2, n) matrix
    # (rows m1 ≥ m2, source frame); the rest are length-n vectors.
    cols = sample_columns(prior, n_samples)
    redshift = cols.redshift

    # Columns are keyed by the notebook's internal names (renamed to bilby on save).
    samples = DataFrame(
        mass_1_source = cols.mass[1, :],
        mass_2_source = cols.mass[2, :],
        redshift = redshift,
        # Derived: luminosity distance in Mpc (matches astropy/bilby convention).
        luminosity_distance = luminosity_distance.(redshift, Ref(cosmo)),
        χ₁ = cols.χ₁,
        χ₂ = cols.χ₂,
        Λ₁ = cols.Λ₁,
        Λ₂ = cols.Λ₂
    )

    @info "drew samples" rows = nrow(samples)
    samples
end

# ╔═╡ b9387660-bdb9-4513-9745-853ea486ad9d
md"""
## Visualizing the distributions of intrinsic parameters
"""

# ╔═╡ a1b2c3d4-0006-4e5f-9a0b-1c2d3e4f5a6b
begin
    fig = Figure(size = (1100, 800))
    for (idx, param) in enumerate(keys(bilby_names))
        row = (idx - 1) ÷ 2 + 1
        colpos = (idx - 1) % 2 + 1
        col = samples[!, param]
        xscale = get(xscales, param, identity)
        ax = Axis(fig[row, colpos]; title = string(param), xscale = xscale,
            ylabel = "density")
        if xscale === identity
            density!(ax, col)
        else
            # Clamp KDE support to the (positive) data range; otherwise the kernel
            # extends below the minimum and log10 hits negative arguments.
            density!(ax, col; boundary = extrema(col))
        end
    end
    fig
end

# ╔═╡ d8baac65-57f4-432c-8442-0b858c6f7b54
md"""
## Saving the samples to a file
"""

# ╔═╡ a1b2c3d4-0007-4e5f-9a0b-1c2d3e4f5a6b
begin
    mkpath(output_dir)
    output_path = joinpath(output_dir, output_file)

    # Rename internal columns to bilby conventions
    output_df = rename(samples, pairs(bilby_names)...)
    CSV.write(output_path, output_df; delim = ' ')
    @info "wrote injection file" path = abspath(output_path) rows = nrow(output_df)
    output_path
end

# ╔═╡ Cell order:
# ╠═a1b2c3d4-0002-4e5f-9a0b-1c2d3e4f5a6b
# ╠═a1b2c3d4-0001-4e5f-9a0b-1c2d3e4f5a6b
# ╠═3bdc27d4-f478-45df-951d-67d4deae22a7
# ╠═a1b2c3d4-0003-4e5f-9a0b-1c2d3e4f5a6b
# ╠═607800d6-5c09-43f4-b268-8e56192173c1
# ╠═a1b2c3d4-0004-4e5f-9a0b-1c2d3e4f5a6b
# ╠═a1b2c3d4-0008-4e5f-9a0b-1c2d3e4f5a6b
# ╠═0feb8a65-c358-4449-828d-78198f26d18d
# ╠═a1b2c3d4-0005-4e5f-9a0b-1c2d3e4f5a6b
# ╠═b9387660-bdb9-4513-9745-853ea486ad9d
# ╠═a1b2c3d4-0006-4e5f-9a0b-1c2d3e4f5a6b
# ╠═d8baac65-57f4-432c-8442-0b858c6f7b54
# ╠═a1b2c3d4-0007-4e5f-9a0b-1c2d3e4f5a6b
