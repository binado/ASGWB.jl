### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ a1b2c3d4-0001-4e5f-9a0b-1c2d3e4f5a6b
begin
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
    using AstroSGWB:
                     DEFAULT_Z_GRID,
                     LambdaCDM,
                     cosmology,
                     OrderedUniformSourceMassPair,
                     AlignedSpinChiSimple,
                     MadauDickinsonSourceFrame,
                     redshift_prior,
                     luminosity_distance
    using AstroSGWBCosmology: AbstractCosmology
    using AstroSGWBDistributions: DefaultBBHMassPair
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

This notebook draws source-frame injection samples from either a BNS or BBH
population model.
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
    nsamples = 100_000
    # Source model: BNS or BBH
    source_model = :BBH

    # Cosmology family used to build the detector-frame redshift prior.
    C = LambdaCDM

    # Per-parameter Makie x-axis scale; absent keys default to `identity`.
    xscales = (luminosity_distance = log10,)

    output_dir = joinpath(@__DIR__, "output")
    timestamp = format(now(), "yyyymmdd-HHMMSS")
end

# ╔═╡ 607800d6-5c09-43f4-b268-8e56192173c1
md"""
## Defining the model

Define a function returning a product distribution for each source population. For a given
flat hyperparameter state ``Λ``, it returns ``p(\theta | \Lambda,~\textrm{cosmo})``.
The caller-owned hyperprior declares the names in ``Λ``.
"""

# ╔═╡ a1b2c3d4-0004-4e5f-9a0b-1c2d3e4f5a6b
begin

    function bns_prior(
            cosmo::AbstractCosmology,
            Λ::NamedTuple;
            z_grid::AbstractVector{<:Real} = DEFAULT_Z_GRID
    )
        z_d = redshift_prior(
            MadauDickinsonSourceFrame(γ = Λ.γ, κ = Λ.κ, zpeak = Λ.zpeak, R₀ = Λ.R₀),
            cosmo; z_grid)
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

    function bbh_prior(
            cosmo::AbstractCosmology,
            Λ::NamedTuple;
            z_grid::AbstractVector{<:Real} = DEFAULT_Z_GRID
    )
        z_d = redshift_prior(
            MadauDickinsonSourceFrame(γ = Λ.γ, κ = Λ.κ, zpeak = Λ.zpeak, R₀ = Λ.R₀),
            cosmo; z_grid)
        spin = AlignedSpinChiSimple(a_max = Λ.a_max)
        return product_distribution((
            mass = DefaultBBHMassPair(;
                α1 = Λ.α1,
                α2 = Λ.α2,
                m_break = Λ.m_break,
                μ1 = Λ.μ1,
                σ1 = Λ.σ1,
                μ2 = Λ.μ2,
                σ2 = Λ.σ2,
                m1_low = Λ.m1_low,
                δm1 = Λ.δm1,
                λ0 = Λ.λ0,
                λ1 = Λ.λ1,
                βq = Λ.βq,
                m2_low = Λ.m2_low,
                δm2 = Λ.δm2,
                m_high = Λ.m_high
            ),
            redshift = z_d,
            χ₁ = spin,
            χ₂ = spin
        ))
    end
end

# ╔═╡ d4a12d20-e275-4c39-a2df-1bbbc3e7d048
begin
    population_model(::Val{:BNS}) = bns_prior
    population_model(::Val{:BBH}) = bbh_prior

    function hyperparameter_values(::Val{:BNS})
        return (;
            H0 = 67.66,
            Ωm = 0.3096,
            γ = 2.7,
            κ = 3.0,
            zpeak = 2.0,
            R₀ = 161.0,
            m_low = 1.1,
            m_high = 2.5,
            a_max = 0.99,
            lambda_max = 5000.0
        )
    end

    function hyperparameter_values(::Val{:BBH})
        return (;
            H0 = 67.66,
            Ωm = 0.3096,
            γ = 2.7,
            κ = 3.0,
            zpeak = 2.0,
            R₀ = 161.0,
            α1 = 1.5,
            α2 = 5.4,
            m_break = 37.5,
            μ1 = 12.5,
            σ1 = 5.0,
            μ2 = 33,
            σ2 = 3.1,
            m1_low = 6.5,
            δm1 = 5.0,
            λ0 = 1 / 3,
            λ1 = 1 / 3,
            βq = 1.0,
            m2_low = 4.75,
            δm2 = 5.0,
            m_high = 300.0,
            a_max = 0.99
        )
    end

    function bilby_column_names(::Val{:BNS})
        return (
            mass_1_source = :mass_1_source,
            mass_2_source = :mass_2_source,
            redshift = :redshift,
            luminosity_distance = :luminosity_distance,
            χ₁ = :chi_1,
            χ₂ = :chi_2,
            Λ₁ = :lambda_1,
            Λ₂ = :lambda_2
        )
    end

    function bilby_column_names(::Val{:BBH})
        return (
            mass_1_source = :mass_1_source,
            mass_2_source = :mass_2_source,
            redshift = :redshift,
            luminosity_distance = :luminosity_distance,
            χ₁ = :chi_1,
            χ₂ = :chi_2
        )
    end

    output_prefix(::Val{:BNS}) = "bns-prior-injections"
    output_prefix(::Val{:BBH}) = "bbh-prior-injections"

    function sample_dataframe(::Val{:BNS}, cols, redshift, cosmo)
        return DataFrame(
            mass_1_source = cols.mass[1, :],
            mass_2_source = cols.mass[2, :],
            redshift = redshift,
            luminosity_distance = luminosity_distance.(redshift, Ref(cosmo)),
            χ₁ = cols.χ₁,
            χ₂ = cols.χ₂,
            Λ₁ = cols.Λ₁,
            Λ₂ = cols.Λ₂
        )
    end

    function sample_dataframe(::Val{:BBH}, cols, redshift, cosmo)
        return DataFrame(
            mass_1_source = cols.mass[1, :],
            mass_2_source = cols.mass[2, :],
            redshift = redshift,
            luminosity_distance = luminosity_distance.(redshift, Ref(cosmo)),
            χ₁ = cols.χ₁,
            χ₂ = cols.χ₂
        )
    end
end

# ╔═╡ e87ef1f8-aeb3-4668-a70f-9e3b6d1c65db
begin
    source_model in (:BNS, :BBH) ||
        throw(ArgumentError("source_model must be either :BNS or :BBH"))
    source = Val(source_model)
    pop = population_model(source)
    Λ = hyperparameter_values(source)
    bilby_names = bilby_column_names(source)
    output_file = "$(output_prefix(source))-seed$(seed)-$(timestamp).dat"
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
function sample_columns(d::ProductNamedTupleDistribution, n::Integer)
    map(component -> rand(component, n), d.dists)
end

# ╔═╡ 0feb8a65-c358-4449-828d-78198f26d18d
md"""
## Drawing samples from the population model
"""

# ╔═╡ a1b2c3d4-0005-4e5f-9a0b-1c2d3e4f5a6b
begin
    Random.seed!(seed)
    cosmo = cosmology(C, Λ)
    prior = pop(cosmo, Λ)

    @info "drawing prior samples" nsamples seed model = source_model
    # `cols` is a NamedTuple keyed like the prior: `mass` is a (2, n) matrix
    # (rows m1 ≥ m2, source frame); the rest are length-n vectors.
    cols = sample_columns(prior, nsamples)
    redshift = cols.redshift

    # Columns are keyed by the notebook's internal names (renamed to bilby on save).
    samples = sample_dataframe(source, cols, redshift, cosmo)

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
    @info "wrote injection file" path=abspath(output_path) rows=nrow(output_df)
    output_path
end

# ╔═╡ Cell order:
# ╠═a1b2c3d4-0002-4e5f-9a0b-1c2d3e4f5a6b
# ╠═a1b2c3d4-0001-4e5f-9a0b-1c2d3e4f5a6b
# ╠═3bdc27d4-f478-45df-951d-67d4deae22a7
# ╠═a1b2c3d4-0003-4e5f-9a0b-1c2d3e4f5a6b
# ╠═607800d6-5c09-43f4-b268-8e56192173c1
# ╠═a1b2c3d4-0004-4e5f-9a0b-1c2d3e4f5a6b
# ╠═d4a12d20-e275-4c39-a2df-1bbbc3e7d048
# ╠═e87ef1f8-aeb3-4668-a70f-9e3b6d1c65db
# ╠═a1b2c3d4-0008-4e5f-9a0b-1c2d3e4f5a6b
# ╠═0feb8a65-c358-4449-828d-78198f26d18d
# ╠═a1b2c3d4-0005-4e5f-9a0b-1c2d3e4f5a6b
# ╠═b9387660-bdb9-4513-9745-853ea486ad9d
# ╠═a1b2c3d4-0006-4e5f-9a0b-1c2d3e4f5a6b
# ╠═d8baac65-57f4-432c-8442-0b858c6f7b54
# ╠═a1b2c3d4-0007-4e5f-9a0b-1c2d3e4f5a6b
