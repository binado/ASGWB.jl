using Test
using Distributions
using CBCDistributions

@testset "PopulationModel interface — TestPop" begin
    pop = TestPop()

    hp = population_hyperprior(pop)
    @test hp isa Distributions.ProductNamedTupleDistribution
    @test keys(hp.dists) == (:α, :β)

    cosmo = LambdaCDM(67.0, 0.315)
    Λ = (H0 = 67.0, Ωm = 0.315, α = 0.5, β = 1.5)
    sep = single_event_prior(pop, cosmo, Λ)
    @test sep isa Distributions.ProductNamedTupleDistribution
    @test :x in keys(sep.dists)
    @test :y in keys(sep.dists)
end

@testset "PopulationModel threads the caller's redshift grid" begin
    cosmo = LambdaCDM(67.0, 0.315)
    Λ = (γ = 2.7, κ = 3.0, zpeak = 2.5)
    z_grid = collect(LinRange(0.0, 3.0, 61))
    prior = single_event_prior(TestRedshiftPop(), cosmo, Λ; z_grid)
    @test minimum(prior.dists.redshift) == first(z_grid)
    @test maximum(prior.dists.redshift) == last(z_grid)
end

@testset "caller-owned hyperprior composition" begin
    pop = TestPop()

    hp = merge_hyperpriors(
        cosmology_hyperprior(LambdaCDM),
        propagation_hyperprior(ModifiedPropagation),
        population_hyperprior(pop)
    )
    @test hp isa Distributions.ProductNamedTupleDistribution
    @test keys(hp.dists) == (:H0, :Ωm, :Ξ₀, :Ξₙ, :α, :β)
end

@testset "cosmology_hyperprior for cosmology types" begin
    hp_lcdm = cosmology_hyperprior(LambdaCDM)
    @test keys(hp_lcdm.dists) == (:H0, :Ωm)

    hp_w0 = cosmology_hyperprior(W0CDM)
    @test keys(hp_w0.dists) == (:H0, :Ωm, :w0)

    hp_cpl = cosmology_hyperprior(W0WaCDM)
    @test keys(hp_cpl.dists) == (:H0, :Ωm, :w0, :wa)
end

@testset "propagation_hyperprior for propagation types" begin
    hp_mod = propagation_hyperprior(ModifiedPropagation)
    @test keys(hp_mod.dists) == (:Ξ₀, :Ξₙ)
end

@testset "canonical_hyperparameters" begin
    order = (:H0, :Ωm, :α, :β)
    Λ_unordered = (α = 0.5f0, H0 = 67.0, β = 1.5, Ωm = 0.315)

    Λc = canonical_hyperparameters(order, Λ_unordered)
    @test keys(Λc) == order
    @test Λc.H0 isa Float64
    @test Λc.α isa Float64

    Λc_raw = canonical_hyperparameters(order, Λ_unordered; eltype = nothing)
    @test Λc_raw.H0 === 67.0
    @test Λc_raw.α === 0.5f0

    @test_throws ArgumentError canonical_hyperparameters(order, (H0 = 1.0,))
    @test_throws ArgumentError canonical_hyperparameters(order, (;
        (k => 1.0 for k in order)..., extra = 0.0))
end

@testset "validate_hyperparameters" begin
    order = (:H0, :Ωm)
    ok = (H0 = 67.0, Ωm = 0.315)
    @test validate_hyperparameters(order, ok) === nothing

    @test_throws ArgumentError validate_hyperparameters(order, (Ωm = 0.315, H0 = 67.0))
    @test_throws ArgumentError validate_hyperparameters(order, (H0 = 67.0,))
end
