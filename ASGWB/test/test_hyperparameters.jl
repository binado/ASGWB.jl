using Test
using ASGWB
using Distributions: product_distribution, Normal, ProductNamedTupleDistribution

if !@isdefined ParityBNSPopulation
    include(joinpath(@__DIR__, "fixture_population.jl"))
end

@testset "Hyperparameter Validation" begin
    C = ModifiedPropagation{LambdaCDM}
    pop = ParityBNSPopulation()
    order = full_hyperparameters(C, pop)
    expected_order = (:H0, :Ωm, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
    @test order == expected_order

    # ProductNamedTupleDistribution setup
    dists = (; (k => Normal(0.0, 1.0) for k in expected_order)...)
    prior = product_distribution(dists)

    @testset "validate_subset" begin
        @test validate_subset((:H0, :Ωm), order) === (:H0, :Ωm)
        @test validate_subset((:H0, :Ωm), prior) === (:H0, :Ωm)
        @test validate_subset((:H0, :Ωm), expected_order) === (:H0, :Ωm)

        @test validate_subset((), order) === ()
        @test validate_subset((), prior) === ()
        @test validate_subset((), expected_order) === ()

        nt = (H0 = 70.0, Ωm = 0.3)
        @test validate_subset(nt, order) === nt
        @test validate_subset(nt, prior) === nt
        @test validate_subset(nt, expected_order) === nt

        @test_throws ArgumentError validate_subset((:invalid_key,), order)
        @test_throws ArgumentError validate_subset((:invalid_key,), prior)
        @test_throws ArgumentError validate_subset((:invalid_key,), expected_order)
        @test_throws ArgumentError validate_subset((invalid_key = 1.0,), order)

        @test_throws ArgumentError validate_subset((:H0, :H0), order)
        @test_throws ArgumentError validate_subset((:H0, :H0), prior)
        @test_throws ArgumentError validate_subset((:H0, :H0), expected_order)
    end

    @testset "validate_hyperparameters" begin
        ok_nt = (; (k => 1.0 for k in expected_order)...)
        @test validate_hyperparameters(order, ok_nt) === nothing

        missing_nt = (H0 = 70.0, Ωm = 0.3)
        @test_throws ArgumentError validate_hyperparameters(order, missing_nt)

        extra_nt = (; (k => 1.0 for k in expected_order)..., extra_key = 1.0)
        @test_throws ArgumentError validate_hyperparameters(order, extra_nt)
    end

    @testset "hyperparameters W0CDM / W0WaCDM" begin
        @test full_hyperparameters(ModifiedPropagation{W0CDM}, ParityBNSPopulation()) ==
              (:H0, :Ωm, :w0, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
        @test full_hyperparameters(ModifiedPropagation{W0WaCDM}, ParityBNSPopulation()) ==
              (:H0, :Ωm, :w0, :wa, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
    end
end

@testset "model cosmology from hyperparameters" begin
    base = (H0 = 67.0, Ωm = 0.3, Ξ₀ = 1.0, Ξₙ = 0.0, γ = 2.7, κ = 5.7, zpeak = 2.0)

    C_lcdm = ModifiedPropagation{LambdaCDM}
    pop = ParityBNSPopulation()
    order_lcdm = full_hyperparameters(C_lcdm, pop)
    Λ_lcdm = canonical_hyperparameters(order_lcdm, base)
    @test cosmology(C_lcdm, Λ_lcdm) isa ModifiedPropagation{<:LambdaCDM}

    C_w0 = ModifiedPropagation{W0CDM}
    order_w0 = full_hyperparameters(C_w0, pop)
    Λ_w0 = canonical_hyperparameters(order_w0, (; base..., w0 = -0.9))
    @test cosmology(C_w0, Λ_w0) isa ModifiedPropagation{<:W0CDM}

    C_cpl = ModifiedPropagation{W0WaCDM}
    order_cpl = full_hyperparameters(C_cpl, pop)
    Λ_cpl = canonical_hyperparameters(order_cpl, (; base..., w0 = -0.9, wa = 0.2))
    @test cosmology(C_cpl, Λ_cpl) isa ModifiedPropagation{<:W0WaCDM}
end
