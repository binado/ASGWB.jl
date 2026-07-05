using Test
using AstroSGWB

if !@isdefined ParityBNSPopulation
    include(joinpath(@__DIR__, "fixture_population.jl"))
end

@testset "Hyperparameter Validation" begin
    C, P = LambdaCDM, ModifiedPropagation
    pop = ParityBNSPopulation()
    order = full_hyperparameters(C, P, pop)
    expected_order = (:H0, :Ωm, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
    @test order == expected_order

    @testset "validate_hyperparameters" begin
        ok_nt = (; (k => 1.0 for k in expected_order)...)
        @test validate_hyperparameters(order, ok_nt) === nothing

        missing_nt = (H0 = 70.0, Ωm = 0.3)
        @test_throws ArgumentError validate_hyperparameters(order, missing_nt)

        extra_nt = (; (k => 1.0 for k in expected_order)..., extra_key = 1.0)
        @test_throws ArgumentError validate_hyperparameters(order, extra_nt)
    end

    @testset "hyperparameters W0CDM / W0WaCDM" begin
        @test full_hyperparameters(W0CDM, ModifiedPropagation, ParityBNSPopulation()) ==
              (:H0, :Ωm, :w0, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
        @test full_hyperparameters(W0WaCDM, ModifiedPropagation, ParityBNSPopulation()) ==
              (:H0, :Ωm, :w0, :wa, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
    end

    @testset "GR propagation drops Ξ params" begin
        @test full_hyperparameters(LambdaCDM, GR, ParityBNSPopulation()) ==
              (:H0, :Ωm, :γ, :κ, :zpeak)
    end
end

@testset "model cosmology and propagation from hyperparameters" begin
    base = (H0 = 67.0, Ωm = 0.3, Ξ₀ = 1.0, Ξₙ = 0.0, γ = 2.7, κ = 5.7, zpeak = 2.0)
    P = ModifiedPropagation

    C_lcdm = LambdaCDM
    pop = ParityBNSPopulation()
    order_lcdm = full_hyperparameters(C_lcdm, P, pop)
    Λ_lcdm = canonical_hyperparameters(order_lcdm, base)
    @test cosmology(C_lcdm, Λ_lcdm) isa LambdaCDM
    @test propagation(P, Λ_lcdm) isa ModifiedPropagation

    C_w0 = W0CDM
    order_w0 = full_hyperparameters(C_w0, P, pop)
    Λ_w0 = canonical_hyperparameters(order_w0, (; base..., w0 = -0.9))
    @test cosmology(C_w0, Λ_w0) isa W0CDM

    C_cpl = W0WaCDM
    order_cpl = full_hyperparameters(C_cpl, P, pop)
    Λ_cpl = canonical_hyperparameters(order_cpl, (; base..., w0 = -0.9, wa = 0.2))
    @test cosmology(C_cpl, Λ_cpl) isa W0WaCDM
end
