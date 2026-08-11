using Test
using AstroSGWB

@testset "caller-owned hyperparameter validation" begin
    expected_order = (:H0, :Ωm, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)

    ok_nt = (; (k => 1.0 for k in expected_order)...)
    @test validate_hyperparameters(expected_order, ok_nt) === nothing

    missing_nt = (H0 = 70.0, Ωm = 0.3)
    @test_throws ArgumentError validate_hyperparameters(expected_order, missing_nt)

    extra_nt = (; (k => 1.0 for k in expected_order)..., extra_key = 1.0)
    @test_throws ArgumentError validate_hyperparameters(expected_order, extra_nt)
end

@testset "model cosmology and propagation from hyperparameters" begin
    base = (H0 = 67.0, Ωm = 0.3, Ξ₀ = 1.0, Ξₙ = 0.0, γ = 2.7, κ = 5.7, zpeak = 2.0)
    P = ModifiedPropagation

    @test cosmology(LambdaCDM, base) isa LambdaCDM
    @test propagation(P, base) isa ModifiedPropagation

    Λ_w0 = (; base..., w0 = -0.9)
    @test cosmology(W0CDM, Λ_w0) isa W0CDM

    Λ_cpl = (; base..., w0 = -0.9, wa = 0.2)
    @test cosmology(W0WaCDM, Λ_cpl) isa W0WaCDM
end
