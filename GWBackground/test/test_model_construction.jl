using Test
using GWBackground

@testset "model cosmology and propagation constructors" begin
    base = (H0 = 67.0, Ωm = 0.3, Ξ₀ = 1.0, Ξₙ = 0.0, γ = 2.7, κ = 5.7, zpeak = 2.0)
    P = ModifiedPropagation

    @test cosmology(LambdaCDM, base) isa LambdaCDM
    @test propagation(P, base) isa ModifiedPropagation

    Λ_w0 = (; base..., w0 = -0.9)
    @test cosmology(W0CDM, Λ_w0) isa W0CDM

    Λ_cpl = (; base..., w0 = -0.9, wa = 0.2)
    @test cosmology(W0WaCDM, Λ_cpl) isa W0WaCDM
end
