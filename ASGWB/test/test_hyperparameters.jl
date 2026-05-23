using Test
using ASGWB
using Distributions: product_distribution, Normal, ProductNamedTupleDistribution

@testset "Hyperparameter Validation" begin
    model = MadauDickinsonModifiedPropagation()
    expected_order = (:H0, :Ωm, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)

    # 1. ProductNamedTupleDistribution setup
    dists = (; (k => Normal(0.0, 1.0) for k in expected_order)...)
    prior = product_distribution(dists)

    @testset "validate_subset" begin
        # Valid subsets (tuple)
        @test validate_subset((:H0, :Ωm), model) === (:H0, :Ωm)
        @test validate_subset((:H0, :Ωm), prior) === (:H0, :Ωm)
        @test validate_subset((:H0, :Ωm), expected_order) === (:H0, :Ωm)

        # Empty subset is allowed in validate_subset
        @test validate_subset((), model) === ()
        @test validate_subset((), prior) === ()
        @test validate_subset((), expected_order) === ()

        # Valid subsets (NamedTuple)
        nt = (H0 = 70.0, Ωm = 0.3)
        @test validate_subset(nt, model) === nt
        @test validate_subset(nt, prior) === nt
        @test validate_subset(nt, expected_order) === nt

        # Unknown symbols throw ArgumentError
        @test_throws ArgumentError validate_subset((:invalid_key,), model)
        @test_throws ArgumentError validate_subset((:invalid_key,), prior)
        @test_throws ArgumentError validate_subset((:invalid_key,), expected_order)
        @test_throws ArgumentError validate_subset((invalid_key = 1.0,), model)

        # Repeating/Duplicate symbols throw ArgumentError
        @test_throws ArgumentError validate_subset((:H0, :H0), model)
        @test_throws ArgumentError validate_subset((:H0, :H0), prior)
        @test_throws ArgumentError validate_subset((:H0, :H0), expected_order)
    end

    @testset "validate_hyperparameters" begin
        # Exact match
        ok_nt = (; (k => 1.0 for k in expected_order)...)
        @test validate_hyperparameters(model, ok_nt) === nothing

        # Missing keys
        missing_nt = (H0 = 70.0, Ωm = 0.3)
        @test_throws ArgumentError validate_hyperparameters(model, missing_nt)

        # Extra keys
        extra_nt = (; (k => 1.0 for k in expected_order)..., extra_key = 1.0)
        @test_throws ArgumentError validate_hyperparameters(model, extra_nt)
    end
end

@testset "build_cosmology(ProposalFiducialParameters)" begin
    base = (H0 = 67.0, Ωm = 0.3, Ξ₀ = 1.0, Ξₙ = 0.0)

    lcdm_fid = ProposalFiducialParameters(; base...)
    @test build_cosmology(lcdm_fid) isa LambdaCDM

    w0_fid = ProposalFiducialParameters(; base..., w0 = -0.9)
    @test build_cosmology(w0_fid) isa W0CDM

    cpl_fid = ProposalFiducialParameters(; base..., w0 = -0.9, wa = 0.2)
    @test build_cosmology(cpl_fid) isa W0WaCDM

    bad_fid = ProposalFiducialParameters(; base..., wa = 0.2)
    @test_throws ArgumentError build_cosmology(bad_fid)
end
