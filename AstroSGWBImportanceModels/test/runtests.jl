using Test
using AstroSGWB
using AstroSGWBImportanceModels
using AstroSGWBInference
using Cosmology
using Distributions: Uniform, product_distribution
using ForwardDiff
using Turing

const FIDUCIALS = (
    H0 = 67.0,
    Ωm = 0.315,
    Ξ₀ = 1.0,
    Ξₙ = 0.0,
    γ = 2.7,
    κ = 3.0,
    zpeak = 2.5
)

const TARGET = (
    H0 = 70.0,
    Ωm = 0.3,
    Ξ₀ = 1.1,
    Ξₙ = 0.2,
    γ = 2.9,
    κ = 3.1,
    zpeak = 2.2
)

const SAMPLES = (
    redshift = [0.1, 0.2],
    luminosity_distance = [430.0, 880.0]
)

function prepared(samples = SAMPLES; C = LambdaCDM, P = ModifiedPropagation,
        fiducials = FIDUCIALS)
    return prepare_bns_madau_dickinson_model(
        samples,
        fiducials,
        C,
        P;
        local_merger_rate = 161.0,
        observation_time = 1.0
    )
end

@testset "BNS Madau–Dickinson hyperparameters" begin
    @test bns_madau_dickinson_hyperparameters(LambdaCDM, GR) ==
          (:H0, :Ωm, :γ, :κ, :zpeak)
    @test bns_madau_dickinson_hyperparameters(LambdaCDM, ModifiedPropagation) ==
          (:H0, :Ωm, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
    @test bns_madau_dickinson_hyperparameters(W0CDM, GR) ==
          (:H0, :Ωm, :w0, :γ, :κ, :zpeak)
    @test bns_madau_dickinson_hyperparameters(W0CDM, ModifiedPropagation) ==
          (:H0, :Ωm, :w0, :Ξ₀, :Ξₙ, :γ, :κ, :zpeak)
    @test AstroSGWBInference.hyperparameters(prepared()) ==
          bns_madau_dickinson_hyperparameters(LambdaCDM, ModifiedPropagation)
end

@testset "catalog sample adaptation" begin
    stored = (
        redshift = [0.1, 0.2],
        luminosity_distance = [12.0, 34.0],
        unused = [1, 2]
    )
    adapted = bns_samples_from_catalog(stored, LambdaCDM, FIDUCIALS)
    @test adapted == (redshift = [0.1, 0.2], luminosity_distance = [12.0, 34.0])
    @test adapted.redshift !== stored.redshift
    @test adapted.luminosity_distance !== stored.luminosity_distance

    without_distance = (redshift = [0.1, 0.2], unused = [1, 2])
    synthesized = bns_samples_from_catalog(without_distance, LambdaCDM, FIDUCIALS)
    expected = luminosity_distance.(
        without_distance.redshift, Ref(cosmology(LambdaCDM, FIDUCIALS)))
    @test synthesized.redshift == without_distance.redshift
    @test synthesized.luminosity_distance ≈ expected
    @test all(isfinite, synthesized.luminosity_distance)
    @test all(>(0), synthesized.luminosity_distance)
end

@testset "preparation caches and fixed-fixture parity" begin
    model = prepared()
    @test model isa BNSMadauDickinsonImportanceModel{LambdaCDM, ModifiedPropagation}
    @test model.z_grid isa Vector{Float64}
    @test model.proposal_log_pdf isa Vector{Float64}
    @test model.local_merger_rate === 161.0
    @test model.observation_time === 1.0
    @test length(model.z_grid) == length(DEFAULT_Z_GRID)
    @test length(model.proposal_log_pdf) == length(SAMPLES.redshift)
    @test all(isfinite, model.proposal_log_pdf)
    @test model.proposal_log_pdf ≈ [-6.274110509399128, -4.919648956007439]

    rate, log_weights = merger_rate_and_log_weights(model, TARGET, SAMPLES)
    @test rate ≈ 0.031115713391297647 rtol = 1.0e-13
    @test log_weights ≈ [-0.07381242172386923, -0.14770006729772409] rtol = 1.0e-12
    @test size(log_weights) == size(SAMPLES.redshift)
    @test all(isfinite, log_weights)
end

@testset "ForwardDiff empty and one-sample evaluations" begin
    dual(x) = ForwardDiff.Dual{Nothing}(x, one(x))
    Λ_dual = NamedTuple{keys(FIDUCIALS)}(map(dual, values(FIDUCIALS)))
    empty_samples = (redshift = Float64[], luminosity_distance = Float64[])
    one_sample = (redshift = [0.1], luminosity_distance = [500.0])

    empty_model = prepared(empty_samples)
    one_model = prepared(one_sample)
    empty_rate,
    empty_weights = merger_rate_and_log_weights(
        empty_model, Λ_dual, empty_samples)
    one_rate, one_weights = merger_rate_and_log_weights(one_model, Λ_dual, one_sample)

    @test isfinite(empty_rate)
    @test isfinite(one_rate)
    @test isempty(empty_weights)
    @test eltype(empty_weights) <: ForwardDiff.Dual
    @test length(one_weights) == 1
    @test all(isfinite, one_weights)
    @test eltype(one_weights) <: ForwardDiff.Dual
end

@testset "concrete adapter integrates with Turing" begin
    model = prepared()
    fluxes = Float64[0.0 0.0; 1.0 1.5; 2.0 2.5]
    observation = ObservationContext(
        [0.0, 20.0, 40.0],
        [Inf, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        BitVector([false, true, true]),
        1.0
    )
    prior = product_distribution((
        H0 = Uniform(20.0, 140.0),
        Ωm = Uniform(0.05, 0.95),
        Ξ₀ = Uniform(0.5, 5.0),
        Ξₙ = Uniform(0.0, 3.0),
        γ = Uniform(0.5, 10.0),
        κ = Uniform(0.05, 10.0),
        zpeak = Uniform(0.05, 10.0)
    ))
    turing_model = build_turing_model(
        model, fluxes, SAMPLES, FIDUCIALS, observation, prior)

    @test turing_model !== nothing
    @test isfinite(Turing.logjoint(turing_model, FIDUCIALS))
end
