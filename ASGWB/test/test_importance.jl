using ASGWB
using ForwardDiff
using Test

if !@isdefined parity_cache_path
    include(joinpath(@__DIR__, "parity_test_cache.jl"))
end
include(joinpath(@__DIR__, "parity_fixtures.jl"))

function _importance_type_test_problem(n::Integer)
    samples = (
        mass = stack_source_masses(fill(1.4, n), fill(1.2, n)),
        redshift = fill(0.1, n),
        χ₁ = fill(0.0, n),
        χ₂ = fill(0.0, n),
        Λ₁ = fill(100.0, n),
        Λ₂ = fill(100.0, n)
    )
    proposal = ProposalData(
        FULL_BNS_INTRINSIC_ORDER,
        samples,
        zeros(n),
        zeros(n, length(FULL_BNS_INTRINSIC_ORDER)),
        zeros(2, n),
        ones(n)
    )
    observation = ObservationConfig(
        [1.0, 2.0],
        [1.0, 1.0],
        [1.0, 1.0],
        BitVector([true, true]),
        [0.0, 0.0],
        1.0,
        1.0
    )
    spec = RedshiftPriorSpec(MadauDickinson, 0.001, 1.0, 32, nothing)
    fid = ProposalFiducialParameters(;
        H0 = 67.0,
        Ωm = 0.315,
        Ξ₀ = 1.0,
        Ξₙ = 0.0,
        γ = 2.7,
        κ = 3.0,
        zpeak = 2.5
    )
    return importance_sampling_problem(proposal, observation, spec, 1.0, fid)
end

@testset "importance smoke" begin
    cache = load_cache(parity_cache_path(:posterior), [Detector("H1"), Detector("L1")])
    theta = PARITY_THETA

    model_evaluation = evaluate_model_terms(
        MadauDickinsonModifiedPropagation(),
        theta,
        cache
    )
    model_evaluation_with_grid = evaluate_model_terms(
        MadauDickinsonModifiedPropagation(),
        theta,
        cache,
        cache.redshift_cache.redshift_grid
    )
    @test length(model_evaluation.weights) == length(cache.proposal.samples.redshift)
    @test all(isfinite, model_evaluation.weights)
    @test all(isfinite, model_evaluation.log_ratio)
    @test all(isfinite, model_evaluation.target_log_prob)
    @test all(isfinite, model_evaluation.spectral_density)
    @test isfinite(model_evaluation.redshift_integral)
    @test isfinite(model_evaluation.expected_number_of_sources)
    @test model_evaluation_with_grid.spectral_density ≈ model_evaluation.spectral_density

    cosmology_cache,
    redshift_prior = cosmology_and_redshift_prior(
        cosmology(MadauDickinsonModifiedPropagation(), theta),
        theta,
        cache.redshift_prior_spec,
        cache.redshift_cache.redshift_grid
    )
    iw = compute_importance_weights(cache, theta, cosmology_cache, redshift_prior)
    @test iw.weights ≈ model_evaluation.weights
    @test iw.log_ratio ≈ model_evaluation.log_ratio
    @test iw.target_log_prob ≈ model_evaluation.target_log_prob
    @test iw.dgw_theta_sq ≈ model_evaluation.dgw_theta_sq

    rate = merger_rate_per_sec(
        redshift_prior,
        cache.local_merger_rate,
        cache.observation.observation_time_yr,
        cache.observation.observation_time_sec
    )
    @test isfinite(rate)
    @test model_evaluation.expected_number_of_sources ≈
          rate * cache.observation.observation_time_sec
end

@testset "empty importance weights preserve AD element types" begin
    dual(x) = ForwardDiff.Dual{Nothing}(x, one(x))
    theta = (
        H0 = dual(67.0),
        Ωm = dual(0.315),
        Ξ₀ = dual(1.0),
        Ξₙ = dual(0.0),
        γ = dual(2.7),
        κ = dual(3.0),
        zpeak = dual(2.5)
    )

    empty_problem = _importance_type_test_problem(0)
    populated_problem = _importance_type_test_problem(1)
    cosmology_dual = cosmology(MadauDickinsonModifiedPropagation(), theta)
    empty_cosmology_cache,
    empty_redshift_prior = cosmology_and_redshift_prior(
        cosmology_dual,
        theta,
        empty_problem.redshift_prior_spec,
        empty_problem.redshift_cache.redshift_grid
    )
    populated_cosmology_cache,
    populated_redshift_prior = cosmology_and_redshift_prior(
        cosmology_dual,
        theta,
        populated_problem.redshift_prior_spec,
        populated_problem.redshift_cache.redshift_grid
    )

    empty_iw = compute_importance_weights(
        empty_problem, theta, empty_cosmology_cache, empty_redshift_prior)
    populated_iw = compute_importance_weights(
        populated_problem, theta, populated_cosmology_cache, populated_redshift_prior)

    @test isempty(empty_iw.weights)
    @test eltype(empty_iw.weights) == eltype(populated_iw.weights)
    @test eltype(empty_iw.log_ratio) == eltype(populated_iw.log_ratio)
    @test eltype(empty_iw.target_log_prob) == eltype(populated_iw.target_log_prob)
    @test eltype(empty_iw.dgw_theta_sq) == eltype(populated_iw.dgw_theta_sq)
end
