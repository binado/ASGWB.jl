using ASGWB
using CBCDistributions: SampleInterpolant
using ForwardDiff
using Test

if !@isdefined parity_catalog_dir
    include(joinpath(@__DIR__, "parity_test_cache.jl"))
end
include(joinpath(@__DIR__, "parity_fixtures.jl"))

const _IMP_C = ModifiedPropagation{LambdaCDM}
const _IMP_POP = ParityBNSPopulation()
const _IMP_ORDER = full_hyperparameters(_IMP_C, _IMP_POP)

# Construct a problem + ModelContext directly (no catalog/detectors) for type/edge tests.
function _importance_type_test_fixture(n::Integer)
    samples = (
        mass = stack_source_masses(fill(1.4, n), fill(1.2, n)),
        redshift = fill(0.1, n),
        χ₁ = fill(0.0, n),
        χ₂ = fill(0.0, n),
        Λ₁ = fill(100.0, n),
        Λ₂ = fill(100.0, n)
    )
    Λ = canonical_hyperparameters(
        _IMP_ORDER,
        (H0 = 67.0, Ωm = 0.315, Ξ₀ = 1.0, Ξₙ = 0.0, γ = 2.7, κ = 3.0, zpeak = 2.5)
    )
    problem = ImportanceSamplingProblem(_IMP_POP, zeros(2, n), samples, Λ)
    z_grid = collect(Float64, DEFAULT_Z_GRID)
    interp = SampleInterpolant(samples.redshift, z_grid)
    obs = ObservationContext(
        [1.0, 2.0], [1.0, 1.0], [1.0, 1.0], BitVector([true, true]), 1.0, 1.0)
    ctx = ModelContext(
        zeros(n), ones(n), z_grid, interp, obs, 1.0, [0.0, 0.0])
    return problem, ctx, Λ
end

@testset "importance smoke and naive/cached parity" begin
    loaded = parity_problem_context(:posterior, [Detector("H1"), Detector("L1")])
    problem, C, ctx = loaded.problem, loaded.cosmology_type, loaded.ctx
    theta = PARITY_THETA

    weights_cached = compute_importance_weights(problem, C, theta, ctx)
    weights_naive = compute_importance_weights(problem, C, theta)
    @test length(weights_cached) == length(problem.samples.redshift)
    @test all(isfinite, weights_cached)
    @test weights_cached ≈ weights_naive   # R2 parity oracle

    rate_cached = merger_rate(problem, C, theta, ctx)
    rate_naive = merger_rate(
        problem, C, theta,
        ctx.local_merger_rate,
        ctx.observation.observation_time_yr,
        ctx.observation.observation_time_sec
    )
    @test isfinite(rate_cached)
    @test rate_cached ≈ rate_naive

    # Spectral-density parity: raw fluxes through the same array kernel on both paths.
    Sh_cached = spectral_density(problem.fluxes, rate_cached; weights = weights_cached)
    Sh_naive = spectral_density(problem.fluxes, rate_naive; weights = weights_naive)
    @test all(isfinite, Sh_cached)
    @test Sh_cached ≈ Sh_naive
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

    empty_problem, empty_ctx, _ = _importance_type_test_fixture(0)
    populated_problem, populated_ctx, _ = _importance_type_test_fixture(1)

    empty_weights = compute_importance_weights(empty_problem, _IMP_C, theta, empty_ctx)
    populated_weights = compute_importance_weights(
        populated_problem, _IMP_C, theta, populated_ctx)

    @test isempty(empty_weights)
    @test all(isfinite, populated_weights)
    @test eltype(populated_weights) <: ForwardDiff.Dual
end
