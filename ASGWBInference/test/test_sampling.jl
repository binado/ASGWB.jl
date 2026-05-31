using LogDensityProblems
using Test
using ASGWB
using Bijectors
using Distributions: product_distribution, Uniform
using ASGWBInference:
                      ASGWBLogDensity,
                      unconstrained_initial_point,
                      ad_logdensity,
                      finite_difference_logdensity_and_gradient,
                      sample_with_advancedhmc

if !@isdefined parity_bundle_dir
    include(joinpath(@__DIR__, "..", "..", "ASGWB", "test", "parity_test_cache.jl"))
end
include(joinpath(@__DIR__, "..", "..", "ASGWB", "test", "parity_fixtures.jl"))

@testset "AdvancedHMC initial point follows model order" begin
    loaded = parity_problem_context(:posterior, [Detector("H1"), Detector("L1")])
    cache, C, ctx = loaded.problem, loaded.cosmology_type, loaded.ctx
    theta0 = PARITY_THETA
    order = _PARITY_ORDER

    reordered_priors = product_distribution((
        zpeak = Uniform(0.0, 5.0),
        κ = Uniform(0.0, 10.0),
        Ξₙ = Uniform(-1.0, 1.0),
        Ξ₀ = Uniform(0.0, 2.0),
        γ = Uniform(0.0, 5.0),
        Ωm = Uniform(0.0, 1.0),
        H0 = Uniform(20.0, 140.0)
    ))

    @test_throws ArgumentError ASGWBLogDensity(cache, C, ctx, reordered_priors)

    problem = ASGWBLogDensity(cache, C, ctx, PARITY_PRIORS)
    ordered_theta0 = (; (k => theta0[k] for k in order)...)

    @test unconstrained_initial_point(problem, theta0) ==
          collect(Bijectors.link(PARITY_PRIORS, ordered_theta0))
end

@testset "AdvancedHMC smoke test" begin
    for variant in (:posterior, :full_intrinsic)
        loaded = parity_problem_context(variant, [Detector("H1"), Detector("L1")])
        cache, C, ctx = loaded.problem, loaded.cosmology_type, loaded.ctx
        theta0 = PARITY_THETA
        priors = PARITY_PRIORS

        problem = ASGWBLogDensity(cache, C, ctx, priors)
        z0 = unconstrained_initial_point(problem, theta0)
        ad_problem = ad_logdensity(problem)
        logdensity,
        gradient = LogDensityProblems.logdensity_and_gradient(ad_problem, z0)
        reference_logdensity,
        reference_gradient = finite_difference_logdensity_and_gradient(problem, z0)

        @test LogDensityProblems.dimension(problem) == 7
        @test LogDensityProblems.dimension(ad_problem) == 7
        @test isfinite(logdensity)
        @test all(isfinite, gradient)
        @test logdensity ≈ reference_logdensity rtol = 1e-9
        @test gradient ≈ reference_gradient rtol = 0.05

        samples, stats,
        sampling_problem = sample_with_advancedhmc(
            cache, C, ctx, priors, theta0; n_adapts = 3, n_samples = 3)

        @test sampling_problem isa ASGWBLogDensity
        @test length(samples) == 3
        @test length(stats) == 3

        prior_bounds = Dict(
            :H0 => (20.0, 140.0),
            :Ωm => (0.05, 0.95),
            :Ξ₀ => (0.5, 5.0),
            :Ξₙ => (0.05, 3.0),
            :γ => (0.5, 10.0),
            :κ => (0.05, 10.0),
            :zpeak => (0.05, 10.0)
        )
        for sample in samples
            for (sym, (low, high)) in prior_bounds
                value = getproperty(sample, sym)
                @test isfinite(value)
                @test low <= value <= high
            end
        end
    end
end
