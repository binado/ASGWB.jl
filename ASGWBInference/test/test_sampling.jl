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

if !@isdefined parity_cache_path
    include(joinpath(@__DIR__, "..", "..", "ASGWB", "test", "parity_test_cache.jl"))
end
include(joinpath(@__DIR__, "..", "..", "ASGWB", "test", "parity_fixtures.jl"))

@testset "AdvancedHMC initial point follows prior order" begin
    cache = load_cache(parity_cache_path(:posterior), [Detector("H1"), Detector("L1")])
    theta0 = PARITY_THETA
    reordered_priors = product_distribution((
        zpeak = Uniform(0.0, 5.0),
        κ = Uniform(0.0, 10.0),
        Ξₙ = Uniform(-1.0, 1.0),
        Ξ₀ = Uniform(0.0, 2.0),
        γ = Uniform(0.0, 5.0),
        Ωm = Uniform(0.0, 1.0),
        H0 = Uniform(20.0, 140.0)
    ))

    problem = ASGWBLogDensity(cache, reordered_priors)
    ordered_theta0 = (; (k => theta0[k] for k in hyperparameter_order(reordered_priors))...)

    @test unconstrained_initial_point(problem, theta0) ==
          collect(Bijectors.link(reordered_priors, ordered_theta0))
end

@testset "AdvancedHMC smoke test" begin
    for variant in (:posterior, :full_intrinsic)
        cache = load_cache(parity_cache_path(variant), [Detector("H1"), Detector("L1")])
        theta0 = PARITY_THETA
        priors = PARITY_PRIORS
        prior_bounds = PARITY_PRIOR_BOUNDS

        problem = ASGWBLogDensity(cache, priors)
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
            cache, priors, theta0; n_adapts = 3, n_samples = 3)

        @test sampling_problem isa ASGWBLogDensity
        @test length(samples) == 3
        @test length(stats) == 3

        _prop_sym = Dict(
            "H0" => :H0,
            "Omega_m" => :Ωm,
            "chi0" => :Ξ₀,
            "chin" => :Ξₙ,
            "gamma" => :γ,
            "kappa" => :κ,
            "z_peak" => :zpeak
        )
        for sample in samples
            for (name, (low, high)) in prior_bounds
                value = getproperty(sample, _prop_sym[name])
                @test isfinite(value)
                @test low <= value <= high
            end
        end
    end
end
