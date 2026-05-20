using LogDensityProblems
using Test
using ASGWB
using ASGWBInference:
                      ASGWBLogDensity,
                      unconstrained_initial_point,
                      ad_logdensity,
                      finite_difference_logdensity_and_gradient,
                      sample_with_advancedhmc

include(joinpath(@__DIR__, "..", "..", "ASGWB", "test", "parity_fixtures.jl"))

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
