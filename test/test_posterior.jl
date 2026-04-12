using HDF5
using Test

@testset "posterior parity" begin
    cache_path = joinpath(@__DIR__, "fixtures", "posterior_cache_julia.h5")
    fixture_path = joinpath(@__DIR__, "fixtures", "deterministic_parity.h5")

    cache = load_cache(cache_path, [Detector("H1"), Detector("L1")])

    h5open(fixture_path, "r") do file
        group = file["posterior_case"]
        theta = HyperParameters((; (
            Symbol(name) => Float64(read(group["theta/$(name)"])) for
            name in ("H0", "Omega_m", "chi0", "chin", "gamma", "kappa", "z_peak")
        )...,))
        priors = build_uniform_priors(
            Dict(
                name => (
                    Float64(read(group["prior_bounds/$(name)/low"])),
                    Float64(read(group["prior_bounds/$(name)/high"])),
                ) for
                name in ("H0", "Omega_m", "chi0", "chin", "gamma", "kappa", "z_peak")
            ),
        )

        expected_log_prior = Float64(read(group["log_prior"]))
        expected_log_likelihood = Float64(read(group["log_likelihood"]))
        expected_log_posterior = Float64(read(group["log_posterior"]))
        expected_ess = Float64(read(group["normalized_ess"]))
        expected_max_weight = Float64(read(group["max_normalized_weight"]))
        expected_log_ratio_variance = Float64(read(group["log_ratio_variance"]))

        evaluation = evaluate_importance_terms(theta, cache)

        @test logprior(theta, priors) ≈ expected_log_prior rtol = 1e-6
        @test loglikelihood(theta, cache) ≈ expected_log_likelihood rtol = 1e-6
        @test logposterior(theta, cache, priors) ≈ expected_log_posterior rtol = 1e-6
        @test normalized_ess(evaluation.weights) ≈ expected_ess rtol = 1e-6
        @test max_normalized_weight(evaluation.weights) ≈ expected_max_weight rtol = 1e-6
        @test log_ratio_variance(evaluation.log_ratio) ≈ expected_log_ratio_variance rtol = 1e-6
    end
end
