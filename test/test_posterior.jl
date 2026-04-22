using HDF5
using Test

@testset "posterior parity" begin
    cache_path = joinpath(@__DIR__, "fixtures", "posterior_cache_julia.h5")
    fixture_path = joinpath(@__DIR__, "fixtures", "deterministic_parity.h5")

    cache = load_cache(cache_path, [Detector("H1"), Detector("L1")])
    @test cache.observation.fiducial_spectral_density ≈ fiducial_spectral_density(cache)

    # Golden log_likelihood / log_posterior in deterministic_parity.h5 were computed against
    # the HDF5 `fiducial_spectral_density` vector; `load_cache` now ignores that dataset for
    # the default observed spectrum and recomputes in Julia.
    obs_disk = h5open(cache_path, "r") do f
        haskey(f, "fiducial_spectral_density") ||
            error("fixture $(repr(cache_path)) missing fiducial_spectral_density")
        vec(Float64.(read(f["fiducial_spectral_density"])))
    end

    h5open(fixture_path, "r") do file
        group = file["posterior_case"]
        theta = HyperParameters((;
            (
            Symbol(name) => Float64(read(group["theta/$(name)"]))
        for
        name in ("H0", "Omega_m", "chi0", "chin", "gamma", "kappa", "z_peak")
        )...,
        ))
        priors = build_uniform_priors(
            Dict(
            name => (
                Float64(read(group["prior_bounds/$(name)/low"])),
                Float64(read(group["prior_bounds/$(name)/high"]))
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

        # Downstream parity tolerances widened because the Julia bundle/lumdist now uses
        # Simpson interpolation rather than trapezoid/QuadGK as in the Python fixture.
        parity_rtol = 3e-2
        @test logprior(theta, priors) ≈ expected_log_prior rtol = 1e-6
        @test loglikelihood(theta, cache; observed_spectral_density = obs_disk) ≈
              expected_log_likelihood rtol = parity_rtol
        @test logposterior(theta, cache, priors; observed_spectral_density = obs_disk) ≈
              expected_log_posterior rtol = parity_rtol
        @test normalized_ess(evaluation.weights) ≈ expected_ess rtol = parity_rtol
        @test max_normalized_weight(evaluation.weights) ≈ expected_max_weight rtol = parity_rtol
        # log_ratio_variance with 2 samples is effectively zero; use atol.
        @test log_ratio_variance(evaluation.log_ratio) ≈ expected_log_ratio_variance atol = 1e-10
    end
end
