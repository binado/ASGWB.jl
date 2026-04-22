using HDF5
using Test

@testset "importance parity" begin
    cache_path = joinpath(@__DIR__, "fixtures", "posterior_cache_julia.h5")
    fixture_path = joinpath(@__DIR__, "fixtures", "deterministic_parity.h5")

    cache = load_cache(cache_path, [Detector("H1"), Detector("L1")])

    h5open(fixture_path, "r") do file
        group = file["posterior_case"]
        theta = HyperParameters((;
            (
            Symbol(name) => Float64(read(group["theta/$(name)"]))
        for
        name in ("H0", "Omega_m", "chi0", "chin", "gamma", "kappa", "z_peak")
        )...,
        ))
        expected_dgw_theta_sq = vec(Float64.(read(group["dgw_theta_sq"])))
        expected_weights = vec(Float64.(read(group["weights"])))
        expected_spectral_density = vec(Float64.(read(group["spectral_density_full"])))
        expected_spectral_density_in_band = vec(Float64.(read(group["spectral_density_in_band"])))
        expected_number_of_sources = Float64(read(group["expected_number_of_sources"]))
        expected_log_ratio = vec(Float64.(read(group["log_ratio"])))
        expected_target_log_prob = vec(Float64.(read(group["target_log_prob"])))
        expected_redshift_integral = Float64(read(group["redshift_integral"]))

        evaluation = evaluate_importance_terms(theta, cache)

        # Fixture values come from the Python trapezoid-based bundle norm and QuadGK-based
        # luminosity distance; Julia now uses composite Simpson for the bundle and a
        # Simpson-interpolated luminosity distance. Tolerances reflect those
        # discretization gaps, not numerical precision.
        parity_rtol = 3e-2
        @test evaluation.dgw_theta_sq ≈ expected_dgw_theta_sq rtol = parity_rtol
        @test evaluation.target_log_prob ≈ expected_target_log_prob rtol = parity_rtol
        @test evaluation.log_ratio ≈ expected_log_ratio rtol = parity_rtol
        @test evaluation.weights ≈ expected_weights rtol = parity_rtol
        @test evaluation.redshift_integral ≈ expected_redshift_integral rtol = parity_rtol
        @test evaluation.expected_number_of_sources ≈ expected_number_of_sources rtol = parity_rtol
        @test evaluation.spectral_density ≈ expected_spectral_density rtol = parity_rtol
        @test evaluation.spectral_density_in_band ≈ expected_spectral_density_in_band rtol = parity_rtol

        bundle = build_redshift_grid_bundle(theta, cache.redshift_prior_spec)
        iw = compute_importance_weights(cache, theta, bundle)
        @test iw.weights ≈ expected_weights rtol = parity_rtol
        @test iw.log_ratio ≈ expected_log_ratio rtol = parity_rtol
        @test iw.target_log_prob ≈ expected_target_log_prob rtol = parity_rtol
        @test iw.dgw_theta_sq ≈ expected_dgw_theta_sq rtol = parity_rtol

        rate = merger_rate_per_sec(
            bundle,
            cache.local_merger_rate,
            cache.observation.observation_time_yr,
            cache.observation.observation_time_sec
        )
        @test rate * cache.observation.observation_time_sec ≈ expected_number_of_sources rtol = parity_rtol
    end
end
