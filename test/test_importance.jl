using HDF5
using Test

@testset "importance parity" begin
    cache_path = joinpath(@__DIR__, "fixtures", "posterior_cache_julia.h5")
    fixture_path = joinpath(@__DIR__, "fixtures", "deterministic_parity.h5")

    cache = load_cache(cache_path, [Detector("H1"), Detector("L1")])

    h5open(fixture_path, "r") do file
        group = file["posterior_case"]
        theta = HyperParameters((; (
            Symbol(name) => Float64(read(group["theta/$(name)"])) for
            name in ("H0", "Omega_m", "chi0", "chin", "gamma", "kappa", "z_peak")
        )...,))
        expected_dgw_theta_sq = vec(Float64.(read(group["dgw_theta_sq"])))
        expected_weights = vec(Float64.(read(group["weights"])))
        expected_spectral_density = vec(Float64.(read(group["spectral_density_full"])))
        expected_spectral_density_in_band = vec(
            Float64.(read(group["spectral_density_in_band"])),
        )
        expected_number_of_sources = Float64(read(group["expected_number_of_sources"]))
        expected_log_ratio = vec(Float64.(read(group["log_ratio"])))
        expected_target_log_prob = vec(Float64.(read(group["target_log_prob"])))
        expected_redshift_integral = Float64(read(group["redshift_integral"]))

        evaluation = evaluate_importance_terms(theta, cache)

        @test evaluation.dgw_theta_sq ≈ expected_dgw_theta_sq rtol = 1e-6
        @test evaluation.target_log_prob ≈ expected_target_log_prob rtol = 1e-6
        @test evaluation.log_ratio ≈ expected_log_ratio rtol = 1e-6
        @test evaluation.weights ≈ expected_weights rtol = 1e-6
        @test evaluation.redshift_integral ≈ expected_redshift_integral rtol = 1e-6
        @test evaluation.expected_number_of_sources ≈ expected_number_of_sources rtol = 1e-6
        @test evaluation.spectral_density ≈ expected_spectral_density rtol = 1e-6
        @test evaluation.spectral_density_in_band ≈ expected_spectral_density_in_band rtol = 1e-6
    end
end
