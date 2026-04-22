using HDF5
using Test

@testset "full intrinsic prior parity" begin
    cache_path = joinpath(@__DIR__, "fixtures", "full_intrinsic_cache_julia.h5")
    fixture_path = joinpath(@__DIR__, "fixtures", "deterministic_parity.h5")

    cache = load_cache(cache_path, [Detector("H1"), Detector("L1")])
    @test cache.observation.fiducial_spectral_density ≈ fiducial_spectral_density(cache)

    obs_disk = h5open(cache_path, "r") do f
        haskey(f, "fiducial_spectral_density") ||
            error("fixture $(repr(cache_path)) missing fiducial_spectral_density")
        vec(Float64.(read(f["fiducial_spectral_density"])))
    end

    h5open(fixture_path, "r") do file
        group = file["full_intrinsic_case"]
        theta = HyperParameters(;
            H0 = Float64(read(group["theta/H0"])),
            Ωm = Float64(read(group["theta/Omega_m"])),
            Ξ₀ = Float64(read(group["theta/chi0"])),
            Ξₙ = Float64(read(group["theta/chin"])),
            γ = Float64(read(group["theta/gamma"])),
            κ = Float64(read(group["theta/kappa"])),
            zpeak = Float64(read(group["theta/z_peak"]))
        )
        priors = build_uniform_priors(
            Dict(
            name => (
                Float64(read(group["prior_bounds/$(name)/low"])),
                Float64(read(group["prior_bounds/$(name)/high"]))
            ) for
        name in ("H0", "Omega_m", "chi0", "chin", "gamma", "kappa", "z_peak")
        ),
        )

        expected_target_log_prob = vec(Float64.(read(group["target_log_prob"])))
        expected_log_ratio = vec(Float64.(read(group["log_ratio"])))
        expected_weights = vec(Float64.(read(group["weights"])))
        expected_dgw_theta_sq = vec(Float64.(read(group["dgw_theta_sq"])))
        expected_redshift_integral = Float64(read(group["redshift_integral"]))
        expected_number_of_sources = Float64(read(group["expected_number_of_sources"]))
        expected_spectral_density = vec(Float64.(read(group["spectral_density_full"])))
        expected_log_prior = Float64(read(group["log_prior"]))
        expected_log_likelihood = Float64(read(group["log_likelihood"]))
        expected_log_posterior = Float64(read(group["log_posterior"]))

        evaluation = evaluate_importance_terms(theta, cache)

        # Fixtures were produced by the Python trapezoid/QuadGK stack; the Julia bundle
        # and luminosity distance now use composite Simpson interpolation.
        parity_rtol = 3e-2
        @test evaluation.target_log_prob ≈ expected_target_log_prob rtol = parity_rtol
        @test evaluation.log_ratio ≈ expected_log_ratio rtol = parity_rtol
        @test evaluation.weights ≈ expected_weights rtol = parity_rtol
        @test evaluation.dgw_theta_sq ≈ expected_dgw_theta_sq rtol = parity_rtol
        @test evaluation.redshift_integral ≈ expected_redshift_integral rtol = parity_rtol
        @test evaluation.expected_number_of_sources ≈ expected_number_of_sources rtol = parity_rtol
        @test evaluation.spectral_density ≈ expected_spectral_density rtol = parity_rtol
        @test logprior(theta, priors) ≈ expected_log_prior rtol = 1e-6
        @test loglikelihood(theta, cache; observed_spectral_density = obs_disk) ≈
              expected_log_likelihood rtol = parity_rtol
        @test logposterior(theta, cache, priors; observed_spectral_density = obs_disk) ≈
              expected_log_posterior rtol = parity_rtol
    end
end
