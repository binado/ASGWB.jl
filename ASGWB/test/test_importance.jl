using HDF5
using ForwardDiff
using Test

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

@testset "importance parity" begin
    cache_path = joinpath(@__DIR__, "fixtures", "posterior_cache_julia.h5")
    fixture_path = joinpath(@__DIR__, "fixtures", "deterministic_parity.h5")

    cache = load_cache(cache_path, [Detector("H1"), Detector("L1")])

    h5open(fixture_path, "r") do file
        group = file["posterior_case"]
        theta = HyperParameters(;
            H0 = Float64(read(group["theta/H0"])),
            Ωm = Float64(read(group["theta/Omega_m"])),
            Ξ₀ = Float64(read(group["theta/chi0"])),
            Ξₙ = Float64(read(group["theta/chin"])),
            γ = Float64(read(group["theta/gamma"])),
            κ = Float64(read(group["theta/kappa"])),
            zpeak = Float64(read(group["theta/z_peak"]))
        )
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

        cosmology_cache,
        redshift_prior = cosmology_and_redshift_prior(
            theta,
            cache.redshift_prior_spec,
            cache.redshift_cache.redshift_grid
        )
        iw = compute_importance_weights(cache, theta, cosmology_cache, redshift_prior)
        @test iw.weights ≈ expected_weights rtol = parity_rtol
        @test iw.log_ratio ≈ expected_log_ratio rtol = parity_rtol
        @test iw.target_log_prob ≈ expected_target_log_prob rtol = parity_rtol
        @test iw.dgw_theta_sq ≈ expected_dgw_theta_sq rtol = parity_rtol

        rate = merger_rate_per_sec(
            redshift_prior,
            cache.local_merger_rate,
            cache.observation.observation_time_yr,
            cache.observation.observation_time_sec
        )
        @test rate * cache.observation.observation_time_sec ≈ expected_number_of_sources rtol = parity_rtol
    end
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
    empty_cosmology_cache,
    empty_redshift_prior = cosmology_and_redshift_prior(
        theta,
        empty_problem.redshift_prior_spec,
        empty_problem.redshift_cache.redshift_grid
    )
    populated_cosmology_cache,
    populated_redshift_prior = cosmology_and_redshift_prior(
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
