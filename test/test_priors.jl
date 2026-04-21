import Distributions: insupport, logpdf
using Random
using Test

@testset "intrinsic prior distributions" begin
    mass_dist = OrderedUniformSourceMassPair()
    expected_mass_logpdf = log(2.0) - 2.0 * log(mass_dist.high - mass_dist.low)
    @test logpdf(mass_dist, [1.4, 1.2]) ≈ expected_mass_logpdf
    @test logpdf(mass_dist, (1.4, 1.2)) ≈ expected_mass_logpdf
    @test !insupport(mass_dist, [1.2, 1.4])
    @test logpdf(mass_dist, [1.2, 1.4]) == -Inf

    mass_sample = rand(MersenneTwister(1), mass_dist)
    @test length(mass_sample) == 2
    @test mass_dist.low <= mass_sample[2] <= mass_sample[1] <= mass_dist.high

    spin_dist = AlignedSpinChiSimple()
    expected_spin_at_zero = log(
        max(
            -log(eps(Float64) / spin_dist.a_max) / (2.0 * spin_dist.a_max),
            floatmin(Float64),
        ),
    )
    @test logpdf(spin_dist, 0.0) ≈ expected_spin_at_zero
    @test logpdf(spin_dist, spin_dist.a_max + 1e-3) == -Inf

    spin_sample = rand(MersenneTwister(2), spin_dist)
    @test minimum(spin_dist) <= spin_sample <= maximum(spin_dist)

    theta = HyperParameters(;
        H0=67.0,
        Omega_m=0.315,
        gamma=2.7,
        kappa=3.0,
        z_peak=2.5,
    )
    spec = RedshiftPriorSpec(MadauDickinson, 0.001, 20.0, 256, nothing)
    bundle = build_redshift_grid_bundle(theta, spec)
    redshift_dist = RedshiftInterpolatedDistribution(bundle)
    @test logpdf(redshift_dist, 0.5) ≈ log_prob_from_bundle(0.5, bundle)

    redshift_sample = rand(MersenneTwister(3), redshift_dist)
    @test minimum(redshift_dist) <= redshift_sample <= maximum(redshift_dist)
end

@testset "generic intrinsic prior aggregation" begin
    theta = HyperParameters(;
        H0=67.0,
        Omega_m=0.315,
        gamma=2.7,
        kappa=3.0,
        z_peak=2.5,
    )
    spec = RedshiftPriorSpec(MadauDickinson, 0.001, 20.0, 256, nothing)
    bundle = build_redshift_grid_bundle(theta, spec)
    terms = intrinsic_prior_terms(FullBNS(), bundle)
    samples = FullBNSSamples(
        [1.4, 1.5],
        [1.2, 1.3],
        [0.1, 0.2],
        [0.0, 0.1],
        [0.0, -0.2],
        [100.0, 200.0],
        [150.0, 250.0],
    )

    expected = Float64[
        logpdf(terms[1].dist, [samples.mass_1_source[i], samples.mass_2_source[i]]) +
        logpdf(terms[2].dist, samples.redshift[i]) +
        logpdf(terms[3].dist, samples.chi_1[i]) +
        logpdf(terms[4].dist, samples.chi_2[i]) +
        logpdf(terms[5].dist, samples.lambda_1[i]) +
        logpdf(terms[6].dist, samples.lambda_2[i]) for i in eachindex(samples.redshift)
    ]

    @test intrinsic_log_prob_samples(samples, terms) ≈ expected
end
