import Distributions: insupport, logpdf
using Distributions: ProductNamedTupleDistribution
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

    # Batched multivariate draw uses the new _rand! path.
    mass_batch = rand(MersenneTwister(1), mass_dist, 4)
    @test size(mass_batch) == (2, 4)
    @test all(mass_batch[1, :] .>= mass_batch[2, :])
    @test all(mass_dist.low .<= mass_batch[2, :])
    @test all(mass_batch[1, :] .<= mass_dist.high)

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
    @test !insupport(redshift_dist, spec.z_min - 0.01)
    @test logpdf(redshift_dist, spec.z_min - 0.01) == -Inf
    @test !insupport(redshift_dist, spec.z_max + 0.5)
    @test logpdf(redshift_dist, spec.z_max + 0.5) == -Inf

    redshift_sample = rand(MersenneTwister(3), redshift_dist)
    @test minimum(redshift_dist) <= redshift_sample <= maximum(redshift_dist)
end

@testset "intrinsic_prior factory returns ProductNamedTupleDistribution" begin
    theta = HyperParameters(;
        H0=67.0,
        Omega_m=0.315,
        gamma=2.7,
        kappa=3.0,
        z_peak=2.5,
    )
    spec = RedshiftPriorSpec(MadauDickinson, 0.001, 20.0, 256, nothing)
    bundle = build_redshift_grid_bundle(theta, spec)
    prior = intrinsic_prior(FullBNS(), bundle)
    @test prior isa ProductNamedTupleDistribution
    @test keys(prior.dists) ==
        (:mass, :redshift, :chi_1, :chi_2, :lambda_1, :lambda_2)

    single = rand(MersenneTwister(7), prior)
    @test keys(single) == (:mass, :redshift, :chi_1, :chi_2, :lambda_1, :lambda_2)
    @test single.mass isa AbstractVector && length(single.mass) == 2
    @test single.mass[1] >= single.mass[2]
    @test logpdf(prior, single) isa Real

    # `rand(prior, n)` returns an AoS Vector of NamedTuples for
    # ProductNamedTupleDistribution. The SoA container is used for proposal samples
    # loaded from HDF5 and exercised through `intrinsic_log_prob_samples` below.
    batched = rand(MersenneTwister(7), prior, 5)
    @test batched isa AbstractVector
    @test length(batched) == 5
    @test eltype(batched) <: NamedTuple
    @test all(s -> length(s.mass) == 2 && s.mass[1] >= s.mass[2], batched)
end

@testset "intrinsic_log_prob_samples SoA fast path matches native logpdf" begin
    theta = HyperParameters(;
        H0=67.0,
        Omega_m=0.315,
        gamma=2.7,
        kappa=3.0,
        z_peak=2.5,
    )
    spec = RedshiftPriorSpec(MadauDickinson, 0.001, 20.0, 256, nothing)
    bundle = build_redshift_grid_bundle(theta, spec)
    prior = intrinsic_prior(FullBNS(), bundle)
    samples = (
        mass=stack_source_masses([1.4, 1.5], [1.2, 1.3]),
        redshift=[0.1, 0.2],
        chi_1=[0.0, 0.1],
        chi_2=[0.0, -0.2],
        lambda_1=[100.0, 200.0],
        lambda_2=[150.0, 250.0],
    )

    expected = [
        logpdf(prior, (
            mass=[samples.mass[1, i], samples.mass[2, i]],
            redshift=samples.redshift[i],
            chi_1=samples.chi_1[i],
            chi_2=samples.chi_2[i],
            lambda_1=samples.lambda_1[i],
            lambda_2=samples.lambda_2[i],
        )) for i in 1:length(samples.redshift)
    ]

    @test intrinsic_log_prob_samples(prior, samples) ≈ expected

    out = zeros(Float64, length(samples.redshift))
    intrinsic_log_prob_samples!(out, prior, samples)
    @test out ≈ expected
end

@testset "intrinsic_log_prob_samples AoS fallback" begin
    theta = HyperParameters(;
        H0=67.0,
        Omega_m=0.315,
        gamma=2.7,
        kappa=3.0,
        z_peak=2.5,
    )
    spec = RedshiftPriorSpec(MadauDickinson, 0.001, 20.0, 256, nothing)
    bundle = build_redshift_grid_bundle(theta, spec)
    prior = intrinsic_prior(FullBNS(), bundle)
    aos = [
        (mass=[1.4, 1.2], redshift=0.1, chi_1=0.0, chi_2=0.0, lambda_1=100.0, lambda_2=150.0),
        (mass=[1.5, 1.3], redshift=0.2, chi_1=0.1, chi_2=-0.2, lambda_1=200.0, lambda_2=250.0),
    ]
    @test intrinsic_log_prob_samples(prior, aos) == [logpdf(prior, s) for s in aos]
end
