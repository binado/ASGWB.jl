import Distributions: insupport, logpdf
using Distributions: ContinuousUnivariateDistribution, Uniform
using ForwardDiff
using Random
using Test
using CBCDistributions

const _theta_default = (
    H0 = 67.0,
    Ωm = 0.315,
    Ξ₀ = 1.0,
    Ξₙ = 0.0,
    γ = 2.7,
    κ = 3.0,
    zpeak = 2.5
)

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
        floatmin(Float64)
    ),
    )
    @test logpdf(spin_dist, 0.0) ≈ expected_spin_at_zero
    @test logpdf(spin_dist, spin_dist.a_max + 1e-3) == -Inf

    spin_sample = rand(MersenneTwister(2), spin_dist)
    @test minimum(spin_dist) <= spin_sample <= maximum(spin_dist)

    mass_f32 = OrderedUniformSourceMassPair(; low = 1.1f0, high = 2.5f0)
    @test eltype(mass_f32) === Float32
    @test typeof(mass_f32.low) === Float32
    @test isfinite(logpdf(mass_f32, (1.4f0, 1.2f0)))
    mass_f32_sample = rand(MersenneTwister(4), mass_f32)
    @test mass_f32.low <= mass_f32_sample[2] <= mass_f32_sample[1] <= mass_f32.high

    spin_f32 = AlignedSpinChiSimple(; a_max = 0.99f0)
    @test eltype(spin_f32) === Float32
    @test typeof(spin_f32.a_max) === Float32
    @test isfinite(logpdf(spin_f32, 0.0f0))
    spin_f32_sample = rand(MersenneTwister(5), spin_f32)
    @test minimum(spin_f32) <= spin_f32_sample <= maximum(spin_f32)

    theta = _theta_default
    spec = RedshiftPriorSpec(MadauDickinson, 0.001, 20.0, 256, nothing)
    redshift_prior = build_redshift_prior(theta, spec, LambdaCDM(theta.H0, theta.Ωm))
    redshift_dist = RedshiftInterpolatedDistribution(redshift_prior)
    @test logpdf(redshift_dist, 0.5) ≈ redshift_log_prob(redshift_prior, 0.5)
    @test !insupport(redshift_dist, spec.z_min - 0.01)
    @test logpdf(redshift_dist, spec.z_min - 0.01) == -Inf
    @test !insupport(redshift_dist, spec.z_max + 0.5)
    @test logpdf(redshift_dist, spec.z_max + 0.5) == -Inf

    redshift_sample = rand(MersenneTwister(3), redshift_dist)
    @test minimum(redshift_dist) <= redshift_sample <= maximum(redshift_dist)
end

@testset "intrinsic_prior factory returns IntrinsicPrior" begin
    prior = intrinsic_prior(FullBNS())
    @test prior isa IntrinsicPrior
    @test keys(prior.dists) == (:mass, :χ₁, :χ₂, :Λ₁, :Λ₂)
end

@testset "validate_batch" begin
    prior = intrinsic_prior(FullBNS())
    samples = (
        mass = stack_source_masses([1.4, 1.5], [1.2, 1.3]),
        redshift = [0.1, 0.2],
        χ₁ = [0.0, 0.1],
        χ₂ = [0.0, -0.2],
        Λ₁ = [100.0, 200.0],
        Λ₂ = [150.0, 250.0]
    )

    @test validate_batch(prior, samples) == 2
    @test validate_batch(
        IntrinsicPrior((χ₁ = prior.dists.χ₁,)),
        (; extra = [1.0, 2.0, 3.0], χ₁ = [0.0, 0.1])
    ) == 2
    @test validate_batch(prior,
        (; samples..., mass = zeros(2, 0), χ₁ = Float64[],
            χ₂ = Float64[], Λ₁ = Float64[], Λ₂ = Float64[])) == 0
    missing_spin = (
        mass = samples.mass,
        redshift = samples.redshift,
        χ₁ = samples.χ₁,
        Λ₁ = samples.Λ₁,
        Λ₂ = samples.Λ₂
    )
    @test_throws ArgumentError validate_batch(prior, missing_spin)
    @test_throws ArgumentError validate_batch(prior, (; samples..., χ₂ = [0.0]))
    @test_throws ArgumentError validate_batch(prior, (; samples..., mass = zeros(3, 2)))
    @test_throws ArgumentError IntrinsicPrior(NamedTuple())
end

@testset "IntrinsicPrior batch logpdf matches manual component sum" begin
    theta = _theta_default
    spec = RedshiftPriorSpec(MadauDickinson, 0.001, 20.0, 256, nothing)
    redshift_prior = build_redshift_prior(theta, spec, LambdaCDM(theta.H0, theta.Ωm))
    prior = intrinsic_prior(FullBNS())
    samples = (
        mass = stack_source_masses([1.4, 1.5], [1.2, 1.3]),
        redshift = [0.1, 0.2],
        χ₁ = [0.0, 0.1],
        χ₂ = [0.0, -0.2],
        Λ₁ = [100.0, 200.0],
        Λ₂ = [150.0, 250.0]
    )

    expected = [logpdf(prior.dists.mass, (samples.mass[1, i], samples.mass[2, i])) +
                logpdf(prior.dists.χ₁, samples.χ₁[i]) +
                logpdf(prior.dists.χ₂, samples.χ₂[i]) +
                logpdf(prior.dists.Λ₁, samples.Λ₁[i]) +
                logpdf(prior.dists.Λ₂, samples.Λ₂[i])
                for i in 1:length(samples.redshift)]

    @test logpdf(prior, samples) ≈ expected

    out = zeros(Float64, length(samples.redshift))
    CBCDistributions.logpdf!(out, prior, samples)
    @test out ≈ expected
    @test logpdf(prior, samples) .+
          redshift_log_prob_samples(redshift_prior, samples.redshift) ≈
          expected .+ redshift_log_prob.(Ref(redshift_prior), samples.redshift)
end

@testset "IntrinsicPrior output eltype and first-key sizing" begin
    prior = IntrinsicPrior((
        χ₁ = AlignedSpinChiSimple(; a_max = 0.99f0),
        Λ₁ = Uniform(0.0, 1.0)
    ))
    samples = (
        leading_extra = [1.0, 2.0, 3.0, 4.0],
        χ₁ = Float32[0.0, 0.1],
        Λ₁ = [0.2, 0.4]
    )
    got = logpdf(prior, samples)
    @test got isa Vector{Float64}
    @test length(got) == 2
    @test isempty(logpdf(prior, (; samples..., χ₁ = Float32[], Λ₁ = Float64[])))
    @test eltype(logpdf(prior, (; samples..., χ₁ = Float32[], Λ₁ = Float64[]))) === Float64
end

@testset "length-1 intrinsic batch" begin
    prior = intrinsic_prior(FullBNS())
    samples = (
        mass = stack_source_masses([1.4], [1.2]),
        χ₁ = [0.0],
        χ₂ = [0.0],
        Λ₁ = [100.0],
        Λ₂ = [150.0]
    )
    got = logpdf(prior, samples)
    expected = [
        logpdf(prior.dists.mass, (1.4, 1.2)) +
        logpdf(prior.dists.χ₁, 0.0) +
        logpdf(prior.dists.χ₂, 0.0) +
        logpdf(prior.dists.Λ₁, 100.0) +
        logpdf(prior.dists.Λ₂, 150.0)
    ]
    @test got ≈ expected
end

struct _ScaledBatchComponent <: ContinuousUnivariateDistribution
    scale::Float64
end

Base.eltype(::_ScaledBatchComponent) = Float64

function CBCDistributions._add_component_logpdf!(
        out::AbstractVector,
        d::_ScaledBatchComponent,
        field::AbstractVector
)
    @inbounds for i in eachindex(out, field)
        out[i] += d.scale * field[i]
    end
    return out
end

@testset "package-local custom component hook" begin
    prior = IntrinsicPrior((x = _ScaledBatchComponent(3.0),))
    samples = (x = [1.0, 2.0],)
    @test logpdf(prior, samples) == [3.0, 6.0]
end

@testset "cached intrinsic logpdf matches redshift composition" begin
    theta = _theta_default
    spec = RedshiftPriorSpec(MadauDickinson, 0.001, 20.0, 256, nothing)
    redshift_prior = build_redshift_prior(theta, spec, LambdaCDM(theta.H0, theta.Ωm))
    prior = intrinsic_prior(FullBNS())
    samples = (
        mass = stack_source_masses([1.4, 1.5], [1.2, 1.3]),
        redshift = [0.1, 0.2],
        χ₁ = [0.0, 0.1],
        χ₂ = [0.0, -0.2],
        Λ₁ = [100.0, 200.0],
        Λ₂ = [150.0, 250.0]
    )
    cached_log_prob = logpdf(prior, samples)
    @test cached_log_prob isa Vector{Float64}
    @test length(cached_log_prob) == length(samples.redshift)
    expected = logpdf(prior, samples)
    expected_with_redshift = expected .+
                             redshift_log_prob.(Ref(redshift_prior), samples.redshift)
    redshift_log_prob_vec = redshift_log_prob_samples(redshift_prior, samples.redshift)
    @test cached_log_prob .+ redshift_log_prob_vec ≈ expected_with_redshift
    out = similar(redshift_log_prob_vec)
    redshift_log_prob_samples!(out, redshift_prior, samples.redshift)
    @test cached_log_prob .+ out ≈ expected_with_redshift
end

@testset "cached intrinsic logpdf with ForwardDiff.Dual population parameter" begin
    theta = _theta_default
    spec = RedshiftPriorSpec(MadauDickinson, 0.001, 20.0, 256, nothing)
    samples = (
        mass = stack_source_masses([1.4, 1.5], [1.2, 1.3]),
        redshift = [0.1, 0.2],
        χ₁ = [0.0, 0.1],
        χ₂ = [0.0, -0.2],
        Λ₁ = [100.0, 200.0],
        Λ₂ = [150.0, 250.0]
    )
    cached_log_prob = logpdf(intrinsic_prior(FullBNS()), samples)
    h_dual = (; theta..., γ = ForwardDiff.Dual(2.7, 1.0))
    redshift_prior_dual = build_redshift_prior(h_dual, spec, LambdaCDM(theta.H0, theta.Ωm))
    prior_dual = intrinsic_prior(FullBNS())
    expected = logpdf(prior_dual, samples) .+
               redshift_log_prob.(Ref(redshift_prior_dual), samples.redshift)
    got = cached_log_prob .+
          redshift_log_prob_samples(redshift_prior_dual, samples.redshift)
    @test expected ≈ got
    @test eltype(got) <: ForwardDiff.Dual

    empty_samples = (
        mass = zeros(2, 0),
        redshift = Float64[],
        χ₁ = Float64[],
        χ₂ = Float64[],
        Λ₁ = Float64[],
        Λ₂ = Float64[]
    )
    empty_cached_log_prob = logpdf(intrinsic_prior(FullBNS()), empty_samples)
    empty_redshift = redshift_log_prob_samples(redshift_prior_dual, empty_samples.redshift)
    empty_got = empty_cached_log_prob .+ empty_redshift
    @test isempty(empty_got)
    @test eltype(empty_redshift) <: ForwardDiff.Dual
    @test eltype(empty_got) <: ForwardDiff.Dual
end
