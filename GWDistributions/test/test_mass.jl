using Distributions
using ForwardDiff
using QuadGK
using Random

function _default_bbh_pair(; kwargs...)
    params = merge(
        (
            α1 = 1.5,
            α2 = 4.0,
            m_break = 35.0,
            μ1 = 10.0,
            σ1 = 2.0,
            μ2 = 35.0,
            σ2 = 6.0,
            m1_low = 5.0,
            δm1 = 4.0,
            λ0 = 0.55,
            λ1 = 0.25,
            βq = 1.2,
            m2_low = 4.0,
            δm2 = 3.0,
            m_high = 120.0
        ),
        kwargs
    )
    return DefaultBBHMassPair(; params...)
end

function _q_planck_reference(d::DefaultBBHMassPair, m1::Real)
    q_low = d.m2_low / m1
    q_low < 1 || return zero(promote_type(typeof(q_low), typeof(d.βq)))
    d.δm2 == 0 && return GWDistributions._q_power_integral(d, q_low, one(q_low))

    δq = d.δm2 / m1
    q_taper_high = min(q_low + δq, one(q_low))
    z = zero(promote_type(typeof(q_low), typeof(d.βq), typeof(δq)))
    if q_taper_high > q_low
        h = (q_taper_high - q_low) / δq
        band,
        _ = quadgk(
            t -> (q_low + δq * t)^d.βq * GWDistributions._planck_unit_taper(t),
            zero(h), h; rtol = 1e-12)
        z += δq * band
    end
    if q_taper_high < 1
        z += GWDistributions._q_power_integral(d, q_taper_high, one(q_low))
    end
    return z
end

@testset "Planck taper boundary behavior" begin
    @test planck_taper(4.9, 5.0, 2.0) == 0.0
    @test planck_taper(5.0, 5.0, 2.0) == 0.0
    @test 0.0 < planck_taper(6.0, 5.0, 2.0) < 1.0
    @test planck_taper(6.0, 5.0, 2.0)≈0.5 atol=1e-12
    @test planck_taper(7.0, 5.0, 2.0) == 1.0
    @test planck_taper(5.0, 5.0, 0.0) == 1.0
    @test isfinite(planck_taper(nextfloat(5.0), 5.0, 2.0))
    @test isfinite(planck_taper(prevfloat(7.0), 5.0, 2.0))
end

@testset "BrokenPowerLaw normalization and support" begin
    d = GWDistributions.BrokenPowerLaw(1.5, 4.0, 35.0, 5.0, 120.0)
    z, _ = quadgk(m -> pdf(d, m), minimum(d), maximum(d))

    @test z≈1.0 rtol=1e-8
    @test isfinite(logpdf(d, 10.0))   # below the break
    @test isfinite(logpdf(d, 75.0))   # above the break
    @test logpdf(d, 4.999) == -Inf
    @test logpdf(d, 120.0) == -Inf
    # The two slopes share the m_break pivot, so the density is continuous there.
    @test pdf(d, prevfloat(35.0))≈pdf(d, 35.0) rtol=1e-10
end

@testset "DefaultBBHPrimaryMass normalization" begin
    d = DefaultBBHPrimaryMass(;
        α1 = 1.0,
        α2 = 1.0,
        m_break = 35.0,
        μ1 = 10.0,
        σ1 = 2.0,
        μ2 = 35.0,
        σ2 = 6.0,
        m1_low = 5.0,
        δm1 = 0.0,
        λ0 = 1.0,
        λ1 = 0.0,
        m_high = 120.0
    )

    z, _ = quadgk(m -> pdf(d, m), minimum(d), maximum(d))
    @test z≈1.0 rtol=1e-8
    @test logpdf(d, d.m_high) == -Inf

    tapered = DefaultBBHPrimaryMass(;
        α1 = 1.5,
        α2 = 4.0,
        m_break = 35.0,
        μ1 = 10.0,
        σ1 = 2.0,
        μ2 = 35.0,
        σ2 = 6.0,
        m1_low = 5.0,
        δm1 = 4.0,
        λ0 = 0.55,
        λ1 = 0.25,
        m_high = 120.0
    )
    z_tapered, _ = quadgk(m -> pdf(tapered, m), minimum(tapered), maximum(tapered))
    @test z_tapered≈1.0 rtol=1e-7
end

@testset "DefaultBBHPrimaryMass untapered mixture formula" begin
    d = DefaultBBHPrimaryMass(;
        α1 = 1.5,
        α2 = 4.0,
        m_break = 35.0,
        μ1 = 10.0,
        σ1 = 2.0,
        μ2 = 35.0,
        σ2 = 6.0,
        m1_low = 5.0,
        δm1 = 0.0,
        λ0 = 0.55,
        λ1 = 0.25,
        m_high = 120.0
    )

    # Independent oracle for the untapered density: the λ-weighted sum of a single
    # broken power law (one normalizer over the full support) and the two truncated
    # Gaussians, derived without reusing the implementation's mixture machinery.
    broken = d.broken
    peak1 = d.peak1
    peak2 = d.peak2
    z1 = GWDistributions.normalizer(broken.lower)
    z2 = GWDistributions.normalizer(broken.upper)
    z = z1 + z2
    g1 = peak1
    g2 = peak2
    m_break = broken.lower.pivot
    broken_pdf(m) = (m / m_break)^(-(m < m_break ? broken.lower.α : broken.upper.α)) / z

    for m in (6.0, 15.0, 35.0, 75.0, 119.0)
        expected = d.λ0 * broken_pdf(m) + d.λ1 * pdf(g1, m) + d.λ2 * pdf(g2, m)
        @test pdf(d, m)≈expected rtol=1e-10
    end
end

@testset "DefaultBBHPrimaryMass logpdf is type-stable" begin
    d = DefaultBBHPrimaryMass(;
        α1 = 1.5,
        α2 = 4.0,
        m_break = 35.0,
        μ1 = 10.0,
        σ1 = 2.0,
        μ2 = 35.0,
        σ2 = 6.0,
        m1_low = 5.0,
        δm1 = 0.0,
        λ0 = 0.55,
        λ1 = 0.25,
        m_high = 120.0
    )
    # Concrete component fields keep the logpdf path inferrable; a heterogeneous
    # MixtureModel field would degrade it to `Any` and propagate into the mass pair.
    @test @inferred(logpdf(d, 30.0)) isa Float64
    pair = _default_bbh_pair()
    @test @inferred(logpdf(pair, (35.0, 25.0))) isa Float64
end

@testset "DefaultBBHMassPair logpdf and conditional normalization" begin
    d = _default_bbh_pair()
    @test isfinite(logpdf(d, (35.0, 25.0)))
    @test logpdf(d, (35.0, 36.0)) == -Inf
    @test logpdf(d, (4.9, 4.5)) == -Inf
    @test logpdf(d, (121.0, 40.0)) == -Inf
    @test_throws ArgumentError _default_bbh_pair(λ0 = 0.8, λ1 = 0.3)

    m1 = 35.0
    logp_m1 = logpdf(d.primary, m1)
    z_conditional,
    _ = quadgk(m2 -> exp(logpdf(d, (m1, m2)) - logp_m1), d.m2_low, m1)
    @test z_conditional≈1.0 rtol=1e-7
end

@testset "DefaultBBHMassPair rand support and shape" begin
    rng = MersenneTwister(1234)
    d = _default_bbh_pair()
    draws = rand(rng, d, 256)

    @test size(draws) == (2, 256)
    @test all(draws[1, :] .>= draws[2, :])
    @test all(draws[1, :] .>= d.primary.m1_low)
    @test all(draws[1, :] .< d.primary.m_high)
    @test all(draws[2, :] .>= d.m2_low)
    @test all(isfinite, [logpdf(d, (draws[1, i], draws[2, i])) for i in axes(draws, 2)])

    low_draws = rand(rng, _default_bbh_pair(δm1 = 8.0, δm2 = 8.0), 2_000)
    @test any(low_draws[1, :] .< 13.0)
    @test count(low_draws[1, :] .< 7.0) < count(low_draws[1, :] .< 13.0)
end

@testset "DefaultBBHMassPair interior logpdf is differentiable in m2" begin
    d = _default_bbh_pair()
    grad = ForwardDiff.gradient(x -> logpdf(d, (35.0, x[1])), [25.0])
    @test all(isfinite, grad)

    # A point inside the secondary taper band [m2_low, m2_low + δm2] exercises the
    # Planck window itself, not just the flat region above it.
    @test d.m2_low < 5.5 < d.m2_low + d.δm2
    grad_band = ForwardDiff.gradient(x -> logpdf(d, (35.0, x[1])), [5.5])
    @test all(isfinite, grad_band)
end

@testset "DefaultBBHMassPair _q_normalizer matches Planck split quadrature" begin
    for βq in (1.2, 2.5, 0.0, -1.0, -2.0), m1 in (10.0, 35.0, 80.0)

        d = _default_bbh_pair(βq = βq)
        ref = _q_planck_reference(d, m1)
        @test GWDistributions._q_normalizer(d, m1)≈ref rtol=2e-8
    end

    # Untapered limit reduces to the closed-form power integral.
    d0 = _default_bbh_pair(δm2 = 0.0)
    q_low = d0.m2_low / 35.0
    ref0, _ = quadgk(q -> q^d0.βq, q_low, 1.0; rtol = 1e-12)
    @test GWDistributions._q_normalizer(d0, 35.0)≈ref0 rtol=1e-10
end

@testset "DefaultBBHMassPair fixed Planck rule convergence" begin
    m1 = 35.0
    for βq in (2.5, 1.2, 0.0, -1.0, -2.0), δm2 in (5.0, 3.0, 0.1, 1e-3)

        d = _default_bbh_pair(βq = βq, δm2 = δm2)
        q_low = d.m2_low / m1
        δq = d.δm2 / m1
        h = (min(q_low + δq, 1.0) - q_low) / δq
        z16 = GWDistributions._q_planck_taper_band_integral(
            q_low, δq, d.βq, h, Val(16))
        z32 = GWDistributions._q_planck_taper_band_integral(
            q_low, δq, d.βq, h, Val(32))
        @test z16≈z32 rtol=1e-7 atol=1e-12
    end
end
