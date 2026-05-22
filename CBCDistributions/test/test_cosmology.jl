using QuadGK
using Test
using ForwardDiff
using CBCDistributions: CumulativeIntegral1D, cdf, hubble_constant_si, interpolate,
                        normalizer

@testset "hubble_constant_si" begin
    H0 = 70.0
    @test hubble_constant_si(H0) ≈ Float64(H0) * 1000.0 / 3.085677581e22
end

@testset "basic cosmology helpers" begin
    h = (H0 = 67.0, Ωm = 0.315)
    cosmology = Cosmology(h)
    @test cosmology.H0 == h.H0
    @test cosmology.Ωm == h.Ωm

    @test E(0.0, 0.315) ≈ 1.0
    @test comoving_distance(0.0, 67.0, 0.315) ≈ 0.0

    z = [0.0, 0.1, 0.2]
    d_l = luminosity_distance.(z, 67.0, 0.315)
    @test d_l[1] ≈ 0.0
    @test d_l[3] > d_l[2] > d_l[1]

    d_gw = gravitational_wave_distance.([0.1, 0.2], [10.0, 20.0], 1.0, 0.0)
    @test d_gw ≈ [10.0, 20.0]
end

@testset "CosmologyCache distance helpers" begin
    H0, Ωm = 67.0, 0.315
    cache = CosmologyCache(Cosmology(H0, Ωm), collect(LinRange(0.0, 10.0, 1024)))
    for z in (0.05, 0.3, 1.2, 4.5, 8.0)
        @test comoving_distance(z, cache) ≈ comoving_distance(z, H0, Ωm) rtol = 1e-4
        @test luminosity_distance(z, cache) ≈ luminosity_distance(z, H0, Ωm) rtol = 1e-4
        @test differential_comoving_volume(z, cache) ≈
              differential_comoving_volume(z, H0, Ωm) rtol = 1e-4
    end

    f = Ωm_dual -> begin
        c = CosmologyCache(Cosmology(H0, Ωm_dual), collect(LinRange(0.0, 10.0, 257)))
        luminosity_distance(1.2, c)
    end
    @test isfinite(ForwardDiff.derivative(f, Ωm))
end

@testset "CumulativeIntegral1D" begin
    @testset "analytic linear antiderivative on smooth integrand" begin
        x = collect(LinRange(0.0, 2π, 513))
        r = CumulativeIntegral1D(x, sin)
        # ∫₀^{2π} sin = 0 (grid-aligned endpoint, trapezoidal is exact at nodes)
        @test isapprox(normalizer(r), 0.0; atol = 1e-10)
        # ∫₀^{π} sin ≈ 2 (trapezoidal antiderivative on a 513-node grid)
        @test isapprox(cdf(r, π), 2.0; rtol = 1e-4)
        # Interpolant matches at nodes and between nodes
        @test interpolate(r, π / 2) ≈ sin(π / 2) atol = 1e-8
        # Outside the grid: DataInterpolations throws an extrapolation error
        @test_throws Exception interpolate(r, 2π + 0.1)
        # cdf clamps outside the grid instead of throwing
        @test cdf(r, -1.0) == 0.0
        @test cdf(r, 2π + 0.1) == normalizer(r)
    end

    @testset "cdf agrees with quadgk on cosmology kernel (trapezoidal bound)" begin
        Ωm = 0.315
        inv_E = w -> inv(E(w, Ωm))
        x = collect(LinRange(0.0, 20.0, 1024))
        r = CumulativeIntegral1D(x, inv_E)
        # Trapezoidal antiderivative error on a 1024-node grid is ≤ 1e-4 everywhere.
        for z in (1e-3, 0.05, 0.17, 1.0, 3.14, 9.87, 19.5)
            expected, _ = quadgk(inv_E, 0.0, z; rtol = 1e-10)
            @test cdf(r, z) ≈ expected rtol = 1e-4
        end
    end

    @testset "cdf uses exact within-cell linear antiderivative" begin
        x = [0.0, 1.0, 2.0]
        r = CumulativeIntegral1D(x, z -> 2.0 + 3.0z)
        @test cdf(r, 0.25) ≈ 2.0 * 0.25 + 0.5 * 3.0 * 0.25^2
        @test cdf(r, 1.5) ≈ cdf(r, 1.0) + 1.0 * (5.0 * 0.5 + 0.5 * 3.0 * 0.5^2)
    end

    @testset "luminosity_distance overload matches scalar path" begin
        H0, Ωm = 67.0, 0.315
        x = collect(LinRange(0.0, 10.0, 1024))
        dist = CumulativeIntegral1D(x, w -> inv(E(w, Ωm)))
        for z in (0.05, 0.3, 1.2, 4.5, 8.0)
            @test luminosity_distance(z, H0, Ωm, dist) ≈
                  luminosity_distance(z, H0, Ωm) rtol = 1e-4
            @test differential_comoving_volume(z, H0, Ωm, dist) ≈
                  differential_comoving_volume(z, H0, Ωm) rtol = 1e-4
        end
    end

    @testset "ForwardDiff Duals propagate through CumulativeIntegral1D" begin
        x = collect(LinRange(0.0, 10.0, 257))
        # Derivative of d_c(H0, Ωm) w.r.t. Ωm, evaluated at a catalog z.
        f = Ωm -> begin
            dist = CumulativeIntegral1D(x, w -> inv(E(w, Ωm)))
            luminosity_distance(1.2, 67.0, Ωm, dist)
        end
        grad = ForwardDiff.derivative(f, 0.315)
        @test isfinite(grad)
    end
end
