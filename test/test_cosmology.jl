using HDF5
using QuadGK
using Test
using ForwardDiff
using ASGWB: CumulativeIntegral1D, cdf, interpolate, normalizer

@testset "basic cosmology helpers" begin
    @test E(0.0, 0.315) ≈ 1.0
    @test comoving_distance(0.0, 67.0, 0.315) ≈ 0.0

    z = [0.0, 0.1, 0.2]
    d_l = luminosity_distance.(z, 67.0, 0.315)
    @test d_l[1] ≈ 0.0
    @test d_l[3] > d_l[2] > d_l[1]

    d_gw = gravitational_wave_distance.([0.1, 0.2], [10.0, 20.0], 1.0, 0.0)
    @test d_gw ≈ [10.0, 20.0]
end

@testset "cosmology parity fixtures" begin
    fixture_path = joinpath(@__DIR__, "fixtures", "cosmology_parity.h5")

    h5open(fixture_path, "r") do file
        z_grid = vec(Float64.(read(file["z_grid"])))
        cases = file["cases"]

        for case_name in sort!(collect(keys(cases)))
            case_group = cases[case_name]
            H0 = Float64(read(case_group["H0"]))
            Ωm = Float64(read(case_group["Omega_m"]))
            Ξ₀ = Float64(read(case_group["chi0"]))
            Ξₙ = Float64(read(case_group["chin"]))

            expected_dl = vec(Float64.(read(case_group["luminosity_distance"])))
            expected_dvc = vec(Float64.(read(case_group["differential_comoving_volume"])))
            expected_dgw = vec(Float64.(read(case_group["gravitational_wave_distance"])))

            @test luminosity_distance.(z_grid, H0, Ωm) ≈ expected_dl rtol = 1e-6
            @test differential_comoving_volume.(z_grid, H0, Ωm) ≈ expected_dvc rtol = 1e-6
            @test gravitational_wave_distance.(z_grid, expected_dl, Ξ₀, Ξₙ) ≈
                  expected_dgw rtol = 1e-6
        end
    end
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
