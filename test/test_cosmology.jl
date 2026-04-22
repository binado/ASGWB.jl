using HDF5
using QuadGK
using Test
using ForwardDiff
using ASGWB: RadialInterpolant

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
            Omega_m = Float64(read(case_group["Omega_m"]))
            chi0 = Float64(read(case_group["chi0"]))
            chin = Float64(read(case_group["chin"]))

            expected_dl = vec(Float64.(read(case_group["luminosity_distance"])))
            expected_dvc = vec(Float64.(read(case_group["differential_comoving_volume"])))
            expected_dgw = vec(Float64.(read(case_group["gravitational_wave_distance"])))

            @test luminosity_distance.(z_grid, H0, Omega_m) ≈ expected_dl rtol = 1e-6
            @test differential_comoving_volume.(z_grid, H0, Omega_m) ≈ expected_dvc rtol = 1e-6
            @test gravitational_wave_distance.(z_grid, expected_dl, chi0, chin) ≈ expected_dgw rtol = 1e-6
        end
    end
end

@testset "RadialInterpolant" begin
    @testset "exact Simpson norm on smooth integrand" begin
        x = collect(LinRange(0.0, 2π, 513))
        r = RadialInterpolant(x, sin)
        # ∫₀^{2π} sin = 0
        @test isapprox(r.norm, 0.0; atol=1e-10)
        # ∫₀^{π} sin = 2
        @test isapprox(ASGWB.integrate(r, π, sin), 2.0; atol=1e-8)
        # Integrand matches at nodes and between nodes
        @test ASGWB.integrand(r, π / 2) ≈ sin(π / 2) atol = 1e-8
        # Outside the grid: Interpolations throws (callers keep evaluations in-bounds via z_max)
        @test_throws BoundsError ASGWB.integrand(r, 2π + 0.1)
    end

    @testset "integrate agrees with quadgk on cosmology kernel" begin
        Omega_m = 0.315
        inv_E = w -> inv(E(w, Omega_m))
        x = collect(LinRange(0.0, 20.0, 1024))
        r = RadialInterpolant(x, inv_E)
        for z in (0.0, 1e-3, 0.05, 0.17, 1.0, 3.14, 9.87, 19.5)
            expected, _ = quadgk(inv_E, 0.0, z; rtol=1e-10)
            @test ASGWB.integrate(r, z, inv_E) ≈ expected rtol = 1e-6
        end
    end

    @testset "luminosity_distance overload matches scalar path" begin
        H0, Omega_m = 67.0, 0.315
        x = collect(LinRange(0.0, 10.0, 1024))
        dist = RadialInterpolant(x, w -> inv(E(w, Omega_m)))
        for z in (0.05, 0.3, 1.2, 4.5, 8.0)
            @test luminosity_distance(z, H0, Omega_m, dist) ≈
                  luminosity_distance(z, H0, Omega_m) rtol = 1e-6
            @test differential_comoving_volume(z, H0, Omega_m, dist) ≈
                  differential_comoving_volume(z, H0, Omega_m) rtol = 1e-6
        end
    end

    @testset "ForwardDiff Duals propagate through RadialInterpolant" begin
        x = collect(LinRange(0.0, 10.0, 257))
        # Derivative of d_c(H0, Ωm) w.r.t. Ωm, evaluated at a catalog z.
        f = Omega_m -> begin
            dist = RadialInterpolant(x, w -> inv(E(w, Omega_m)))
            luminosity_distance(1.2, 67.0, Omega_m, dist)
        end
        grad = ForwardDiff.derivative(f, 0.315)
        @test isfinite(grad)
    end
end
