using QuadGK
using Test
using ForwardDiff
using CBCDistributions: CumulativeIntegral1D, cdf, hubble_constant_si, interpolate,
                        normalizer, cosmology_parameters, cosmology, cosmology_type,
                        cosmology_config_name, SUPPORTED_COSMOLOGIES, comoving_distance,
                        W0CDM, W0WaCDM, ModifiedPropagation, hyperparameters,
                        base_cosmology

@testset "hubble_constant_si" begin
    H0 = 70.0
    @test hubble_constant_si(H0) ≈ Float64(H0) * 1000.0 / 3.085677581e22
end

@testset "basic cosmology helpers" begin
    c = LambdaCDM(67.0, 0.315)
    @test H0(c) == 67.0
    @test Ωm(c) == 0.315

    @test E(0.0, c) ≈ 1.0
    @test comoving_distance(0.0, c) ≈ 0.0

    z = [0.0, 0.1, 0.2]
    d_l = luminosity_distance.(z, c)
    @test d_l[1] ≈ 0.0
    @test d_l[3] > d_l[2] > d_l[1]

    d_gw = gravitational_wave_distance.([0.1, 0.2], [10.0, 20.0], 1.0, 0.0)
    @test d_gw ≈ [10.0, 20.0]
end

@testset "cosmology_parameters and cosmology" begin
    @test cosmology_parameters(LambdaCDM) == (:H0, :Ωm)
    @test cosmology_parameters(W0CDM) == (:H0, :Ωm, :w0)
    @test cosmology_parameters(W0WaCDM) == (:H0, :Ωm, :w0, :wa)
    @test hyperparameters(ModifiedPropagation{LambdaCDM}) == (:H0, :Ωm, :Ξ₀, :Ξₙ)

    h_lcdm = (H0 = 67.0, Ωm = 0.315)
    @test cosmology(LambdaCDM, h_lcdm) == LambdaCDM(67.0, 0.315)
    @test LambdaCDM(h_lcdm) == LambdaCDM(67.0, 0.315)
    @test cosmology(h_lcdm) == LambdaCDM(67.0, 0.315)

    h_w0 = (; h_lcdm..., w0 = -0.9)
    @test cosmology(W0CDM, h_w0) == W0CDM(67.0, 0.315, -0.9)
    @test cosmology(h_w0) == W0CDM(67.0, 0.315, -0.9)

    h_cpl = (; h_w0..., wa = 0.2)
    @test cosmology(W0WaCDM, h_cpl) == W0WaCDM(67.0, 0.315, -0.9, 0.2)
    @test cosmology(h_cpl) == W0WaCDM(67.0, 0.315, -0.9, 0.2)

    @test cosmology_config_name(LambdaCDM) == "LambdaCDM"
    @test cosmology_config_name(ModifiedPropagation{LambdaCDM}) ==
          "ModifiedPropagation{LambdaCDM}"
    @test cosmology_type("W0CDM") === W0CDM
    @test ModifiedPropagation{LambdaCDM} in SUPPORTED_COSMOLOGIES
    @test_throws ArgumentError cosmology_type("not_a_model")

    h_mod = (; h_lcdm..., Ξ₀ = 1.2, Ξₙ = 2.0)
    c_mod = cosmology(ModifiedPropagation{LambdaCDM}, h_mod)
    @test c_mod isa ModifiedPropagation{<:LambdaCDM}
    @test base_cosmology(c_mod) == LambdaCDM(67.0, 0.315)
    @test gravitational_wave_distance(0.5, c_mod) ≈
          gravitational_wave_distance(0.5, luminosity_distance(0.5, c_mod.base), 1.2, 2.0)
end

@testset "dark_energy_eos" begin
    lcdm = LambdaCDM(67.0, 0.3)
    w0cdm = W0CDM(67.0, 0.3, -0.8)
    w0wacdm = W0WaCDM(67.0, 0.3, -0.8, 0.3)
    for z in (0.0, 0.5, 1.0, 2.0)
        @test dark_energy_eos(lcdm, z) ≈ -1.0
        @test dark_energy_eos(w0cdm, z) ≈ -0.8
        @test dark_energy_eos(w0wacdm, z) ≈ -0.8 + 0.3 * z / (1 + z)
    end
end

@testset "de_density_ratio" begin
    lcdm = LambdaCDM(67.0, 0.3)
    w0cdm = W0CDM(67.0, 0.3, -0.8)
    w0wacdm = W0WaCDM(67.0, 0.3, -0.9, 0.2)
    for z in (0.1, 0.5, 1.0, 2.0, 5.0)
        @test de_density_ratio(lcdm, z) ≈ 1.0
        # w0CDM closed form vs quadgk integral
        expected_w0, _ = quadgk(
            zp -> 3 * (1 + (-0.8)) / (1 + zp), 0.0, z; rtol = 1e-10
        )
        @test de_density_ratio(w0cdm, z) ≈ exp(expected_w0) rtol = 1e-10
        # w0waCDM closed form vs quadgk integral
        expected_cpl,
        _ = quadgk(
            zp -> 3 * (1 + dark_energy_eos(w0wacdm, zp)) / (1 + zp), 0.0, z; rtol = 1e-10
        )
        @test de_density_ratio(w0wacdm, z) ≈ exp(expected_cpl) rtol = 1e-8
    end
end

@testset "comoving_distance preserves AD tags at z=0" begin
    c_w0 = W0CDM(67.0, ForwardDiff.Dual(0.315), -0.9)
    @test comoving_distance(0.0, c_w0) ≈ 0.0
    @test comoving_distance(0.0, c_w0) isa ForwardDiff.Dual

    zs = [0.0, 0.1]
    r = comoving_distance.(zs, Ref(c_w0))
    @test all(x -> x isa ForwardDiff.Dual, r)

    c_wa = W0WaCDM(67.0, 0.315, -0.9, ForwardDiff.Dual(0.2))
    @test comoving_distance(0.0, c_wa) ≈ 0.0
    @test comoving_distance(0.0, c_wa) isa ForwardDiff.Dual
end

@testset "dark_energy_eos preserves ForwardDiff derivatives" begin
    f_w0 = w0 -> dark_energy_eos(W0CDM(67.0, 0.3, w0), 0.5)
    @test ForwardDiff.derivative(f_w0, -0.8) ≈ 1.0

    g_w0 = w0 -> E(0.5, W0CDM(67.0, 0.3, w0))
    @test isfinite(ForwardDiff.derivative(g_w0, -0.8))
    @test ForwardDiff.derivative(g_w0, -0.8) != 0.0

    h_wa = wa -> dark_energy_eos(W0WaCDM(67.0, 0.3, -0.9, wa), 0.5)
    @test ForwardDiff.derivative(h_wa, 0.2) ≈ 0.5 / (1 + 0.5)
end

@testset "E(z) reduces to ΛCDM at w0=-1" begin
    lcdm = LambdaCDM(70.0, 0.3)
    w0cdm_lim = W0CDM(70.0, 0.3, -1.0)
    w0wacdm_lim = W0WaCDM(70.0, 0.3, -1.0, 0.0)
    for z in (0.0, 0.1, 0.5, 2.0)
        @test E(z, lcdm) ≈ E(z, w0cdm_lim)
        @test E(z, lcdm) ≈ E(z, w0wacdm_lim)
    end
end

@testset "CosmologyCache distance helpers" begin
    c = LambdaCDM(67.0, 0.315)
    cache = CosmologyCache(c, collect(LinRange(0.0, 10.0, 1024)))
    for z in (0.05, 0.3, 1.2, 4.5, 8.0)
        @test comoving_distance(z, cache) ≈ comoving_distance(z, c) rtol = 1e-4
        @test luminosity_distance(z, cache) ≈ luminosity_distance(z, c) rtol = 1e-4
        @test differential_comoving_volume(z, cache) ≈
              differential_comoving_volume(z, c) rtol = 1e-4
    end

    f = Ωm_dual -> begin
        c2 = CosmologyCache(LambdaCDM(67.0, Ωm_dual), collect(LinRange(0.0, 10.0, 257)))
        luminosity_distance(1.2, c2)
    end
    @test isfinite(ForwardDiff.derivative(f, 0.315))
end

@testset "CosmologyCache W0CDM distances" begin
    w0cdm = W0CDM(67.0, 0.315, -0.9)
    cache = CosmologyCache(w0cdm, collect(LinRange(0.0, 10.0, 1024)))
    for z in (0.05, 0.3, 1.2, 4.5)
        @test comoving_distance(z, cache) ≈ comoving_distance(z, w0cdm) rtol = 1e-4
    end

    f = w0_dual -> begin
        c2 = CosmologyCache(W0CDM(67.0, 0.315, w0_dual), collect(LinRange(0.0, 10.0, 257)))
        luminosity_distance(1.2, c2)
    end
    @test isfinite(ForwardDiff.derivative(f, -0.9))
end

@testset "CumulativeIntegral1D" begin
    @testset "analytic linear antiderivative on smooth integrand" begin
        x = collect(LinRange(0.0, 2π, 513))
        r = CumulativeIntegral1D(x, sin)
        @test isapprox(normalizer(r), 0.0; atol = 1e-10)
        @test isapprox(cdf(r, π), 2.0; rtol = 1e-4)
        @test interpolate(r, π / 2) ≈ sin(π / 2) atol = 1e-8
        @test_throws Exception interpolate(r, 2π + 0.1)
        @test cdf(r, -1.0) == 0.0
        @test cdf(r, 2π + 0.1) == normalizer(r)
    end

    @testset "cdf agrees with quadgk on ΛCDM cosmology kernel" begin
        c = LambdaCDM(67.0, 0.315)
        inv_E = w -> inv(E(w, c))
        x = collect(LinRange(0.0, 20.0, 1024))
        r = CumulativeIntegral1D(x, inv_E)
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

    @testset "luminosity_distance CumulativeIntegral1D overload matches scalar path" begin
        c = LambdaCDM(67.0, 0.315)
        x = collect(LinRange(0.0, 10.0, 1024))
        dist = CumulativeIntegral1D(x, w -> inv(E(w, c)))
        for z in (0.05, 0.3, 1.2, 4.5, 8.0)
            @test luminosity_distance(z, c, dist) ≈ luminosity_distance(z, c) rtol = 1e-4
            @test differential_comoving_volume(z, c, dist) ≈
                  differential_comoving_volume(z, c) rtol = 1e-4
        end
    end

    @testset "ForwardDiff Duals propagate through CumulativeIntegral1D" begin
        x = collect(LinRange(0.0, 10.0, 257))
        f = Ωm -> begin
            c = LambdaCDM(67.0, Ωm)
            dist = CumulativeIntegral1D(x, w -> inv(E(w, c)))
            luminosity_distance(1.2, c, dist)
        end
        @test isfinite(ForwardDiff.derivative(f, 0.315))
    end
end
