using QuadGK
using Test
using ForwardDiff
using Cosmology: CumulativeIntegral1D, cdf, hubble_constant_si, interpolate,
                 normalizer, cosmology, cosmology_type, cosmology_config_name,
                 SUPPORTED_COSMOLOGIES, comoving_distance, W0CDM, W0WaCDM,
                 GR, ModifiedPropagation, hyperparameters,
                 propagation, propagation_type, propagation_config_name,
                 propagation_hyperparameters, SUPPORTED_PROPAGATIONS

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

    # GW luminosity distance is gw_em_distance_ratio(z, ...) * D_L; GR ⇒ identity.
    d_gw = gw_em_distance_ratio.([0.1, 0.2], 1.0, 0.0) .* [10.0, 20.0]
    @test d_gw ≈ [10.0, 20.0]
end

@testset "gw_em_distance_ratio" begin
    zs = (0.0, 0.3, 1.0, 2.5)
    c = LambdaCDM(67.0, 0.315)

    # GR propagation recovers Ξ ≡ 1, so D_gw = D_L for any background cosmology.
    for cosmo in (LambdaCDM(67.0, 0.315), W0CDM(67.0, 0.315, -0.9),
        W0WaCDM(67.0, 0.315, -0.9, 0.2))
        for z in zs
            @test gw_em_distance_ratio(z, GR()) ≈ 1.0
            @test gw_em_distance_ratio(z, GR()) * luminosity_distance(z, cosmo) ≈
                  luminosity_distance(z, cosmo)
        end
    end

    # ModifiedPropagation applies Ξ(z) = Ξ₀ + (1 - Ξ₀)/(1 + z)^Ξₙ, independent of cosmology.
    Ξ₀, Ξₙ = 1.2, 2.0
    p_mod = ModifiedPropagation(Ξ₀, Ξₙ)
    for z in zs
        Ξ = Ξ₀ + (1 - Ξ₀) / (1 + z)^Ξₙ
        @test gw_em_distance_ratio(z, p_mod) ≈ Ξ
        @test gw_em_distance_ratio(z, Ξ₀, Ξₙ) ≈ Ξ
        # GW luminosity distance is exactly Ξ(z) · D_L.
        @test gw_em_distance_ratio(z, p_mod) * luminosity_distance(z, c) ≈
              gw_em_distance_ratio(z, p_mod) * luminosity_distance(z, c)
    end
end

@testset "apply_gw_distance_correction" begin
    z = [0.1, 0.5, 2.0]
    polarization_power = Float64[1.0 2.0 3.0
                                 4.0 5.0 6.0]
    p_mod = ModifiedPropagation(1.4, 0.7)

    # GR is a true no-op and the bang form hands back the same object.
    gr_in = copy(polarization_power)
    @test apply_gw_distance_correction!(gr_in, z, GR()) === gr_in
    @test gr_in == polarization_power

    # ModifiedPropagation divides column j by Ξ(z[j])².
    expected = reduce(hcat,
        [polarization_power[:, j] ./ gw_em_distance_ratio(z[j], p_mod)^2
         for j in eachindex(z)])
    @test apply_gw_distance_correction(polarization_power, z, p_mod) ≈ expected
    # Ξ₀ = 1 makes ModifiedPropagation exactly the identity too.
    @test apply_gw_distance_correction(polarization_power, z, ModifiedPropagation(1.0, 0.7)) ==
          polarization_power

    # Out-of-place never aliases or mutates its input; the bang form mutates in place.
    untouched = copy(polarization_power)
    out = apply_gw_distance_correction(polarization_power, z, p_mod)
    @test out !== polarization_power
    @test polarization_power == untouched
    bang_target = copy(polarization_power)
    @test apply_gw_distance_correction!(bang_target, z, p_mod) === bang_target
    @test bang_target ≈ expected

    # Not idempotent: a second application squares the factor. Documented, and the
    # reason notebook call sites use the out-of-place form.
    @test apply_gw_distance_correction!(copy(expected), z, p_mod) ≈
          reduce(hcat,
        [polarization_power[:, j] ./ gw_em_distance_ratio(z[j], p_mod)^4
         for j in eachindex(z)])

    # A redshift vector that does not match the polarization-power columns (e.g. a subsetted sample
    # set) is caught rather than silently correcting only a prefix — including under GR.
    @test_throws DimensionMismatch apply_gw_distance_correction!(
        copy(polarization_power), z[1:2],
        p_mod)
    @test_throws DimensionMismatch apply_gw_distance_correction!(
        copy(polarization_power), z[1:2],
        GR())
    @test_throws DimensionMismatch apply_gw_distance_correction(
        polarization_power, [z;
                             3.0], p_mod)
end

@testset "cosmology hyperparameters and cosmology" begin
    @test hyperparameters(LambdaCDM) == (:H0, :Ωm)
    @test hyperparameters(W0CDM) == (:H0, :Ωm, :w0)
    @test hyperparameters(W0WaCDM) == (:H0, :Ωm, :w0, :wa)

    h_lcdm = (H0 = 67.0, Ωm = 0.315)
    @test cosmology(LambdaCDM, h_lcdm) == LambdaCDM(67.0, 0.315)
    @test LambdaCDM(h_lcdm) == LambdaCDM(67.0, 0.315)
    @test cosmology(h_lcdm) == LambdaCDM(67.0, 0.315)
    # Extra propagation keys in `h` are ignored by the cosmology builder.
    @test cosmology(LambdaCDM, (; h_lcdm..., Ξ₀ = 1.2, Ξₙ = 2.0)) == LambdaCDM(67.0, 0.315)

    h_w0 = (; h_lcdm..., w0 = -0.9)
    @test cosmology(W0CDM, h_w0) == W0CDM(67.0, 0.315, -0.9)
    @test cosmology(h_w0) == W0CDM(67.0, 0.315, -0.9)

    h_cpl = (; h_w0..., wa = 0.2)
    @test cosmology(W0WaCDM, h_cpl) == W0WaCDM(67.0, 0.315, -0.9, 0.2)
    @test cosmology(h_cpl) == W0WaCDM(67.0, 0.315, -0.9, 0.2)

    @test cosmology_config_name(LambdaCDM) == "LambdaCDM"
    @test cosmology_type("W0CDM") === W0CDM
    @test Set(SUPPORTED_COSMOLOGIES) == Set((LambdaCDM, W0CDM, W0WaCDM))
    @test_throws ArgumentError cosmology_type("not_a_model")
end

@testset "propagation axis" begin
    @test propagation_hyperparameters(GR) == ()
    @test propagation_hyperparameters(ModifiedPropagation) == (:Ξ₀, :Ξₙ)

    @test propagation(GR, (;)) === GR()
    h_mod = (Ξ₀ = 1.2, Ξₙ = 2.0)
    p_mod = propagation(ModifiedPropagation, h_mod)
    @test p_mod isa ModifiedPropagation
    @test p_mod.Ξ₀ == 1.2 && p_mod.Ξₙ == 2.0
    # Extra (cosmology) keys are ignored when building propagation.
    @test propagation(ModifiedPropagation, (; h_mod..., H0 = 67.0, Ωm = 0.315)) == p_mod

    @test propagation_config_name(GR) == "GR"
    @test propagation_config_name(ModifiedPropagation) == "ModifiedPropagation"
    @test propagation_type("GR") === GR
    @test propagation_type("ModifiedPropagation") === ModifiedPropagation
    @test Set(SUPPORTED_PROPAGATIONS) == Set((GR, ModifiedPropagation))
    @test_throws ArgumentError propagation_type("not_a_model")
end

@testset "ModifiedPropagation promotes mixed eltypes" begin
    # `ModifiedPropagation{T}` shares one type parameter between both fields, so without
    # the promoting outer constructor these are `MethodError`s.
    @test ModifiedPropagation(1.4, 1) === ModifiedPropagation(1.4, 1.0)
    @test propagation(ModifiedPropagation, (Ξ₀ = 1.4, Ξₙ = 1)) ===
          ModifiedPropagation(1.4, 1.0)

    # The shape a partially-sampled run produces: one slot `Dual`, the other `Float64`.
    # `Ξ(z) = Ξ₀ + (1 - Ξ₀)(1 + z)^(-Ξₙ)`, so ∂Ξ/∂Ξ₀ = 1 - (1 + z)^(-Ξₙ).
    z, Ξₙ = 0.7, 1.9
    dΞ = ForwardDiff.derivative(
        Ξ₀ -> gw_em_distance_ratio(z, propagation(ModifiedPropagation, (; Ξ₀, Ξₙ))), 1.4)
    @test dΞ ≈ 1 - (1 + z)^(-Ξₙ)

    # And the mirrored case: `Ξₙ` free, `Ξ₀` fixed.
    dΞₙ = ForwardDiff.derivative(
        Ξₙ -> gw_em_distance_ratio(z, propagation(ModifiedPropagation, (Ξ₀ = 1.4, Ξₙ))),
        1.9)
    @test dΞₙ ≈ -(1 - 1.4) * log(1 + z) * (1 + z)^(-1.9)
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

    @testset "from-values constructor matches function constructor" begin
        c = LambdaCDM(67.0, 0.315)
        x = collect(LinRange(0.0, 10.0, 256))
        inv_E = w -> inv(E(w, c))
        from_fn = CumulativeIntegral1D(x, inv_E)
        from_vals = CumulativeIntegral1D(x, map(inv_E, x))
        @test from_vals.y == from_fn.y
        @test from_vals.cumulative == from_fn.cumulative
        @test_throws ArgumentError CumulativeIntegral1D([0.0], [1.0])
        @test_throws ArgumentError CumulativeIntegral1D(x, [1.0, 2.0])
    end
end

@testset "trapz / cumtrapz" begin
    # Linear integrand: the trapezoidal rule is exact, so compare to the antiderivative.
    x = [0.0, 1.0, 2.0]
    y = @. 2.0 + 3.0 * x
    @test cumtrapz(x, y) ≈ [0.0, 2.0 + 1.5, 4.0 + 6.0]
    @test trapz(x, y) ≈ 2.0 * 2.0 + 1.5 * 2.0^2

    # The load-bearing identity: same accumulation order, so bit-for-bit equal.
    xr = collect(LinRange(1e-3, 20.0, 256))
    yr = @. exp(-xr) * (2 + sin(7xr))
    @test trapz(xr, yr) === last(cumtrapz(xr, yr))
    @test cumtrapz(xr, yr)[1] == 0.0

    # …and equal to what `CumulativeIntegral1D` stores, which now shares the same kernel.
    @test cumtrapz(xr, yr) == CumulativeIntegral1D(xr, yr).cumulative
    @test trapz(xr, yr) === normalizer(CumulativeIntegral1D(xr, yr))

    @test_throws ArgumentError trapz([0.0, 1.0], [1.0])
    @test_throws ArgumentError cumtrapz([0.0, 1.0], [1.0])

    # Duals propagate and the eltype follows `y`.
    yd = ForwardDiff.Dual{Nothing}.(yr, 1.0)
    @test eltype(cumtrapz(xr, yd)) <: ForwardDiff.Dual
    @test trapz(xr, yd) isa ForwardDiff.Dual
end

@testset "distance_and_volume_grid" begin
    z_grid = collect(LinRange(1e-3, 20.0, 256))

    # The load-bearing test: agreement with the `CosmologyCache` scalar path at every
    # node. NOT against quadgk — `luminosity_distance(z, ::AbstractCosmology)` and
    # `luminosity_distance(z, ::CosmologyCache)` differ by ~1% at z = 0.1 because this
    # grid starts at 1e-3 rather than 0, which is a separate (real) systematic.
    for c in (LambdaCDM(67.0, 0.315), W0CDM(67.0, 0.315, -0.9),
        W0WaCDM(67.0, 0.315, -0.9, 0.2))
        g = distance_and_volume_grid(c, z_grid)
        cache = CosmologyCache(c, z_grid)
        @test g.comoving_distance ≈ comoving_distance.(z_grid, Ref(cache)) rtol = 1e-12
        @test g.luminosity_distance ≈ luminosity_distance.(z_grid, Ref(cache)) rtol = 1e-12
        # `d_h · d_c² · inv_E` vs `d_h · d_c² / E(z)`: equal to ~1 ulp, and that ulp is
        # exactly the saved `E(z)` evaluation.
        @test g.differential_comoving_volume ≈
              differential_comoving_volume.(z_grid, Ref(cache)) rtol = 1e-12
    end

    # ForwardDiff through the cosmology parameters, matching the cache path.
    for (name, build, x0) in (
        (:Ωm, v -> LambdaCDM(67.0, v), 0.315),
        (:H0, v -> LambdaCDM(v, 0.315), 67.0)
    )
        f_grid = v -> sum(distance_and_volume_grid(build(v), z_grid).luminosity_distance)
        f_cache = v -> sum(luminosity_distance.(z_grid, Ref(CosmologyCache(build(v),
            z_grid))))
        d = ForwardDiff.derivative(f_grid, x0)
        @test isfinite(d)
        @test d != 0.0
        @test d ≈ ForwardDiff.derivative(f_cache, x0) rtol = 1e-12
    end
end

@testset "GridInterpolator" begin
    c = LambdaCDM(67.0, 0.315)
    z_grid = collect(LinRange(0.0, 2.0, 101))
    ci = CumulativeIntegral1D(z_grid, w -> inv(E(w, c)))
    points = [0.0, 0.137, 0.9, 2.0]

    @testset "batched interpolation matches the scalar verb" begin
        interp = GridInterpolator(points, z_grid)
        # Cross-check against the scalar `interpolate(c, x0)`, the coverage the deleted
        # `interpolate(c, ::GridQuery, i)` testset used to provide.
        @test interp(ci.y) ≈ [interpolate(ci, z) for z in points]
        # Bit-for-bit against the formula itself: `y[i] + t*(y[i+1] - y[i])`, no `muladd`
        # (which could contract to an FMA and shift the last bit).
        expected_ci = map(points) do z
            i = clamp(searchsortedlast(z_grid, z), 1, length(z_grid) - 1)
            t = (z - z_grid[i]) / (z_grid[i + 1] - z_grid[i])
            ci.y[i] + t * (ci.y[i + 1] - ci.y[i])
        end
        @test interp(ci.y) == expected_ci

        # A plain functor over any grid-valued vector, not just `ci.y`.
        g = distance_and_volume_grid(c, z_grid)
        expected = map(points) do z
            i = clamp(searchsortedlast(z_grid, z), 1, length(z_grid) - 1)
            t = (z - z_grid[i]) / (z_grid[i + 1] - z_grid[i])
            y = g.luminosity_distance
            y[i] + t * (y[i + 1] - y[i])
        end
        @test interp(g.luminosity_distance) == expected
    end

    @testset "plain linear d_L, not the exact within-cell antiderivative" begin
        # Decision A. Interpolating the tabulated `d_L` linearly is NOT the same as
        # `luminosity_distance(z, c, ci)`, which integrates 1/E exactly within the cell.
        # Linear-interpolation error is O(Δz²·f″) in absolute terms, but d_L → 0 as
        # z → 0, so the *relative* error carries a 1/z factor and is largest at low z.
        # This is the intended behaviour — it matches the Python stack — so pin it.
        interp = GridInterpolator(points, z_grid)
        g = distance_and_volume_grid(c, z_grid)
        linear = interp(g.luminosity_distance)
        exact = [luminosity_distance(z, c, ci) for z in points]

        # Nodes agree exactly (t == 0 kills both interpolants' cell terms).
        @test linear[1] == exact[1]
        @test linear[4] ≈ exact[4] rtol = 1e-13
        # Mid-cell at low z: linear over-estimates by O(1e-4) relative here (Δz = 0.02);
        # on DEFAULT_Z_GRID (Δz ≈ 0.0784) the same effect reaches O(1e-2).
        @test linear[2] != exact[2]
        @test linear[2] ≈ exact[2] rtol = 1e-3
        @test linear[2] > exact[2]
    end

    @testset "clamps out-of-grid points in both directions" begin
        interp = GridInterpolator([-5.0, 7.0], z_grid)
        y = collect(Float64, 1:101)
        # Clamped, not extrapolated: the grid's own endpoint values come back.
        @test interp(y) == [y[1], y[end]]
        @test_throws ArgumentError GridInterpolator([-0.1], z_grid; check_bounds = true)
        @test_throws ArgumentError GridInterpolator([2.1], z_grid; check_bounds = true)
        @test_throws ArgumentError GridInterpolator([0.5], [1.0])
    end

    @testset "empty points give a concretely-typed empty vector" begin
        interp = GridInterpolator(Float64[], z_grid)
        out = interp(ci.y)
        @test isempty(out)
        @test eltype(out) === Float64
        dual_y = ForwardDiff.Dual{Nothing}.(ci.y, 1.0)
        @test eltype(interp(dual_y)) <: ForwardDiff.Dual
        # Integer-valued grids promote rather than truncating against the Float64 t.
        @test eltype(GridInterpolator([0.5], [0.0, 1.0])(collect(1:2))) === Float64
        @test GridInterpolator([0.5], [0.0, 1.0])(collect(1:2)) == [1.5]
    end

    @testset "ForwardDiff propagates through interpolated values" begin
        interp = GridInterpolator(points, z_grid)
        f = Ωm -> sum(interp(distance_and_volume_grid(LambdaCDM(67.0, Ωm),
            z_grid).luminosity_distance))
        d = ForwardDiff.derivative(f, 0.315)
        @test isfinite(d)
        h = 1e-6
        @test d ≈ (f(0.315 + h) - f(0.315 - h)) / (2h) rtol = 1e-6
        @test eltype(interp(ForwardDiff.Dual{Nothing}.(ci.y, 1.0))) <: ForwardDiff.Dual
    end

    @testset "length mismatch against the grid is caught" begin
        interp = GridInterpolator(points, z_grid)
        @test_throws DimensionMismatch interp(collect(Float64, 1:100))
    end
end
