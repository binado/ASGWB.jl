using ASGWB: HyperParameters, spectral_snr, spectral_snr_squared
using Statistics
using Test

@testset "omegagw" begin
    H0_kms = 70.0
    # Same km/s/Mpc → s⁻¹ conversion as `ASGWB.hubble_constant_si` (not exported).
    h0_si = Float64(H0_kms) * 1000.0 / 3.085677581e22
    f = [10.0, 20.0]
    sh = [1.0e-45, 2.0e-45]
    pre = 4 * pi^2 / (3 * h0_si^2)
    expected = @. pre * f^3 * sh
    @test omegagw(sh, f, H0_kms) ≈ expected
    θ = HyperParameters(;
        H0 = H0_kms,
        Ωm = 0.3,
        Ξ₀ = 1.0,
        Ξₙ = 0.0,
        γ = 2.0,
        κ = 1.0,
        zpeak = 1.0
    )
    @test omegagw(sh, f, θ) ≈ omegagw(sh, f, H0_kms)
    @test omegagw(1.0e-45, 10.0, H0_kms) ≈ pre * 10.0^3 * 1.0e-45
end

@testset "spectral_density primitive" begin
    fluxes = Float64[1.0 2.0 3.0; 4.0 5.0 6.0]
    rate = 2.5
    n_samples = size(fluxes, 2)

    @testset "unweighted average over samples" begin
        expected = 0.4 .* rate .* vec(mean(fluxes; dims = 2))
        @test spectral_density(fluxes, rate) ≈ expected
    end

    @testset "weighted contraction without weight normalization" begin
        w = [0.5, 1.0, 2.0]
        expected = 0.4 .* rate .* (fluxes * w) ./ n_samples
        @test spectral_density(fluxes, rate; weights = w) ≈ expected
    end

    @testset "uniform weights equal to ones give the unweighted mean" begin
        w = ones(n_samples)
        @test spectral_density(fluxes, rate; weights = w) ≈ spectral_density(fluxes, rate)
    end

    @testset "length mismatch errors from matrix multiply" begin
        @test_throws DimensionMismatch spectral_density(fluxes, rate; weights = [1.0, 2.0])
    end

    @testset "output length matches n_freq" begin
        @test length(spectral_density(fluxes, rate)) == size(fluxes, 1)
        @test length(spectral_density(fluxes, rate; weights = rand(n_samples))) ==
              size(fluxes, 1)
    end
end

@testset "spectral_snr / spectral_snr_squared" begin
    s = [1.0, 2.0, 3.0]
    σ = [0.5, 1.0, 0.25]
    f = [10.0, 20.0, 30.0]
    T = 1.0
    # Same σ as the old `sgwb_scale` path: σ = effective_psd / sqrt(2 T df), df = 10 Hz
    df_bins = f[2] - f[1]
    eff = @. σ * sqrt(2.0 * T * df_bins)
    expected_sq = sum(s .^ 2 ./ σ .^ 2)
    @test spectral_snr_squared(s, eff, f, T, df_bins) ≈ expected_sq
    @test spectral_snr(s, eff, f, T, df_bins) ≈ sqrt(expected_sq)
    @test spectral_snr(s, eff, f, T, df_bins) ≈
          sqrt(spectral_snr_squared(s, eff, f, T, df_bins))

    T1 = 1.0
    df0 = 0.5
    @test spectral_snr_squared([2.0], [4.0], [100.0], T1, df0) == 0.25
    @test spectral_snr([2.0], [4.0], [100.0], T1, df0) == 0.5

    @test_throws DimensionMismatch spectral_snr_squared(
        [1.0, 2.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        T1,
        1.0
    )
    @test isfinite(
        spectral_snr_squared([1.0, 2.0], [1.0, 2.0], [2.0, 1.0], T1, df_bins),
    )
    @test isinf(spectral_snr_squared([1.0, 2.0], [0.0, 1.0], [1.0, 2.0], T1, df_bins))
    r = spectral_snr_squared([1.0, 2.0], [-1.0, 1.0], [1.0, 2.0], T1, df_bins)
    @test r isa Real && isfinite(r)
end
