using ASGWB: spectral_snr, spectral_snr_squared
using Test

@testset "spectral_snr / spectral_snr_squared" begin
    s = [1.0, 2.0, 3.0]
    σ = [0.5, 1.0, 0.25]
    f = [10.0, 20.0, 30.0]
    T = 1.0
    # Same σ as the old `sgwb_scale` path: σ = effective_psd / sqrt(2 T df), df = 10 Hz
    df_bins = f[2] - f[1]
    eff = @. σ * sqrt(2.0 * T * df_bins)
    expected_sq = sum(s .^ 2 ./ σ .^ 2)
    @test spectral_snr_squared(s, eff, T, df_bins) ≈ expected_sq
    @test spectral_snr(s, eff, T, df_bins) ≈ sqrt(expected_sq)
    @test spectral_snr(s, eff, T, df_bins) ≈
          sqrt(spectral_snr_squared(s, eff, T, df_bins))

    T1 = 1.0
    df0 = 0.5
    @test spectral_snr_squared([2.0], [4.0], T1, df0) == 0.25
    @test spectral_snr([2.0], [4.0], T1, df0) == 0.5

    @test_throws DimensionMismatch spectral_snr_squared(
        [1.0, 2.0],
        [1.0, 2.0, 3.0],
        T1,
        1.0
    )
    @test isfinite(spectral_snr_squared([1.0, 2.0], [1.0, 2.0], T1, df_bins))
    @test isinf(spectral_snr_squared([1.0, 2.0], [0.0, 1.0], T1, df_bins))
    r = spectral_snr_squared([1.0, 2.0], [-1.0, 1.0], T1, df_bins)
    @test r isa Real && isfinite(r)
end
