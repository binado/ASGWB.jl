using ASGWB
using ASGWB:
             Detector,
             PowerSpectralDensity,
             effective_psd,
             gaussian_bin_scale,
             frequency_bin_width,
             build_observation_config,
             default_detector_data_dir,
             overlap_reduction_function,
             pairwise_overlap_reduction_function
using HDF5
using NPZ
using Test

# Reference arrays from GWFast: `test/fixtures/orf_gwfast_reference.npz`.
# Regenerate from the repo root with::
#   uv run --script scripts/generate_orf_fixtures.py
# (Same logic as Python `asgwb/scripts/generate_orf_fixtures.py`, pinned gwfast.)

@testset "ORF matches Python asgwb GWFast reference fixture" begin
    path = joinpath(@__DIR__, "fixtures", "orf_gwfast_reference.npz")
    if !isfile(path)
        # Fixture is not committed; copy from Python `asgwb` (see comment above).
        @test_skip false
    else
        data = NPZ.npzread(path)
        freqs = Vector{Float64}(vec(data["frequencies"]))
        pairs = (("H1", "L1", "H1_L1"), ("H1", "V1", "H1_V1"), ("L1", "V1", "L1_V1"))
        for (n1, n2, key) in pairs
            ref = Vector{Float64}(vec(data[key]))
            d1 = Detector(n1)
            d2 = Detector(n2)
            jl = overlap_reduction_function(freqs, d1, d2)
            @test size(jl) == size(ref)
            @test jl≈ref atol=1.0e-4 rtol=1.0e-6
        end
        dets = [Detector("H1"), Detector("L1"), Detector("V1")]
        pw = pairwise_overlap_reduction_function(freqs, dets)
        @test pw[1, 2, :]≈Vector{Float64}(vec(data["H1_L1"])) atol=1.0e-4 rtol=1.0e-6
        @test pw[1, 3, :]≈Vector{Float64}(vec(data["H1_V1"])) atol=1.0e-4 rtol=1.0e-6
        @test pw[2, 3, :]≈Vector{Float64}(vec(data["L1_V1"])) atol=1.0e-4 rtol=1.0e-6
    end
end

@testset "PowerSpectralDensity file load" begin
    noise = joinpath(default_detector_data_dir(), "noise_curves", "AplusDesign_psd.txt")
    psd_obj = PowerSpectralDensity(noise; curve_type = :psd)
    x_min = first(psd_obj.frequency)
    x_max = last(psd_obj.frequency)
    ε = 1.0e-6 * (x_max - x_min)
    v = psd_obj([20.0, 100.0, 1e6])
    @test isfinite(v[1]) && isfinite(v[2])
    @test v[3] == Inf
    @test psd_obj(x_min) == psd_obj.psd[1]
    @test psd_obj(x_max) == psd_obj.psd[end]
    @test psd_obj(x_min - ε) == Inf
    @test psd_obj(x_max + ε) == Inf
end

@testset "Detector from vendored TOML" begin
    d = Detector("H1")
    @test d.name == "H1"
    @test d.length == 4.0
    @test_throws ArgumentError Detector("NONEXISTENT_DETECTOR_XYZ")
    v = Detector("V1")
    @test v.name == "V1"
    @test v.length == 3.0
end

@testset "effective_psd and gaussian_bin_scale" begin
    d1 = Detector("H1")
    d2 = Detector("L1")
    f = collect(range(20.0; step = 20.0, length = 16))
    eff = effective_psd(f, [d1, d2])
    @test length(eff) == length(f)
    @test all(isfinite, eff) && all(eff .> 0)
    mask = trues(length(f))
    scale = gaussian_bin_scale(;
        effective_psd = eff,
        frequencies = f,
        in_band_mask = mask,
        observation_time_sec = 3.15576e7
    )
    @test length(scale) == count(mask)
    @test all(isfinite, scale) && all(scale .> 0)
end

@testset "load_cache reconstructs effective_psd from detectors" begin
    path = parity_cache_path(:posterior_v2_minimal)
    isfile(path) || error("missing fixture $path")
    d1 = Detector("H1")
    d2 = Detector("L1")
    p = load_cache(path, [d1, d2])
    @test length(p.observation.effective_psd) == length(p.observation.frequencies)
    @test all(isfinite, p.observation.effective_psd)
    @test length(p.observation.sgwb_scale) == length(p.observation.frequencies)
end

@testset "load_cache is deterministic for the same path and detectors" begin
    path = parity_cache_path(:posterior)
    dets = [Detector("H1"), Detector("L1")]
    p1 = load_cache(path, dets)
    p2 = load_cache(path, dets)
    @test p1.observation.effective_psd == p2.observation.effective_psd
    @test p1.observation.sgwb_scale == p2.observation.sgwb_scale
end
