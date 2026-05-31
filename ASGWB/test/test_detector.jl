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
    psd = PowerSpectralDensity(noise; curve_type = :psd)
    v = psd([20.0, 100.0, 1e6])
    @test isfinite(v[1]) && isfinite(v[2])
    @test v[3] == Inf
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

@testset "build_model_context reconstructs effective_psd from detectors" begin
    if !@isdefined parity_bundle_dir
        include(joinpath(@__DIR__, "parity_test_cache.jl"))
    end
    d1 = Detector("H1")
    d2 = Detector("L1")
    obs = parity_problem_context(:posterior_v2_minimal, [d1, d2]).ctx.observation
    @test length(obs.effective_psd) == length(obs.frequencies)
    # In-band bins have finite PSD; f=0 Hz (DC) is excluded by in_band_mask and may be Inf.
    @test all(isfinite, obs.effective_psd[obs.in_band_mask])
    @test length(obs.sgwb_scale) == length(obs.frequencies)
end

@testset "build_model_context is deterministic for the same paths and detectors" begin
    if !@isdefined parity_bundle_dir
        include(joinpath(@__DIR__, "parity_test_cache.jl"))
    end
    dets = [Detector("H1"), Detector("L1")]
    obs1 = parity_problem_context(:posterior, dets).ctx.observation
    obs2 = parity_problem_context(:posterior, dets).ctx.observation
    @test obs1.effective_psd == obs2.effective_psd
    @test obs1.sgwb_scale == obs2.sgwb_scale
end
