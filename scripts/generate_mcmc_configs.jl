#!/usr/bin/env julia
# Generate one MCMC TOML config per detector/sample_only sweep point.
#
# Run from the repository root:
#   julia --project=scripts/run scripts/generate_mcmc_configs.jl [output_dir]

module GenerateMCMCConfigsCLI

using AstroSGWBInference: MCMCConfig, SamplerConfig, save_config

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_OUTPUT_DIR = "config/mcmc/sweep"
const CATALOG_PATH = "catalog.h5"

const DETECTOR_NETWORKS = (
    "ET-triangular" => ["E1", "E2", "E3"],
    "ET-triangular-CE-Hanford" => ["E1", "E2", "E3", "C1"],
    "ET-2L-aligned" => ["S1", "R1"],
    "ET-2L-aligned-CE-Hanford" => ["S1", "R1", "C1"],
    "ET-2L-misaligned" => ["S2", "R2"],
    "ET-2L-misaligned-CE-Hanford" => ["S2", "R2", "C1"]
)

const SAMPLE_ONLY_SETS = (
    "H0" => [:H0],
    "Omega_m" => [:H0, :Ωm],
    "w0" => [:H0, :w0],
    "modified-propagation" => [:Ξ₀, :Ξₙ],
    "H0-MD" => [:H0, :γ, :κ, :zpeak],
    "H0-peak" => [:H0, :zpeak],
    "Xi_0-MD" => [:Ξ₀, :γ, :κ, :zpeak]
)

const BASE_SAMPLER = SamplerConfig(
    3000,         # nsamples
    3000,         # nadapts
    0.9,          # target_acceptance
    "ForwardDiff",
    0            # nchains; 0 -> Base.Threads.nthreads() in scripts/run_mcmc.jl
)

const BASE_FIDUCIALS = Dict{Symbol, Float64}(
    :H0 => 67.66,
    :Ωm => 0.3096,
    :w0 => -1.0,
    :Ξ₀ => 1.0,
    :Ξₙ => 1.91,
    :γ => 2.7,
    :κ => 5.7,
    :zpeak => 2.0,
    :R₀ => 161.0
)

function _resolve_output_dir(path::AbstractString)
    return isabspath(path) ? normpath(path) : normpath(joinpath(REPO_ROOT, path))
end

function _config(detectors::Vector{String}, sample_only::Vector{Symbol})
    return MCMCConfig(
        3,
        CATALOG_PATH,
        copy(detectors),
        42,
        1.0,
        BASE_SAMPLER,
        copy(BASE_FIDUCIALS),
        copy(sample_only),
        # The sweep is a default-likelihood sweep; amplitude marginalization is opted into
        # per-config, since it changes which parameter has a latent variable at all.
        "default",
        nothing,
        1024,
        10.0,
        "chains",
        "chains"
    )
end

function generate_configs(output_dir::AbstractString = DEFAULT_OUTPUT_DIR)
    resolved_output_dir = _resolve_output_dir(output_dir)
    mkpath(resolved_output_dir)

    written = String[]
    skipped = String[]

    for (network_label, detectors) in DETECTOR_NETWORKS
        for (sample_label, sample_only) in SAMPLE_ONLY_SETS
            filename = "$(network_label)__$(sample_label).toml"
            path = joinpath(resolved_output_dir, filename)
            if ispath(path)
                push!(skipped, path)
                @info "skipping existing config" path
                continue
            end

            cfg = _config(detectors, sample_only)
            save_config(cfg, path)
            push!(written, path)
            @info "wrote config" path detectors sample_only
        end
    end

    @info "done" output_dir=resolved_output_dir written=length(written) skipped=length(skipped)
    return (; output_dir = resolved_output_dir, written, skipped)
end

end # module GenerateMCMCConfigsCLI

function (@main)(args::Vector{String})
    length(args) <= 1 || throw(
        ArgumentError("usage: generate_mcmc_configs.jl [output_dir]")
    )
    output_dir = isempty(args) ? GenerateMCMCConfigsCLI.DEFAULT_OUTPUT_DIR : only(args)
    GenerateMCMCConfigsCLI.generate_configs(output_dir)
    return 0
end
