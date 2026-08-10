module Config

using TOML

export MCMCConfig, SamplerConfig, load_config, save_config

"""Current config schema version. Bump on any breaking layout change."""
const SCHEMA_VERSION = 2

"""AD backends the notebook knows how to resolve (mirrors `resolve_adtype`)."""
const SUPPORTED_AD_BACKENDS = ("ForwardDiff",)

"""
    SamplerConfig

NUTS sampler options for a run. `nchains == 0` means "resolve to
`Base.Threads.nthreads()` at run time" — the resolution stays the caller's job
so the config records intent verbatim.
"""
struct SamplerConfig
    nsamples::Int
    nadapts::Int
    target_acceptance::Float64
    ad_backend::String
    nchains::Int
end

"""
    MCMCConfig

Strongly-typed, serializable record of the *data* that defines an MCMC run:
input/output paths, detector network, seed, observation time, sampler options,
fiducial values, and `sample_only`.

It deliberately does **not** capture the priors, population model, cosmology
family, or propagation family — those are code (live objects / types) that stay
hardcoded in the run script. Full reproducibility is therefore "this config TOML +
the git commit of the run script", not the TOML alone.

`detectors` are stored as plain name strings (e.g. `"S1"`); the caller
materializes them with `Detector.(cfg.detectors)`. `fiducials` is a flat
`Symbol => Float64` map matching the flat-NamedTuple hyperparameter convention.
`sample_only` is optional: an absent TOML key decodes to `nothing` (TOML has no
null), and `nothing` is omitted on write.

Schema v2 dropped `local_merger_rate`: it is an ordinary hyperparameter (`R₀`, in
Gpc⁻³ yr⁻¹) and lives in `[fiducials]`, so it is fixed by default and sampled by adding
it to `sample_only` and to the runner's hyperprior. `observation_time` stays -- unlike
the rate it does not cancel, and the Gaussian bin scale and the SNR tracking branch
both read it.

Construct from a parsed dict via `MCMCConfig(d)` or from a file via
[`load_config`](@ref); serialize with [`save_config`](@ref).
"""
struct MCMCConfig
    version::Int
    catalog_path::String
    detectors::Vector{String}
    seed::Int
    observation_time::Float64
    sampler::SamplerConfig
    fiducials::Dict{Symbol, Float64}
    sample_only::Union{Nothing, Vector{Symbol}}
    output_dir::String
    output_prefix::String
end

# Field-wise equality and hashing so round-trip tests and "did the config drift?"
# checks are trivial. Nested `SamplerConfig` and the `Dict`/`Vector` fields all
# compare structurally.
for T in (SamplerConfig, MCMCConfig)
    @eval Base.:(==)(a::$T, b::$T) = all(getfield(a, f) == getfield(b, f)
    for f in fieldnames($T))
    @eval function Base.hash(x::$T, h::UInt)
        h = hash($T, h)
        for f in fieldnames($T)
            h = hash(getfield(x, f), h)
        end
        return h
    end
end

"""
    SamplerConfig(d::AbstractDict)

Build and validate a `SamplerConfig` from a `[sampler]` sub-table.
"""
function SamplerConfig(d::AbstractDict)
    nsamples = Int(d["nsamples"])
    nadapts = Int(d["nadapts"])
    target_acceptance = Float64(d["target_acceptance"])
    ad_backend = String(d["ad_backend"])
    nchains = Int(d["nchains"])

    ad_backend in SUPPORTED_AD_BACKENDS || throw(ArgumentError(
        "unsupported ad_backend $(repr(ad_backend)); supported: $(SUPPORTED_AD_BACKENDS)",
    ))
    0 < target_acceptance < 1 || throw(ArgumentError(
        "target_acceptance must be in (0, 1); got $target_acceptance",
    ))
    nsamples ≥ 0 || throw(ArgumentError("nsamples must be ≥ 0; got $nsamples"))
    nadapts ≥ 0 || throw(ArgumentError("nadapts must be ≥ 0; got $nadapts"))
    nchains ≥ 0 || throw(ArgumentError("nchains must be ≥ 0; got $nchains"))

    return SamplerConfig(nsamples, nadapts, target_acceptance, ad_backend, nchains)
end

"""
    MCMCConfig(d::AbstractDict)

The single validating constructor: every load path (file or in-memory dict)
funnels through here, so there is exactly one place a malformed run is rejected.
"""
function MCMCConfig(d::AbstractDict)
    version = Int(get(d, "version", 0))
    version == SCHEMA_VERSION || throw(ArgumentError(
        "unsupported config version $version; this build supports version $SCHEMA_VERSION",
    ))

    observation_time = Float64(d["observation_time"])
    observation_time > 0 || throw(ArgumentError(
        "observation_time must be > 0; got $observation_time",
    ))
    sampler = SamplerConfig(d["sampler"])
    fiducials = Dict{Symbol, Float64}(Symbol(k) => Float64(v) for (k, v) in d["fiducials"])

    sample_only_raw = get(d, "sample_only", nothing)
    sample_only = sample_only_raw === nothing ? nothing :
                  Vector{Symbol}(Symbol.(sample_only_raw))

    return MCMCConfig(
        version,
        String(d["catalog_path"]),
        Vector{String}(String.(d["detectors"])),
        Int(d["seed"]),
        observation_time,
        sampler,
        fiducials,
        sample_only,
        String(d["output_dir"]),
        String(d["output_prefix"])
    )
end

"""
    load_config(path) -> MCMCConfig

Parse a TOML file and build a validated [`MCMCConfig`](@ref).
"""
load_config(path::AbstractString)::MCMCConfig = MCMCConfig(TOML.parsefile(path))

"""
    save_config(cfg::MCMCConfig, path)

Serialize `cfg` to TOML at `path`. `nothing`-valued optional fields are omitted
(decoded back as `nothing`). Output keys are sorted for stable, diffable files;
Unicode fiducial keys are emitted as quoted keys. Written atomically via a
temp file + `mv`.
"""
function save_config(cfg::MCMCConfig, path::AbstractString)
    d = Dict{String, Any}(
        "version" => cfg.version,
        "catalog_path" => cfg.catalog_path,
        "detectors" => cfg.detectors,
        "seed" => cfg.seed,
        "observation_time" => cfg.observation_time,
        "output_dir" => cfg.output_dir,
        "output_prefix" => cfg.output_prefix,
        "sampler" => Dict{String, Any}(
            "nsamples" => cfg.sampler.nsamples,
            "nadapts" => cfg.sampler.nadapts,
            "target_acceptance" => cfg.sampler.target_acceptance,
            "ad_backend" => cfg.sampler.ad_backend,
            "nchains" => cfg.sampler.nchains
        ),
        "fiducials" => Dict{String, Any}(String(k) => v for (k, v) in cfg.fiducials)
    )
    if cfg.sample_only !== nothing
        d["sample_only"] = String.(cfg.sample_only)
    end

    tmp = path * ".tmp"
    open(tmp, "w") do io
        TOML.print(io, d; sorted = true)
    end
    mv(tmp, path; force = true)
    return nothing
end

end # module Config
