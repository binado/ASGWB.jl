using ASGWB: HyperParameters, Detector
using DelimitedFiles
using JSON3

struct Interval
    low::Float64
    high::Float64
end

struct PriorBounds
    H0::Interval
    Ωm::Interval
    Ξ₀::Interval
    Ξₙ::Interval
    γ::Interval
    κ::Interval
    zpeak::Interval
end

struct InitPoint
    H0::Float64
    Ωm::Float64
    Ξ₀::Float64
    Ξₙ::Float64
    γ::Float64
    κ::Float64
    zpeak::Float64
end

struct SamplerConfig
    n_samples::Int
    n_adapts::Int
    target_acceptance::Float64
end

struct Settings
    cache::String
    detectors::Vector{Detector}
    priors::PriorBounds
    init::InitPoint
    sampler::SamplerConfig
    seed::Union{Nothing, Int}
    observed_spectral_density_csv::Union{Nothing, String}
    output_jls::Union{Nothing, String}
    """If non-empty, only these hyperparameters are sampled; others are fixed via Turing `|` conditioning."""
    sample_only::Union{Nothing, Vector{Symbol}}
end

function _interval(obj, path::AbstractString)
    low = Float64(obj.low)
    high = Float64(obj.high)
    isfinite(low) && isfinite(high) ||
        throw(ArgumentError("$path: low and high must be finite"))
    low < high || throw(ArgumentError("$path: require low < high, got ($low, $high)"))
    return Interval(low, high)
end

"""Map JSON / legacy ASCII hyperparameter names to package `Symbol`s (`DEFAULT_PARAMETER_ORDER`)."""
const _LEGACY_SAMPLE_ONLY = Dict{Symbol, Symbol}(
    :Omega_m => :Ωm,
    :chi0 => :Ξ₀,
    :chin => :Ξₙ,
    :gamma => :γ,
    :kappa => :κ,
    :z_peak => :zpeak
)

function _canonical_hyper_symbol(s::Symbol)
    return get(_LEGACY_SAMPLE_ONLY, s, s)
end

function _prior_bounds(raw)
    haskey(raw, "priors") || throw(ArgumentError("config: missing key \"priors\""))
    p = raw.priors
    return PriorBounds(
        _interval(p.H0, "priors.H0"),
        _interval(p.Omega_m, "priors.Omega_m"),
        _interval(p.chi0, "priors.chi0"),
        _interval(p.chin, "priors.chin"),
        _interval(p.gamma, "priors.gamma"),
        _interval(p.kappa, "priors.kappa"),
        _interval(p.z_peak, "priors.z_peak")
    )
end

function _init_point(raw)
    haskey(raw, "init") || throw(ArgumentError("config: missing key \"init\""))
    z = raw.init
    return InitPoint(
        Float64(z.H0),
        Float64(z.Omega_m),
        Float64(z.chi0),
        Float64(z.chin),
        Float64(z.gamma),
        Float64(z.kappa),
        Float64(z.z_peak)
    )
end

function _sampler(raw)
    haskey(raw, "sampler") || throw(ArgumentError("config: missing key \"sampler\""))
    s = raw.sampler
    n_samples = Int(s.n_samples)
    n_adapts = Int(s.n_adapts)
    target_acceptance = Float64(s.target_acceptance)
    n_samples > 0 || throw(ArgumentError("sampler.n_samples must be positive"))
    n_adapts >= 0 || throw(ArgumentError("sampler.n_adapts must be non-negative"))
    0 < target_acceptance < 1 ||
        throw(ArgumentError("sampler.target_acceptance must be in (0, 1)"))
    return SamplerConfig(n_samples, n_adapts, target_acceptance)
end

function _optional_string(raw, key::AbstractString)
    haskey(raw, key) || return nothing
    v = raw[key]
    v === nothing && return nothing
    return String(v)
end

function _optional_int(raw, key::AbstractString)
    haskey(raw, key) || return nothing
    v = raw[key]
    v === nothing && return nothing
    return Int(v)
end

function _optional_sample_only(raw)
    !haskey(raw, "sample_only") && return nothing
    v = raw.sample_only
    v === nothing && return nothing
    v isa AbstractVector ||
        throw(ArgumentError("config.sample_only must be a JSON array of strings or null"))
    isempty(v) && return nothing
    return [_canonical_hyper_symbol(Symbol(String(x))) for x in v]
end

function _detectors_from_config(raw)
    if !haskey(raw, "detectors")
        return [Detector("H1"), Detector("L1")]
    end
    d = raw.detectors
    d isa AbstractVector || throw(ArgumentError("config.detectors must be a JSON array"))
    length(d) >= 2 ||
        throw(ArgumentError("config.detectors must list at least two detectors"))
    return [Detector(String(name)) for name in d]
end

"""
    load_settings(path::AbstractString) -> Settings

Read a JSON config file. Expected top-level keys: `cache`, `priors`, `init`, `sampler`;
optional: `detectors` (array of detector name strings, default `[\"H1\", \"L1\"]`), `seed`,
`observed_spectral_density_csv`, `output_jls`, `sample_only` (array of hyperparameter names
to sample; if set, all others are fixed to `init` via Turing conditioning). JSON prior/init
keys remain ASCII (`Omega_m`, `gamma`, …); `sample_only` entries may use those names or
the package Unicode symbols (`Ωm`, `Ξ₀`, `γ`, …).
"""
function load_settings(path::AbstractString)
    isfile(path) || throw(ArgumentError("config file not found: $(repr(path))"))
    raw = JSON3.read(read(path, String))
    raw isa JSON3.Object || throw(ArgumentError("config must be a JSON object"))
    haskey(raw, "cache") || throw(ArgumentError("config: missing key \"cache\""))
    cache = String(raw.cache)
    isempty(strip(cache)) && throw(ArgumentError("config.cache must be a non-empty path"))
    return Settings(
        cache,
        _detectors_from_config(raw),
        _prior_bounds(raw),
        _init_point(raw),
        _sampler(raw),
        _optional_int(raw, "seed"),
        _optional_string(raw, "observed_spectral_density_csv"),
        _optional_string(raw, "output_jls"),
        _optional_sample_only(raw)
    )
end

function validate_init_in_priors(s::Settings)
    nt = (
        H0 = (s.init.H0, s.priors.H0),
        Ωm = (s.init.Ωm, s.priors.Ωm),
        Ξ₀ = (s.init.Ξ₀, s.priors.Ξ₀),
        Ξₙ = (s.init.Ξₙ, s.priors.Ξₙ),
        γ = (s.init.γ, s.priors.γ),
        κ = (s.init.κ, s.priors.κ),
        zpeak = (s.init.zpeak, s.priors.zpeak)
    )
    for (name, (v, b)) in pairs(nt)
        b.low <= v <= b.high || throw(
            ArgumentError("init.$name = $v is outside prior bounds [$(b.low), $(b.high)]"),
        )
    end
    return nothing
end

function prior_dict(s::Settings)
    p = s.priors
    return Dict{String, Tuple{Float64, Float64}}(
        "H0" => (p.H0.low, p.H0.high),
        "Omega_m" => (p.Ωm.low, p.Ωm.high),
        "chi0" => (p.Ξ₀.low, p.Ξ₀.high),
        "chin" => (p.Ξₙ.low, p.Ξₙ.high),
        "gamma" => (p.γ.low, p.γ.high),
        "kappa" => (p.κ.low, p.κ.high),
        "z_peak" => (p.zpeak.low, p.zpeak.high)
    )
end

function theta0(s::Settings)
    z = s.init
    return HyperParameters(;
        H0 = z.H0,
        Ωm = z.Ωm,
        Ξ₀ = z.Ξ₀,
        Ξₙ = z.Ξₙ,
        γ = z.γ,
        κ = z.κ,
        zpeak = z.zpeak
    )
end

function load_observed_spectral_density(path::AbstractString, expected_len::Int)
    isfile(path) || throw(ArgumentError("observed spectrum file not found: $(repr(path))"))
    v = vec(readdlm(path, ',', Float64))
    length(v) == expected_len || throw(
        ArgumentError(
        "observed_spectral_density_csv has length $(length(v)), expected $expected_len",
    ),
    )
    return v
end

"""
    parse_sample_only_cli(s::AbstractString) -> Union{Nothing,Vector{Symbol}}

Parse a comma-separated list of hyperparameter names (e.g. `\"H0\"` or `\"H0,Omega_m\"`;
Unicode symbols such as `Ωm` are accepted). Empty or whitespace-only string means “use config file / no CLI override”.
"""
function parse_sample_only_cli(s::AbstractString)::Union{Nothing, Vector{Symbol}}
    t = strip(s)
    isempty(t) && return nothing
    parts = split(t, ','; keepempty = false)
    return [_canonical_hyper_symbol(Symbol(strip(p))) for p in parts]
end

function merge_settings(
        s::Settings;
        cache::Union{Nothing, String} = nothing,
        detectors::Union{Nothing, Vector{Detector}} = nothing,
        n_samples::Union{Nothing, Int} = nothing,
        n_adapts::Union{Nothing, Int} = nothing,
        target_acceptance::Union{Nothing, Float64} = nothing,
        seed::Union{Nothing, Int} = nothing,
        observed_spectral_density_csv::Union{Nothing, String} = nothing,
        output_jls::Union{Nothing, String} = nothing,
        sample_only::Union{Nothing, Vector{Symbol}} = nothing
)
    sam = s.sampler
    new_sampler = SamplerConfig(
        something(n_samples, sam.n_samples),
        something(n_adapts, sam.n_adapts),
        something(target_acceptance, sam.target_acceptance)
    )
    so = if sample_only === nothing
        s.sample_only
    else
        _canonical_hyper_symbol.(sample_only)
    end
    return Settings(
        something(cache, s.cache),
        detectors === nothing ? s.detectors : detectors,
        s.priors,
        s.init,
        new_sampler,
        seed === nothing ? s.seed : seed,
        observed_spectral_density_csv === nothing ? s.observed_spectral_density_csv :
        observed_spectral_density_csv,
        output_jls === nothing ? s.output_jls : output_jls,
        so
    )
end
