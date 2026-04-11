using ASGWB: HyperParameters
using DelimitedFiles
using JSON3

struct Interval
    low::Float64
    high::Float64
end

struct PriorBounds
    H0::Interval
    Omega_m::Interval
    chi0::Interval
    chin::Interval
    gamma::Interval
    kappa::Interval
    z_peak::Interval
end

struct InitPoint
    H0::Float64
    Omega_m::Float64
    chi0::Float64
    chin::Float64
    gamma::Float64
    kappa::Float64
    z_peak::Float64
end

struct SamplerConfig
    n_samples::Int
    n_adapts::Int
    target_acceptance::Float64
end

struct Settings
    cache::String
    priors::PriorBounds
    init::InitPoint
    sampler::SamplerConfig
    seed::Union{Nothing,Int}
    observed_spectral_density_csv::Union{Nothing,String}
    output_jls::Union{Nothing,String}
end

function _interval(obj, path::AbstractString)
    low = Float64(obj.low)
    high = Float64(obj.high)
    isfinite(low) && isfinite(high) || throw(ArgumentError("$path: low and high must be finite"))
    low < high || throw(ArgumentError("$path: require low < high, got ($low, $high)"))
    return Interval(low, high)
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
        _interval(p.z_peak, "priors.z_peak"),
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
        Float64(z.z_peak),
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

"""
    load_settings(path::AbstractString) -> Settings

Read a JSON config file. Expected top-level keys: `cache`, `priors`, `init`, `sampler`;
optional: `seed`, `observed_spectral_density_csv`, `output_jls`.
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
        _prior_bounds(raw),
        _init_point(raw),
        _sampler(raw),
        _optional_int(raw, "seed"),
        _optional_string(raw, "observed_spectral_density_csv"),
        _optional_string(raw, "output_jls"),
    )
end

function validate_init_in_priors(s::Settings)
    nt = (
        H0 = (s.init.H0, s.priors.H0),
        Omega_m = (s.init.Omega_m, s.priors.Omega_m),
        chi0 = (s.init.chi0, s.priors.chi0),
        chin = (s.init.chin, s.priors.chin),
        gamma = (s.init.gamma, s.priors.gamma),
        kappa = (s.init.kappa, s.priors.kappa),
        z_peak = (s.init.z_peak, s.priors.z_peak),
    )
    for (name, (v, b)) in pairs(nt)
        b.low <= v <= b.high || throw(
            ArgumentError(
                "init.$name = $v is outside prior bounds [$(b.low), $(b.high)]",
            ),
        )
    end
    return nothing
end

function prior_dict(s::Settings)
    p = s.priors
    return Dict{String,Tuple{Float64,Float64}}(
        "H0" => (p.H0.low, p.H0.high),
        "Omega_m" => (p.Omega_m.low, p.Omega_m.high),
        "chi0" => (p.chi0.low, p.chi0.high),
        "chin" => (p.chin.low, p.chin.high),
        "gamma" => (p.gamma.low, p.gamma.high),
        "kappa" => (p.kappa.low, p.kappa.high),
        "z_peak" => (p.z_peak.low, p.z_peak.high),
    )
end

function theta0(s::Settings)
    z = s.init
    return HyperParameters(;
        H0 = z.H0,
        Omega_m = z.Omega_m,
        chi0 = z.chi0,
        chin = z.chin,
        gamma = z.gamma,
        kappa = z.kappa,
        z_peak = z.z_peak,
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

function merge_settings(
    s::Settings;
    cache::Union{Nothing,String} = nothing,
    n_samples::Union{Nothing,Int} = nothing,
    n_adapts::Union{Nothing,Int} = nothing,
    target_acceptance::Union{Nothing,Float64} = nothing,
    seed::Union{Nothing,Int} = nothing,
    observed_spectral_density_csv::Union{Nothing,String} = nothing,
    output_jls::Union{Nothing,String} = nothing,
)
    sam = s.sampler
    new_sampler = SamplerConfig(
        something(n_samples, sam.n_samples),
        something(n_adapts, sam.n_adapts),
        something(target_acceptance, sam.target_acceptance),
    )
    return Settings(
        something(cache, s.cache),
        s.priors,
        s.init,
        new_sampler,
        seed === nothing ? s.seed : seed,
        observed_spectral_density_csv === nothing ? s.observed_spectral_density_csv :
        observed_spectral_density_csv,
        output_jls === nothing ? s.output_jls : output_jls,
    )
end
