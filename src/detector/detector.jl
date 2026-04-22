using TOML

"""
    Detector

Interferometer metadata and tabulated PSD used for isotropic SGWB network effective PSD.
Mirrors the Python `asgwb.detector.Detector` schema (angles in degrees).
"""
struct Detector
    name::String
    psd::PowerSpectralDensity
    minimum_frequency::Float64
    maximum_frequency::Float64
    length::Float64
    latitude::Float64
    longitude::Float64
    elevation::Float64
    xarm_azimuth::Float64
    yarm_azimuth::Float64
    xarm_tilt::Float64
    yarm_tilt::Float64
    duty_factor::Float64
end

"""Directory bundled with ASGWB (`assets/detector`)."""
function default_detector_data_dir()::String
    return normpath(joinpath(@__DIR__, "..", "..", "assets", "detector"))
end

function _join_noise_curve(noise_dir::AbstractString, name::AbstractString)
    return normpath(joinpath(noise_dir, basename(String(name))))
end

function Detector(
        data::AbstractDict,
        psd::PowerSpectralDensity;
        name::Union{Nothing, String} = nothing
)
    n = something(name, get(data, "name", nothing))
    n === nothing && throw(ArgumentError("detector entry must include a name"))
    return Detector(
        String(n),
        psd,
        Float64(data["minimum_frequency"]),
        Float64(data["maximum_frequency"]),
        Float64(data["length"]),
        Float64(data["latitude"]),
        Float64(data["longitude"]),
        Float64(data["elevation"]),
        Float64(data["xarm_azimuth"]),
        Float64(data["yarm_azimuth"]),
        Float64(get(data, "xarm_tilt", 0.0)),
        Float64(get(data, "yarm_tilt", 0.0)),
        Float64(data["duty_factor"])
    )
end

"""
    Detector(name; data_dir=default_detector_data_dir(), toml_name="detectors.toml", noise_subdir="noise_curves")

Load a detector definition from the vendored TOML table (same layout as Python `asgwb`).
"""
function Detector(
        name::AbstractString;
        data_dir::AbstractString = default_detector_data_dir(),
        toml_name::AbstractString = "detectors.toml",
        noise_subdir::AbstractString = "noise_curves"
)
    path = joinpath(data_dir, toml_name)
    isfile(path) || throw(ArgumentError("detector table not found: $(repr(path))"))
    table = TOML.parsefile(path)
    haskey(table, "detectors") || throw(ArgumentError("toml missing [detectors] table"))
    detectors = table["detectors"]
    haskey(detectors, String(name)) ||
        throw(ArgumentError("unknown detector name $(repr(name))"))
    row = detectors[String(name)]
    row isa AbstractDict || throw(ArgumentError("invalid detector entry for $(repr(name))"))
    curve = row["default_noise_curve"]
    noise_dir = joinpath(data_dir, noise_subdir)
    psd_path = _join_noise_curve(noise_dir, curve)
    isfile(psd_path) ||
        throw(ArgumentError("noise curve file not found: $(repr(psd_path))"))
    psd = PowerSpectralDensity(psd_path; curve_type = :psd)
    return Detector(row, psd; name = String(name))
end
