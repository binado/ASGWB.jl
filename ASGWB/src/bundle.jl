using HDF5
using SHA: sha256

"""HDF5 `proposal_samples` group attribute naming the compact-object proposal class."""
const PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR = "source_type"

"""`proposal_samples` / [`PROPOSAL_SAMPLES_SOURCE_TYPE_ATTR`](@ref) value for BNS importance samples."""
const PROPOSAL_SAMPLES_SOURCE_TYPE_BNS = "BNS"

"""
    FrequencyGrid

Frequency-axis specification mirroring the Python `FrequencyGrid` dataclass in
`scripts/generate_waveforms.py`. `frequencies` and `in_band_mask` are derived
properties; the five scalars are the canonical on-disk representation.
"""
struct FrequencyGrid
    duration::Float64
    sampling_frequency::Float64
    reference_frequency::Float64
    minimum_frequency::Float64
    maximum_frequency::Float64
end

"""
    frequencies(g::FrequencyGrid) -> Vector{Float64}

Full frequency axis: `linspace(0, f_nyquist, nsamples÷2+1)`.
"""
function frequencies(g::FrequencyGrid)
    nsamples = round(Int, g.duration * g.sampling_frequency)
    nfreq = nsamples ÷ 2 + 1
    return collect(range(0.0, g.sampling_frequency / 2.0; length = nfreq))
end

"""
    in_band_mask(g::FrequencyGrid) -> BitVector

`true` for frequency bins satisfying `f_min ≤ f ≤ f_max`.
"""
function in_band_mask(g::FrequencyGrid)
    f = frequencies(g)
    return BitVector(@. g.minimum_frequency <= f <= g.maximum_frequency)
end

"""
    WaveformMetadata

Provenance and grid metadata stored in a [`WaveformCatalog`](@ref) bundle.
"""
struct WaveformMetadata
    approximant::String
    source_type::Symbol
    grid::FrequencyGrid
    model_sha256::String
    git_revision::String
    command::String
end

"""
    WaveformCatalog{S<:NamedTuple}

On-disk waveform catalog: per-sample parameter vectors paired with per-sample
per-frequency fluxes `|h_+|² + |h_×|²` (before the fiducial `(D_L/D_gw)²` scaling).

`samples` is a NamedTuple whose keys are the parameter column names (e.g.
`:mass_1_source`, `:redshift`, `:luminosity_distance`, …). `fluxes` has shape
`(n_freq, n_samples)` — column-major friendly for the importance-sampling hot loop.

"""
struct WaveformCatalog{S <: NamedTuple}
    samples::S
    fluxes::Matrix{Float64}
    metadata::WaveformMetadata
end

n_samples(c::WaveformCatalog) = size(c.fluxes, 2)
n_freq(c::WaveformCatalog) = size(c.fluxes, 1)

"""
    verify_model_fingerprint(catalog, model_path)

Assert that the SHA-256 of `model_path` matches `catalog.metadata.model_sha256`.
"""
function verify_model_fingerprint(
        catalog::WaveformCatalog,
        model_path::AbstractString
)
    expected = catalog.metadata.model_sha256
    actual = bytes2hex(sha256(read(model_path)))
    expected == actual || throw(
        ArgumentError(
        "model fingerprint mismatch: bundle expects sha256=$(expected) " *
        "but $(basename(model_path)) has sha256=$(actual); " *
        "rebuild the bundle or use the matching model.toml",
    ),
    )
    return nothing
end

"""
    save_bundle(path, catalog)

Write `catalog` to an HDF5 bundle file at `path`.

HDF5 layout:
- `/samples/<column_name>` — one float64 dataset per sample column
- `/fluxes` — float64 matrix of shape `(n_freq, n_samples)`
- root attributes: `approximant`, `source_type`, `grid_*`, `model_sha256`,
  `git_revision`, `command`
"""
function save_bundle(path::AbstractString, catalog::WaveformCatalog)
    h5open(path, "w") do f
        sg = create_group(f, "samples")
        for (k, v) in pairs(catalog.samples)
            write(sg, String(k), collect(Float64, v))
        end
        write(f, "fluxes", catalog.fluxes)
        a = attributes(f)
        m = catalog.metadata
        a["approximant"] = m.approximant
        a["source_type"] = String(m.source_type)
        g = m.grid
        a["grid_duration"] = g.duration
        a["grid_sampling_frequency"] = g.sampling_frequency
        a["grid_reference_frequency"] = g.reference_frequency
        a["grid_minimum_frequency"] = g.minimum_frequency
        a["grid_maximum_frequency"] = g.maximum_frequency
        a["model_sha256"] = m.model_sha256
        a["git_revision"] = m.git_revision
        a["command"] = m.command
    end
    return nothing
end

function _read_bundle_attr(attrs, name::AbstractString)
    haskey(attrs, name) ||
        throw(ArgumentError("bundle.h5 missing required attribute: $(name)"))
    return read(attrs[name])
end

"""
    load_bundle(path) -> WaveformCatalog

Read a bundle HDF5 file into a [`WaveformCatalog`](@ref). The sample columns are
returned in the order they appear in the HDF5 `/samples` group.
"""
function load_bundle(path::AbstractString)::WaveformCatalog
    h5open(path, "r") do f
        a = attributes(f)
        approx = String(_read_bundle_attr(a, "approximant"))
        src_type = Symbol(String(_read_bundle_attr(a, "source_type")))
        grid = FrequencyGrid(
            Float64(_read_bundle_attr(a, "grid_duration")),
            Float64(_read_bundle_attr(a, "grid_sampling_frequency")),
            Float64(_read_bundle_attr(a, "grid_reference_frequency")),
            Float64(_read_bundle_attr(a, "grid_minimum_frequency")),
            Float64(_read_bundle_attr(a, "grid_maximum_frequency"))
        )
        model_sha = String(_read_bundle_attr(a, "model_sha256"))
        git_rev = String(_read_bundle_attr(a, "git_revision"))
        cmd = String(_read_bundle_attr(a, "command"))
        metadata = WaveformMetadata(approx, src_type, grid, model_sha, git_rev, cmd)

        haskey(f, "samples") || throw(ArgumentError("bundle.h5 missing /samples group"))
        sg = f["samples"]
        raw_keys = keys(sg)
        sample_keys = Tuple(Symbol(k) for k in raw_keys)
        sample_vecs = Tuple(Vector{Float64}(read(sg[k])) for k in raw_keys)
        samples = NamedTuple{sample_keys}(sample_vecs)

        haskey(f, "fluxes") || throw(ArgumentError("bundle.h5 missing /fluxes dataset"))
        fluxes = Matrix{Float64}(read(f["fluxes"]))

        return WaveformCatalog(samples, fluxes, metadata)
    end
end
