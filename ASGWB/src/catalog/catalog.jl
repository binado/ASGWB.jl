"""HDF5 root attribute naming the compact-object catalog source class."""
const CATALOG_SOURCE_TYPE_ATTR = "source_type"

"""[`CATALOG_SOURCE_TYPE_ATTR`](@ref) value for BNS waveform catalogs."""
const CATALOG_SOURCE_TYPE_BNS = "BNS"

"""
    WaveformCatalog{S<:NamedTuple}

Waveform catalog payload: per-sample parameter vectors paired with per-sample
per-frequency fluxes `|h_+|² + |h_×|²` (before the fiducial `(D_L/D_gw)²` scaling).

`samples` is a NamedTuple whose keys are the parameter column names (e.g.
`:mass_1_source`, `:redshift`, `:luminosity_distance`, ...). `fluxes` has shape
`(n_freq, n_samples)` -- column-major friendly for the importance-sampling hot loop.
"""
struct WaveformCatalog{S <: NamedTuple}
    samples::S
    fluxes::Matrix{Float64}

    function WaveformCatalog(samples::S, fluxes::AbstractMatrix{<:Real}) where {S <:
                                                                                NamedTuple}
        sample_lengths = [length(v) for v in values(samples)]
        if !isempty(sample_lengths) && !all(==(first(sample_lengths)), sample_lengths)
            throw(ArgumentError("all sample columns must have equal length"))
        end
        n = isempty(sample_lengths) ? 0 : first(sample_lengths)
        size(fluxes, 2) == n ||
            throw(ArgumentError("flux sample count must match sample column length"))
        return new{S}(samples, Matrix{Float64}(fluxes))
    end
end

n_samples(c::WaveformCatalog) = size(c.fluxes, 2)
n_freq(c::WaveformCatalog) = size(c.fluxes, 1)

"""
    WaveformCatalogMetadata

Provenance and grid metadata stored beside a [`WaveformCatalog`](@ref) in a catalog file.
"""
struct WaveformCatalogMetadata
    approximant::String
    source_type::Symbol
    grid::FrequencyGrid
    git_revision::String
    command::String
end

"""
    WaveformCatalogFile

Loaded catalog file containing the pure catalog payload and passive file metadata.
"""
struct WaveformCatalogFile{C <: WaveformCatalog, M <: WaveformCatalogMetadata}
    catalog::C
    metadata::M
end
