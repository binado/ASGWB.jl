using HDF5

"""
    save_catalog(path, file)

Write `file` to an HDF5 catalog file at `path`.

HDF5 layout:
- `/samples/<column_name>` -- one float64 dataset per sample column
- `/fluxes` -- float64 matrix of shape `(n_freq, n_samples)`
- root attributes: `approximant`, `source_type`, `grid_*`, `git_revision`, `command`
"""
function save_catalog(path::AbstractString, file::WaveformCatalogFile)
    h5open(path, "w") do f
        sg = create_group(f, "samples")
        for (k, v) in pairs(file.catalog.samples)
            write(sg, String(k), collect(Float64, v))
        end
        write(f, "fluxes", file.catalog.fluxes)
        a = attributes(f)
        m = file.metadata
        a["approximant"] = m.approximant
        a[CATALOG_SOURCE_TYPE_ATTR] = String(m.source_type)
        g = m.grid
        for fld in fieldnames(FrequencyGrid)
            a["grid_$fld"] = getfield(g, fld)
        end
        a["git_revision"] = m.git_revision
        a["command"] = m.command
    end
    return nothing
end

function _read_catalog_attr(attrs, name::AbstractString, catalog_label::AbstractString)
    haskey(attrs, name) ||
        throw(ArgumentError("$(catalog_label) missing required attribute: $(name)"))
    return read(attrs[name])
end

"""
    load_catalog(path) -> WaveformCatalogFile

Read a catalog HDF5 file into a [`WaveformCatalogFile`](@ref). The sample columns are
returned in the order they appear in the HDF5 `/samples` group.
"""
function load_catalog(path::AbstractString)::WaveformCatalogFile
    catalog_label = basename(path)
    h5open(path, "r") do f
        a = attributes(f)
        approx = String(_read_catalog_attr(a, "approximant", catalog_label))
        src_type = Symbol(String(_read_catalog_attr(a, CATALOG_SOURCE_TYPE_ATTR, catalog_label)))
        grid = FrequencyGrid((
            Float64(_read_catalog_attr(a, "grid_$f", catalog_label))
        for f in fieldnames(FrequencyGrid)
        )...)
        git_rev = String(_read_catalog_attr(a, "git_revision", catalog_label))
        cmd = String(_read_catalog_attr(a, "command", catalog_label))
        metadata = WaveformCatalogMetadata(approx, src_type, grid, git_rev, cmd)

        haskey(f, "samples") ||
            throw(ArgumentError("$(catalog_label) missing /samples group"))
        sg = f["samples"]
        raw_keys = keys(sg)
        sample_keys = Tuple(Symbol(k) for k in raw_keys)
        sample_vecs = Tuple(Vector{Float64}(read(sg[k])) for k in raw_keys)
        samples = NamedTuple{sample_keys}(sample_vecs)

        haskey(f, "fluxes") ||
            throw(ArgumentError("$(catalog_label) missing /fluxes dataset"))
        fluxes = Matrix{Float64}(read(f["fluxes"]))

        return WaveformCatalogFile(WaveformCatalog(samples, fluxes), metadata)
    end
end
