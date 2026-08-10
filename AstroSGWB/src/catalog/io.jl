import PlusCross

"""
    load_catalog(path) -> SGWBCatalog

Read a `waveform_catalog` v1 HDF5 file (see `SPEC.md` in the `pluscross`
repository) and reduce it to the quantities SGWB inference needs.

The file stores the fundamental artifact -- complex `h₊`/`h×` under
`/polarizations` -- so the polarization power `|h₊|² + |h×|²` is computed here
rather than read. This is the same file the Python `astrogwb` package consumes,
so both implementations see identical fluxes for a given catalog.

Validation and the format/version checks belong to `PlusCross.load_catalog`;
this function only reduces. Sample columns are returned in the order
they appear in the HDF5 `/source_parameters` group. The file's
`minimum_frequency`/`maximum_frequency` band edges are not carried over:
callers that want a restricted band slice `frequencies` and the rows of
`fluxes` themselves before calling a model.
"""
function load_catalog(path::AbstractString)::SGWBCatalog
    c = PlusCross.load_catalog(path)
    fluxes = abs2.(c.plus) .+ abs2.(c.cross)
    return SGWBCatalog(c.frequencies, fluxes, c.source_parameters, c.approximant)
end
