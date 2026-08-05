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
this function only reduces and re-bands. Sample columns are returned in the order
they appear in the HDF5 `/source_parameters` group.
"""
function load_catalog(path::AbstractString)::SGWBCatalog
    c = PlusCross.load_catalog(path)
    fluxes = abs2.(c.plus) .+ abs2.(c.cross)
    mask = BitVector(
        (c.frequencies .>= c.minimum_frequency) .&
        (c.frequencies .<= c.maximum_frequency)
    )
    return SGWBCatalog(c.frequencies, fluxes, c.source_parameters, mask, c.approximant)
end
