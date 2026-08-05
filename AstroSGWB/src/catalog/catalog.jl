"""
    SGWBCatalog{S<:NamedTuple}

Waveform catalog reduced for SGWB inference: the shared frequency axis, per-sample
per-frequency polarization power `|h₊|² + |h×|²` (before the fiducial `(D_L/D_gw)²`
scaling), and the per-sample source parameters that produced it.

`samples` is a NamedTuple whose keys are the source-parameter column names (e.g.
`:mass_1_source`, `:redshift`, `:inclination`, ...). `fluxes` has shape
`(nfreq, nsamples)` -- column-major friendly for the importance-sampling hot loop,
and the orientation HDF5.jl reads the on-disk `(nsamples, nfreq)` C-order
polarization datasets into without a transpose.

`in_band_mask` and `frequencies` come from the file rather than being derived from
`(duration, sampling_frequency)` scalars, so the band convention travels with the
data. Built by [`load_catalog`](@ref) from a `waveform_catalog` v1 file.
"""
struct SGWBCatalog{S <: NamedTuple}
    frequencies::Vector{Float64}
    fluxes::Matrix{Float64}
    samples::S
    in_band_mask::BitVector
    approximant::String
end

nsamples(c::SGWBCatalog) = size(c.fluxes, 2)
nfreq(c::SGWBCatalog) = size(c.fluxes, 1)

"""
    average_mode(catalog::SGWBCatalog) -> AbstractAverageMode

Derive the inclination-averaging convention from the catalog's own `inclination`
column: an all-zero column means the waveforms were generated face-on and the
`2/5` inclination average must still be applied analytically
([`AnalyticInclination`](@ref)); any non-zero entry means the Monte Carlo average
over catalog samples already performs it ([`CatalogInclination`](@ref)).

Catalogs with no `inclination` column fall back to [`AnalyticInclination`](@ref),
matching the convention of the legacy face-on generator. Every `gwmock-pop`
catalog does emit the column, so that fallback only applies to hand-built
fixtures and pre-`gwmock` files.

Pass `average_mode` explicitly to [`spectral_density`](@ref) or
`AstroSGWBInference.build_turing_model` to override the derived value.
"""
function average_mode(catalog::SGWBCatalog)
    haskey(catalog.samples, :inclination) || return AnalyticInclination()
    return all(iszero, catalog.samples.inclination) ? AnalyticInclination() :
           CatalogInclination()
end
