"""
    SGWBCatalog{S<:NamedTuple}

Waveform catalog reduced for SGWB inference: the shared frequency axis, per-sample
per-frequency polarization power `|h₊|² + |h×|²`, and the per-sample source parameters
that produced it.

As loaded, `fluxes` is referenced to the **electromagnetic** luminosity distance, i.e.
before the fiducial `(D_L/D_gw)²` scaling. Callers re-reference it to the fiducial GW
distance with [`apply_gw_distance_correction!`](@ref) before preparing an importance
model; after an in-place call the field is no longer EM-referenced, and because the
correction is not idempotent a second call on the same object squares the factor. Use
the out-of-place [`apply_gw_distance_correction`](@ref) where a cell or block may re-run.

`samples` is a NamedTuple whose keys are the source-parameter column names (e.g.
`:mass_1_source`, `:redshift`, `:inclination`, ...). `fluxes` has shape
`(nfreq, nsamples)` -- column-major friendly for the importance-sampling hot loop,
and the orientation HDF5.jl reads the on-disk `(nsamples, nfreq)` C-order
polarization datasets into without a transpose.

`frequencies` comes from the file rather than being derived from
`(duration, sampling_frequency)` scalars. Band selection is the caller's job:
slice `frequencies` and the rows of `fluxes` before building an
[`ObservationContext`](@ref) or calling an inference model. Built by
[`load_catalog`](@ref) from a `waveform_catalog` v1 file.
"""
struct SGWBCatalog{S <: NamedTuple}
    frequencies::Vector{Float64}
    fluxes::Matrix{Float64}
    samples::S
    approximant::String
end

nsamples(c::SGWBCatalog) = size(c.fluxes, 2)
nfreq(c::SGWBCatalog) = size(c.fluxes, 1)

"""
    apply_gw_distance_correction!(catalog::SGWBCatalog, prop) -> catalog

Re-reference `catalog.fluxes` in place to the fiducial GW luminosity distance, pairing
the flux matrix with the catalog's own `redshift` column. This is the form to prefer at
call sites: it removes the one way the correction can go mechanically wrong, namely
pairing the flux matrix with a redshift vector that has been subsetted or reordered.

Not idempotent — see [`Cosmology.apply_gw_distance_correction!`](@ref).
"""
function apply_gw_distance_correction!(c::SGWBCatalog, prop::AbstractPropagation)
    apply_gw_distance_correction!(c.fluxes, c.samples.redshift, prop)
    return c
end

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
