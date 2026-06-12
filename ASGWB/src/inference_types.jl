"""
    ImportanceSamplingProblem{M <: PopulationModel}

Pure importance-sampling specification: the minimal raw inputs needed to define the
forward model, with no derived arrays, no cosmology, and no detector state.

Fields:
- `population_model::M` — the [`PopulationModel`](@ref) whose `single_event_prior` is the
  importance-sampling target / proposal density.
- `fluxes::Matrix{Float64}` — raw per-sample fluxes `|h₊|² + |h×|²` from the waveform
  catalog, *before* the fiducial `(D_L/D_gw)²` scaling, `(n_freq, n_samples)`.
- `samples::NamedTuple` — restructured per-event parameters (struct-of-arrays). Keys must
  match `keys(single_event_prior(...).dists)` so `batched_logpdf` lines up (e.g. `mass`,
  `redshift`, `χ₁`, `χ₂`, `Λ₁`, `Λ₂`).
- `fiducial_hyperparameters::NamedTuple` — canonical fiducial hyperparameters; the
  cosmology + propagation + population state at which the proposal caches are built.

The raw `fluxes` are used directly in the spectral-density contraction; the full distance
correction `(D_L,fid/D_L,θ)²·(1/Ξ_θ²)` lives in the importance weights, not in the fluxes.
All derived/`Λ`-independent caches (`dl_fid_sq`, proposal log-prob, redshift
interpolant, detector PSDs, fiducial spectral density) live in [`ModelContext`](@ref),
built by [`build_model_context`](@ref). The cosmology family `C` is passed as a call
argument, never stored here.
"""
struct ImportanceSamplingProblem{M <: PopulationModel}
    population_model::M
    fluxes::Matrix{Float64}
    samples::NamedTuple
    fiducial_hyperparameters::NamedTuple
end

redshift(s::NamedTuple) = s.redshift

redshift(problem::ImportanceSamplingProblem) = redshift(problem.samples)

"""
    ModelContext

Flat catalog-derived cache of all `Λ`-independent derived state for an [`ImportanceSamplingProblem`](@ref),
built once by [`build_model_context`](@ref) and reused by every likelihood/model call:

- proposal caches at the fiducial point: `proposal_prior` (the fiducial
  `single_event_prior` itself, used by [`logprobdiff`](@ref) for the egal component
  skip), `proposal_log_prob` (a `NamedTuple` of per-component log-density vectors),
  and `dl_fid_sq` (squared EM luminosity distance at the fiducial cosmology),
- redshift state: `redshift_grid` and the proposal `sample_interpolant`,
- detector/observation state grouped in [`ObservationContext`](@ref),
- `local_merger_rate` (events Gpc⁻³ yr⁻¹), and
- `fiducial_spectral_density`, the default observed data of this context.

A `ModelContext` is built for a specific cosmology family `C`; calling cached atomics with
a different `C` would silently mix mismatched caches. Coherence is guaranteed by
construction: `build_model_context` and the model that uses it close over a single literal
`C`.
"""
struct ModelContext{P <: ProductNamedTupleDistribution, L <: NamedTuple}
    proposal_prior::P
    proposal_log_prob::L
    dl_fid_sq::Vector{Float64}
    redshift_grid::Vector{Float64}
    sample_interpolant::SampleInterpolant
    observation::ObservationContext
    local_merger_rate::Float64
    fiducial_spectral_density::Vector{Float64}
end
