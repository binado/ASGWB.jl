"""
    BNSMadauDickinsonImportanceModel{C, P}

Prepared BNS importance model using a Madau–Dickinson source-frame merger rate,
background cosmology `C`, and GW propagation model `P`. Detector state is intentionally
kept in a separate `AstroSGWB.ObservationContext`.

The model is a **functor**: `model(Λ, samples) -> (rate, log_weights)` is the whole
contract `AstroSGWBInference.build_turing_model` consumes, so this package adds no methods
to foreign generics and does not depend on the inference package at all.

`log_Ξ_fid` is `log Ξ(z_i)` at the **fiducial** propagation, captured at prepare time
because the hot path only ever sees the live `Λ`. It enters the log-weights as
`+2 log Ξ_fid`, which is the term that makes the weights consistent with a flux matrix
re-referenced to the fiducial GW distance by [`apply_gw_distance_correction!`](@ref). The
two must be applied together.
"""
struct BNSMadauDickinsonImportanceModel{
    C <: AbstractCosmology, P <: AbstractPropagation}
    z_grid::Vector{Float64}
    interp::GridInterpolator
    proposal_log_pdf::Vector{Float64}
    log_Ξ_fid::Vector{Float64}
    local_merger_rate::Float64
    observation_time::Float64
end

const _NON_GR_FIDUCIAL_NOTICE = "BNS model: non-GR fiducial propagation; log-weights " *
                                "carry the +2 log Ξ_fid term — the flux matrix must " *
                                "have been passed through apply_gw_distance_correction! " *
                                "at the same fiducials"

"""
    bns_samples_from_catalog(catalog_samples, C, fiducials) -> NamedTuple

Keep the catalog columns used by the BNS importance-weight loop. A stored
`luminosity_distance` column is copied verbatim; otherwise EM luminosity distances are
synthesized at the fiducial cosmology `C`.
"""
function bns_samples_from_catalog(
        catalog_samples::NamedTuple,
        ::Type{C},
        fiducials::NamedTuple
) where {C <: AbstractCosmology}
    z = copy(catalog_samples.redshift)
    d_l = haskey(catalog_samples, :luminosity_distance) ?
          copy(catalog_samples.luminosity_distance) :
          luminosity_distance.(z, Ref(cosmology(C, fiducials)))
    return (redshift = z, luminosity_distance = d_l)
end

"""
    prepare_bns_madau_dickinson_model(samples, fiducials, C, P;
        local_merger_rate, observation_time, z_grid=DEFAULT_Z_GRID)

Precompute the Float64 proposal caches for the canonical BNS Madau–Dickinson importance
adapter. `local_merger_rate` is the local merger rate in events per year; `observation_time`
is the observation duration in years (Julian year). Returns the prepared model directly.
Construct detector state separately with `AstroSGWB.build_observation_context`.

The returned model's log-weights are referenced to the **fiducial GW** luminosity
distance, so the flux matrix passed alongside must have been through
[`apply_gw_distance_correction!`](@ref) at the same `fiducials`. Under a `GR` (or
`Ξ₀ = 1`) fiducial both are no-ops; otherwise a mismatch is a silent `Ξ_fid²` bias, and
this function emits an `@info` reminder.
"""
function prepare_bns_madau_dickinson_model(
        samples::NamedTuple,
        fiducials::NamedTuple,
        ::Type{C},
        ::Type{P};
        local_merger_rate::Real,
        observation_time::Real,
        z_grid::AbstractVector{<:Real} = DEFAULT_Z_GRID
) where {C <: AbstractCosmology, P <: AbstractPropagation}
    z = samples.redshift
    zg = collect(Float64, z_grid)

    # Decision B: `GridInterpolator` itself clamps (it is a general-purpose primitive with
    # Python-matching numerics), but a proposal sample outside the integration grid is a
    # setup error, so report it loudly here. `all` on an empty collection is `true`, so
    # preparing against an empty sample set still works.
    all(zg[1] .<= z .<= zg[end]) || throw(ArgumentError(
        "proposal redshifts must lie inside the integration grid " *
        "[$(zg[1]), $(zg[end])]; got extrema $(extrema(z))"))
    interp = GridInterpolator(z, zg)

    proposal_log_pdf = _bns_grid_terms(C, fiducials, zg, interp).log_p::Vector{Float64}

    # `Float64[...]` is load-bearing: a `Vector{Dual}` field here would poison the
    # ForwardDiff fast path in `AstroSGWB.spectral_density`, which dispatches on
    # `fluxes::AbstractMatrix{<:Real}`.
    prop_fid = propagation(P, fiducials)
    log_Ξ_fid = Float64[log(gw_em_distance_ratio(zi, prop_fid)) for zi in z]

    # The call-site correction and this field are computed independently, so a call site
    # that forgets `apply_gw_distance_correction!` under a non-GR fiducial is wrong by
    # Ξ_fid² with no error. Never fires on a Ξ₀ = 1 corpus.
    if any(!iszero, log_Ξ_fid)
        @info _NON_GR_FIDUCIAL_NOTICE Ξ_fid=extrema(exp, log_Ξ_fid)
    end

    return BNSMadauDickinsonImportanceModel{C, P}(
        zg,
        interp,
        proposal_log_pdf,
        log_Ξ_fid,
        Float64(local_merger_rate),
        Float64(observation_time)
    )
end

"""
    _bns_grid_terms(C, Λ, zg, interp) -> (; log_p, d_l, norm)

Single source of truth for the detector-frame redshift log-density at the proposal
samples, the interpolated EM luminosity distances, and the redshift normalizer.

`prepare_bns_madau_dickinson_model` calls it with `Float64` fiducials and the model's own
call operator calls it with the live (possibly `ForwardDiff.Dual`) `Λ`. Sharing one code
path is what makes `log_p_target - proposal_log_pdf` **exactly** `0.0` at
`Λ == fiducials`; writing the formula twice would let accumulation order diverge by an
ulp, and every posterior would then carry a spurious per-sample offset.
"""
function _bns_grid_terms(
        ::Type{C},
        Λ::NamedTuple,
        zg::AbstractVector{<:Real},
        interp::GridInterpolator
) where {C <: AbstractCosmology}
    g = distance_and_volume_grid(cosmology(C, Λ), zg)
    sfd = source_frame_distribution.(Ref(MadauDickinsonSourceFrame()), zg, Ref(Λ))
    dN_dz = detector_frame_merger_rate_density.(zg, g.differential_comoving_volume, sfd)
    norm = trapz(zg, dN_dz)
    # Hoisted out of the broadcast: `@. interp(x)` would apply the functor elementwise.
    p = interp(dN_dz)
    # No underflow floor, matching astrogwb's `logpdf = log(pdf) - log(integral)`. The
    # density is strictly positive for every z > 0 under a Madau–Dickinson rate, and
    # `prepare_bns_madau_dickinson_model` rejects samples outside the grid, so the only
    # way to reach `log(0)` is a sample at exactly z = 0 — where the volume element
    # vanishes and `-Inf` is the honest answer. astrogwb lands on the same value there
    # via `jnp.interp(..., left=0.0)`.
    log_p = @. log(p) - log(norm)
    return (; log_p, d_l = interp(g.luminosity_distance), norm)
end

"""
    (model::BNSMadauDickinsonImportanceModel)(Λ, samples) -> (rate, log_weights)

The model contract: detector-frame merger rate in events per second, and one log
importance weight per catalog sample, at the live hyperparameters `Λ`.
"""
function (model::BNSMadauDickinsonImportanceModel{C, P})(
        Λ::NamedTuple,
        samples
) where {C <: AbstractCosmology, P <: AbstractPropagation}
    length(samples.redshift) == length(model.proposal_log_pdf) || throw(DimensionMismatch(
        "model was prepared for $(length(model.proposal_log_pdf)) samples but got " *
        "$(length(samples.redshift))"))

    t = _bns_grid_terms(C, Λ, model.z_grid, model.interp)
    Ξ_θ = gw_em_distance_ratio.(samples.redshift, Ref(propagation(P, Λ)))
    log_weights = @. t.log_p - model.proposal_log_pdf +
                     2 * (log(samples.luminosity_distance) - log(t.d_l) - log(Ξ_θ) +
                      model.log_Ξ_fid)

    rate = merger_rate_per_sec(t.norm, model.local_merger_rate, model.observation_time)
    return (rate, log_weights)
end
