"""
    BNSMadauDickinsonImportanceModel{C, P}

Prepared BNS importance model using a Madau–Dickinson source-frame merger rate,
background cosmology `C`, and GW propagation model `P`. Detector state (frequencies,
effective PSD, observation time) is intentionally kept out of this model and passed to
`AstroSGWBInference.build_turing_model` as flattened arrays.

The model is a **functor**: `model(Λ, samples) -> (rate, log_weights)` is the whole
contract `AstroSGWBInference.build_turing_model` consumes, so this package adds no methods
to foreign generics and does not depend on the inference package at all.

`log_Ξ_fid` is `log Ξ(z_i)` at the **fiducial** propagation, captured at prepare time
because the hot path only ever sees the live `Λ`. It enters the log-weights as
`+2 log Ξ_fid`, which is the term that makes the weights consistent with a polarization-power matrix
re-referenced to the fiducial GW distance by [`apply_gw_distance_correction!`](@ref). The
two must be applied together.
"""
struct BNSMadauDickinsonImportanceModel{
    C <: AbstractCosmology, P <: AbstractPropagation}
    z_grid::Vector{Float64}
    z_samples::Vector{Float64}
    proposal_log_pdf::Vector{Float64}
    log_Ξ_fid::Vector{Float64}
end

const _NON_GR_FIDUCIAL_NOTICE = "BNS model: non-GR fiducial propagation; log-weights " *
                                "carry the +2 log Ξ_fid term — the polarization-power matrix must " *
                                "have been passed through apply_gw_distance_correction! " *
                                "at the same fiducials"

"""
    prepare_bns_madau_dickinson_model(samples, fiducials, C, P; z_grid=DEFAULT_Z_GRID)

Precompute the Float64 proposal caches for the canonical BNS Madau–Dickinson importance
adapter. Returns the prepared model directly. Compute the detector-side effective PSD
separately with `AstroSGWB.effective_psd`.

The local merger rate is a live hyperparameter, read as `Λ.R₀` (in Gpc⁻³ yr⁻¹) on every
call, not a frozen field -- it is a real astrophysical unknown that scales the rate
linearly, so a caller can sample it by adding `R₀` to the prior or hold it fixed by
putting it in `constants`. `observation_time` is gone entirely: it cancelled
algebraically, and detector state never belongs in the importance model.

The returned model's log-weights are referenced to the **fiducial GW** luminosity
distance, so the polarization-power matrix passed alongside must have been through
[`apply_gw_distance_correction!`](@ref) at the same `fiducials`. Under a `GR` (or
`Ξ₀ = 1`) fiducial both are no-ops; otherwise a mismatch is a silent `Ξ_fid²` bias, and
this function emits an `@info` reminder.
"""
function prepare_bns_madau_dickinson_model(
        samples::NamedTuple,
        fiducials::NamedTuple,
        ::Type{C},
        ::Type{P};
        z_grid::AbstractVector{<:Real} = DEFAULT_Z_GRID
) where {C <: AbstractCosmology, P <: AbstractPropagation}
    z = samples.redshift
    zg = collect(Float64, z_grid)

    # DataInterpolations throws outside the grid. Report this as a model-setup error
    # before preparing proposal values. `all` on an empty collection is `true`, so
    # preparing against an empty sample set still works.
    all(zg[1] .<= z .<= zg[end]) || throw(ArgumentError(
        "proposal redshifts must lie inside the integration grid " *
        "[$(zg[1]), $(zg[end])]; got extrema $(extrema(z))"))
    z_samples = collect(Float64, z)

    proposal_log_pdf = _bns_grid_terms(C, fiducials, zg, z_samples).log_p::Vector{Float64}

    # `Float64[...]` is load-bearing: a `Vector{Dual}` field here would poison the
    # ForwardDiff fast path in `AstroSGWB.spectral_density`, which dispatches on
    # `polarization_power::AbstractMatrix{<:Real}`.
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
        z_samples,
        proposal_log_pdf,
        log_Ξ_fid
    )
end

"""
    _bns_grid_terms(C, Λ, zg, z_samples) -> (; log_p, d_l, norm)

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
        z_samples::AbstractVector{<:Real}
) where {C <: AbstractCosmology}
    g = distance_and_volume_grid(cosmology(C, Λ), zg)
    sfd = source_frame_distribution.(Ref(MadauDickinsonSourceFrame()), zg, Ref(Λ))
    dN_dz = detector_frame_merger_rate_density.(zg, g.differential_comoving_volume, sfd)
    norm = trapz(zg, dN_dz)
    p = _linear_interpolate(dN_dz, zg, z_samples)
    # No underflow floor, matching astrogwb's `logpdf = log(pdf) - log(integral)`. The
    # density is strictly positive for every z > 0 under a Madau–Dickinson rate, and
    # `prepare_bns_madau_dickinson_model` rejects samples outside the grid, so the only
    # way to reach `log(0)` is a sample at exactly z = 0 — where the volume element
    # vanishes and `-Inf` is the honest answer. astrogwb lands on the same value there
    # via `jnp.interp(..., left=0.0)`.
    log_p = @. log(p) - log(norm)
    d_l = _linear_interpolate(g.luminosity_distance, zg, z_samples)
    return (; log_p, d_l, norm)
end

function _linear_interpolate(
        values::AbstractVector,
        grid::AbstractVector{<:Real},
        points::AbstractVector{<:Real}
)
    isempty(points) && return similar(values, 0)
    return LinearInterpolation(values, grid)(points)
end

"""
    (model::BNSMadauDickinsonImportanceModel)(Λ, samples) -> (rate, log_weights)

The model contract: detector-frame merger rate in events per second, and one log
importance weight per catalog sample, at the live hyperparameters `Λ`.

`Λ` must carry the cosmology parameters of `C`, the propagation parameters of `P`, the
Madau–Dickinson shape `(:γ, :κ, :zpeak)`, and `:R₀`, the local merger rate in
Gpc⁻³ yr⁻¹. A missing key is a `KeyError` here, on the first evaluation.
"""
function (model::BNSMadauDickinsonImportanceModel{C, P})(
        Λ::NamedTuple,
        samples
) where {C <: AbstractCosmology, P <: AbstractPropagation}
    length(samples.redshift) == length(model.proposal_log_pdf) || throw(DimensionMismatch(
        "model was prepared for $(length(model.proposal_log_pdf)) samples but got " *
        "$(length(samples.redshift))"))

    t = _bns_grid_terms(C, Λ, model.z_grid, model.z_samples)
    Ξ_θ = gw_em_distance_ratio.(samples.redshift, Ref(propagation(P, Λ)))
    log_weights = @. t.log_p - model.proposal_log_pdf +
                     2 * (log(samples.luminosity_distance) - log(t.d_l) - log(Ξ_θ) +
                      model.log_Ξ_fid)

    rate = merger_rate_per_sec(t.norm, Λ.R₀)
    return (rate, log_weights)
end
