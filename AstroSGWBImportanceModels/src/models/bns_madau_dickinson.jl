"""
    AMPLITUDE_PARAMETERS

Hyperparameters of [`BNSMadauDickinsonImportanceModel`](@ref) that enter the predicted
spectrum as a **pure multiplicative factor**, and can therefore be integrated out of the
Gaussian likelihood by
`AstroSGWBInference.astrosgwb_amplitude_marginalized_turing_model` instead of sampled.

The property is exact for this adapter and independent of `w0`, `Ωm`, `Ξ₀`, and `Ξₙ`; it
is asserted directly against `forward_model` in this package's tests.
"""
const AMPLITUDE_PARAMETERS = (:H0, :R₀)

# The predicted spectrum factorizes into two independently-scaling pieces -- the total
# merger rate and the importance-weighted polarization-power contraction (the mean energy
# flux) -- so the full scaling is `f = g_R · g_F`.
#
# `R₀` enters only through `merger_rate_per_sec(∫, Λ.R₀)` (linear, and absent from
# `log_weights`), so `g_R = φ`, `g_F = 1`, `f = φ`.
#
# `H0` enters the rate through `dV_c/dz ∝ H0⁻³` and the weights through
# `-2 log d_L ∝ H0²` -- the normalized redshift density `log_p - log(norm)` is
# H0-invariant, so that is the *only* surviving H0 dependence in the weights. Hence
# `g_R = φ⁻³`, `g_F = φ²`, `f = φ⁻¹`.
#
# These are plain top-level callables, not closures over the fiducial: the consumer forms
# the ratio `f(φ)/f(φ_fid)` itself, and a top-level `const` function keeps them cheap to
# pass around and identical across constructions. Nothing here imports
# `AstroSGWBInference` -- like `merger_rate_and_log_weights_fn`, the scalings reach the
# sampler as callables passed in at the call site.

"""
    merger_rate_amplitude_H0(φ)

Merger-rate scaling ``g_R(H_0) = H_0^{-3}``, from ``dV_c/dz \\propto H_0^{-3}``.
"""
merger_rate_amplitude_H0(φ) = φ^-3

"""
    amplitude_H0(φ)

Total multiplicative scaling ``f(H_0) = g_R g_F = H_0^{-3} \\cdot H_0^{2} = H_0^{-1}``.
"""
amplitude_H0(φ) = inv(φ)

"""
    merger_rate_amplitude_R₀(φ)

Merger-rate scaling ``g_R(\\mathcal{R}_0) = \\mathcal{R}_0``.
"""
merger_rate_amplitude_R₀(φ) = φ

"""
    amplitude_R₀(φ)

Total multiplicative scaling ``f(\\mathcal{R}_0) = g_R g_F = \\mathcal{R}_0 \\cdot 1``.
"""
amplitude_R₀(φ) = φ

"""
    bns_amplitude_scalings(name::Symbol) -> (; amplitude_fn, merger_rate_fn)

Look up the amplitude scalings for the hyperparameter `name`.

`amplitude_fn` is the full ``f = g_R g_F`` the marginalization integrates against;
`merger_rate_fn` is ``g_R`` alone, which post-processing needs to turn the published
`template_merger_rate` back into the physical rate (see
`AstroSGWBInference.reconstruct_amplitude`). Only ratios to the fiducial are ever used, so
an overall normalization of either cancels.

Throws an `ArgumentError` for any name outside [`AMPLITUDE_PARAMETERS`](@ref) -- a
parameter that is not strictly multiplicative would be silently mis-marginalized.
"""
function bns_amplitude_scalings(name::Symbol)
    name === :H0 && return (; amplitude_fn = amplitude_H0,
        merger_rate_fn = merger_rate_amplitude_H0)
    name === :R₀ && return (; amplitude_fn = amplitude_R₀,
        merger_rate_fn = merger_rate_amplitude_R₀)
    throw(ArgumentError(
        "unsupported amplitude parameter $(repr(name)); the BNS Madau-Dickinson adapter " *
        "supports $(AMPLITUDE_PARAMETERS)",
    ))
end

"""
    BNSMadauDickinsonImportanceModel{C, P}

Prepared BNS importance model using a Madau–Dickinson source-frame merger rate,
background cosmology `C`, and GW propagation model `P`. Detector state (frequencies,
effective PSD, observation time) is intentionally kept out of this model and passed to
`AstroSGWBInference.astrosgwb_importance_turing_model` as flattened arrays.

The model is a **functor**: `model(Λ, samples) -> (rate, log_weights)` is the whole
contract `AstroSGWBInference.astrosgwb_importance_turing_model` consumes, so this package
adds no methods to foreign generics and does not depend on the inference package at all.

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
linearly, so a caller holds it fixed by conditioning (`model | (; R₀ = …)`) and samples
it by dropping the conditioning. `observation_time` is gone entirely: it cancelled
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
    source_model = MadauDickinsonSourceFrame(γ = Λ.γ, κ = Λ.κ, zpeak = Λ.zpeak)
    sfd = source_frame_distribution.(Ref(source_model), zg)
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
