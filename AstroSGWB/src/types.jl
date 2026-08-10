"""
    ObservationContext

Detector-side SGWB observation layout: frequency grid, per-bin effective strain PSD
amplitude from the detector network (ORFs and tabulated PSDs; square matches network
variance), Gaussian bin scales for the likelihood, and observation time metadata
(`observation_time`, in years).

All arrays are exactly the bins the likelihood scores: band selection is the caller's
job, done by slicing `frequencies` and the rows of the flux matrix before calling
[`build_observation_context`](@ref).

The observed spectral density is intentionally *not* part of this object; callers pass it
explicitly to inference likelihood routines or let `AstroSGWBInference.build_turing_model`
synthesize it from catalog fluxes, samples, and fiducials when `observed` is omitted.
"""
struct ObservationContext
    frequencies::Vector{Float64}
    effective_psd::Vector{Float64}
    sgwb_scale::Vector{Float64}
    observation_time::Float64
end
