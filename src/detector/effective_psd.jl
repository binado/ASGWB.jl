function _orf_psd_pairwise_weight(
        orf::AbstractArray{Float64, 3},
        psds::AbstractMatrix{Float64}
)
    # orf[i,j,f], psds[f, i] matches Python stacking on axis=-1
    # Returns g(f) such that network variance V(f)=1/g(f) and effective_psd(f)=sqrt(V(f)).
    n_freq = size(orf, 3)
    n_det = size(orf, 1)
    size(psds, 1) == n_freq || throw(DimensionMismatch("psds row count"))
    size(psds, 2) == n_det || throw(DimensionMismatch("psds column count"))
    out = Vector{Float64}(undef, n_freq)
    @inbounds for f in 1:n_freq
        acc = 0.0
        for i in 1:n_det
            si = 1.0 / psds[f, i]
            for j in (i + 1):n_det
                γ = orf[i, j, f]
                sj = 1.0 / psds[f, j]
                acc += si * γ^2 * sj
            end
        end
        out[f] = 0.16 * acc
    end
    return out
end

"""
    effective_psd(frequencies, detectors) -> Vector{Float64}

Per-frequency **effective strain PSD** (amplitude units, like ``\\sqrt{S_h}``) for an isotropic
background: the square root of the reciprocal of the pairwise ORF–PSD aggregate. The square
`effective_psd .^ 2` matches the per-frequency network variance returned historically by Python
`asgwb.likelihood.core.covariance.covariance_on_grid` / the former Julia `covariance_on_grid`.

Requires at least two detectors.
"""
function effective_psd(
        frequencies::AbstractVector{<:Real},
        detectors::AbstractVector{<:Detector}
)
    length(detectors) >= 2 ||
        throw(ArgumentError("effective_psd requires at least two detectors"))
    f = Float64.(collect(frequencies))
    orf = pairwise_overlap_reduction_function(f, detectors)
    psd_vecs = [det.psd(f) for det in detectors]
    psds = Matrix(reduce(hcat, psd_vecs)) # n_freq × n_det
    denom = _orf_psd_pairwise_weight(orf, psds)
    eff = similar(denom)
    @inbounds for i in eachindex(eff)
        if denom[i] == 0.0
            eff[i] = Inf
        else
            eff[i] = sqrt(1.0 / denom[i])
        end
    end
    return eff
end
