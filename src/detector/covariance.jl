function _inverse_covariance_from_orf_psd(
        orf::AbstractArray{Float64, 3},
        psds::AbstractMatrix{Float64}
)
    # orf[i,j,f], psds[f, i] matches Python stacking on axis=-1
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
    covariance_on_grid(frequencies, detectors)

Per-frequency isotropic SGWB variance `σ²(f)` (diagonal in Ω), matching Python
`asgwb.likelihood.core.covariance.covariance_on_grid`.
Requires at least two detectors.
"""
function covariance_on_grid(
        frequencies::AbstractVector{<:Real},
        detectors::AbstractVector{<:Detector}
)
    length(detectors) >= 2 ||
        throw(ArgumentError("covariance_on_grid requires at least two detectors"))
    f = Float64.(collect(frequencies))
    orf = pairwise_overlap_reduction_function(f, detectors)
    psd_vecs = [det.psd(f) for det in detectors]
    psds = Matrix(reduce(hcat, psd_vecs)) # n_freq × n_det
    inv_cov = _inverse_covariance_from_orf_psd(orf, psds)
    cov = similar(inv_cov)
    @inbounds for i in eachindex(cov)
        inv_cov[i] == 0.0 ? (cov[i] = Inf) : (cov[i] = 1.0 / inv_cov[i])
    end
    return cov
end
