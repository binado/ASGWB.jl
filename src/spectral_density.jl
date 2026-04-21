"""
    spectral_density(fluxes, merger_rate_per_sec; weights=nothing) -> Vector

Collapse per-sample flux contributions into a spectral density vector.

`fluxes` is a `(n_freq, n_samples)` matrix (column-major friendly). When `weights`
is `nothing`, samples are averaged uniformly: `mean_flux = sum(fluxes; dims=2) / n_samples`.
When `weights` is supplied, the contraction is `fluxes * weights / n_samples`
(no normalization of `weights`). The `0.4 = 2/5` prefactor captures the
average over the inclination angle.
"""
function spectral_density(
    fluxes::AbstractMatrix{<:Real},
    merger_rate_per_sec::Real;
    weights::Union{Nothing,AbstractVector{<:Real}}=nothing,
)
    n_samples = size(fluxes, 2)
    mean_flux = if weights === nothing
        vec(sum(fluxes; dims=2)) ./ n_samples
    else
        length(weights) == n_samples || throw(
            ArgumentError(
                "weights length ($(length(weights))) must match fluxes sample count ($(n_samples))",
            ),
        )
        (fluxes * weights) ./ n_samples
    end
    return 0.4 .* merger_rate_per_sec .* mean_flux
end
