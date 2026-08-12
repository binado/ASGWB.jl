using Distributions
using Random

export RedshiftInterpolatedDistribution,
       detector_frame_merger_rate_density,
       DEFAULT_Z_GRID

"""
    DEFAULT_Z_GRID

Default redshift integration grid: 256 uniformly-spaced points on [0, 20].
Shared across [`redshift_prior`](@ref) calls that do not pass an explicit grid.

The grid must start at `0` and be strictly increasing. [`distance_and_volume_grid`](@ref)
enforces these requirements because its cumulative comoving-distance integral assumes
`d_c(0) = 0`.
"""
const DEFAULT_Z_GRID = collect(LinRange(0.0, 20.0, 256))

function detector_frame_merger_rate_density(
        z::Real,
        differential_comoving_volume::Real,
        source_frame_distribution::Real
)
    return 4π * differential_comoving_volume * source_frame_distribution / (1 + z)
end

"""
    RedshiftInterpolatedDistribution

Detector-frame redshift distribution: composes an [`Interpolated1DDistribution`](@ref)
whose tabulated density is the detector-frame merger-rate density. When the source-frame
callable includes the local merger rate and unit conversions, [`normalizer`](@ref) is the
detector-frame merger rate in events/sec.
"""
struct RedshiftInterpolatedDistribution{D <: Interpolated1DDistribution} <:
       ContinuousUnivariateDistribution
    dist::D
end

"""
    RedshiftInterpolatedDistribution(source_frame_fn, cosmology, z_grid)

Tabulate the detector-frame redshift density on `z_grid` from `source_frame_fn` and the
cosmology volume grid, then wrap it as an [`Interpolated1DDistribution`](@ref).
"""
function RedshiftInterpolatedDistribution(
        source_frame_fn,
        cosmo::AbstractCosmology,
        z_grid::AbstractVector{<:Real}
)
    z_grid_f = z_grid isa AbstractVector{Float64} ? z_grid : collect(Float64, z_grid)
    grid = distance_and_volume_grid(cosmo, z_grid_f)
    y = map(eachindex(z_grid_f)) do i
        @inbounds detector_frame_merger_rate_density(
            z_grid_f[i],
            grid.differential_comoving_volume[i],
            source_frame_fn(z_grid_f[i])
        )
    end
    return RedshiftInterpolatedDistribution(Interpolated1DDistribution(z_grid_f, y))
end

normalizer(d::RedshiftInterpolatedDistribution) = normalizer(d.dist)
Base.minimum(d::RedshiftInterpolatedDistribution) = minimum(d.dist)
Base.maximum(d::RedshiftInterpolatedDistribution) = maximum(d.dist)
Base.eltype(d::RedshiftInterpolatedDistribution) = eltype(d.dist)

function Distributions.insupport(d::RedshiftInterpolatedDistribution, value::Real)
    return insupport(d.dist, value)
end

function Distributions.logpdf(d::RedshiftInterpolatedDistribution, value::Real)
    return logpdf(d.dist, value)
end

function Random.rand(rng::AbstractRNG, d::RedshiftInterpolatedDistribution)
    return rand(rng, d.dist)
end
