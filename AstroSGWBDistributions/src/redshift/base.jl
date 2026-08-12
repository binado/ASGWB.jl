using Distributions
using Random

export AbstractSourceFrame,
       source_frame_distribution,
       RedshiftInterpolatedDistribution,
       DEFAULT_Z_GRID

"""
    AbstractSourceFrame

Source-frame merger-rate density model. Subtypes implement
[`source_frame_distribution`](@ref)`(sf, z)`.

Used as a constructor argument to [`RedshiftInterpolatedDistribution`](@ref); the
tabulated detector-frame distribution does not store the source-frame model.
"""
abstract type AbstractSourceFrame end

Base.broadcastable(sf::AbstractSourceFrame) = Ref(sf)

"""
    source_frame_distribution(sf::AbstractSourceFrame, z) -> Real

Source-frame merger-rate density at redshift `z`. Subtypes of
[`AbstractSourceFrame`](@ref) must implement this method.
"""
function source_frame_distribution end

"""
    DEFAULT_Z_GRID

Default redshift integration grid: 256 uniformly-spaced points on [0, 20].
Shared across callers that do not pass an explicit grid.

The grid must start at `0` and be strictly increasing.
`distance_and_volume_grid` enforces these requirements because its cumulative
comoving-distance integral assumes `d_c(0) = 0`.
"""
const DEFAULT_Z_GRID = collect(LinRange(0.0, 20.0, 256))

"""
    RedshiftInterpolatedDistribution

Detector-frame redshift distribution: composes an [`Interpolated1DDistribution`](@ref)
whose tabulated density is the detector-frame merger-rate density. When the source-frame
model includes the local merger rate and unit conversions, [`normalizer`](@ref) is the
detector-frame merger rate in events/sec.
"""
struct RedshiftInterpolatedDistribution{D <: Interpolated1DDistribution} <:
       ContinuousUnivariateDistribution
    dist::D
end

"""
    RedshiftInterpolatedDistribution(sf, differential_comoving_volume, z_grid)

Tabulate the detector-frame redshift density
`4π · dV/dz · ψ(z) / (1 + z)` on `z_grid` from an [`AbstractSourceFrame`](@ref) and a
precomputed differential-comoving-volume array, then wrap it as an
[`Interpolated1DDistribution`](@ref).

Cosmology-independent: callers supply `differential_comoving_volume` themselves (e.g. from
[`distance_and_volume_grid`](@ref)).
"""
function RedshiftInterpolatedDistribution(
        sf::AbstractSourceFrame,
        differential_comoving_volume::AbstractVector{<:Real},
        z_grid::AbstractVector{<:Real}
)
    length(differential_comoving_volume) == length(z_grid) || throw(DimensionMismatch(
        "differential_comoving_volume and z_grid must have the same length"))
    z_grid_f = z_grid isa AbstractVector{Float64} ? z_grid : collect(Float64, z_grid)
    y = @. 4π * differential_comoving_volume *
           source_frame_distribution(sf, z_grid_f) / (1 + z_grid_f)
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
