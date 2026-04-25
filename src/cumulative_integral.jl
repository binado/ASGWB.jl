using DataInterpolations: LinearInterpolation, integral

"""
    CumulativeIntegral1D(x, f)

Linear interpolant of a scalar function `f` on strictly increasing nodes `x`,
plus a cumulative antiderivative at each node: prefix sum of trapezoids between
neighbors (exact for the `LinearInterpolation` antiderivative at grid points).

Query entry points:
- [`interpolate`](@ref) evaluates `f` via linear interpolation on `[x[1], x[end]]`
  (out-of-domain queries throw `BoundsError`).
- [`cdf`](@ref) evaluates the antiderivative at an arbitrary `x0`, clamping at the
  grid boundaries. The integral is exact under the linear interpolant (analytic
  trapezoidal rule), so no user-supplied `f` is needed at query time.
- [`normalizer`](@ref) returns the full integral `∫ f dx` over the grid
  (`last(cumulative)`).

# Fields
- `x`          : strictly increasing grid nodes (`Float64`)
- `y`          : `f` evaluated at each node
- `cumulative` : cumulative antiderivative at each node, `cumulative[1] = 0`
- `itp`        : cached `LinearInterpolation(y, x)` object
"""
struct CumulativeIntegral1D{
    TX <: AbstractVector{Float64},
    TY <: AbstractVector,
    TC <: AbstractVector,
    TI
}
    x::TX
    y::TY
    cumulative::TC
    itp::TI
end

"""
    CumulativeIntegral1D(x, f)

Build a [`CumulativeIntegral1D`](@ref) by evaluating `f` at each node of `x`,
building a `LinearInterpolation`, and computing nodal cumulative integrals as
the prefix sum of trapezoids ``(x_{i+1}-x_i)(y_i+y_{i+1})/2`` (O(n), identical
to the linear interpolant’s antiderivative on the nodes). `x` must be strictly
increasing with length ≥ 2.

Off-grid [`cdf`](@ref) queries still use `DataInterpolations.integral` on the
cached interpolant.
"""
function _cumulative_at_nodes_trapezoid(x::AbstractVector{Float64}, y::AbstractVector)
    n = length(x)
    length(y) == n || throw(ArgumentError("x and y must have the same length"))
    cumulative = similar(y)
    @inbounds cumulative[1] = zero(y[1])
    acc = cumulative[1]
    @inbounds for i in 1:(n - 1)
        dx = x[i + 1] - x[i]
        acc = acc + dx * (y[i] + y[i + 1]) * 0.5
        cumulative[i + 1] = acc
    end
    return cumulative
end

function _cumulative_integral_from_values(
        x::AbstractVector{<:Real},
        y::AbstractVector
)
    n = length(x)
    n >= 2 || throw(ArgumentError("CumulativeIntegral1D requires at least 2 grid points"))
    length(y) == n || throw(ArgumentError("x and y must have the same length"))
    x_float = x isa AbstractVector{Float64} ? x : collect(Float64, x)
    itp = LinearInterpolation(y, x_float)
    cumulative = _cumulative_at_nodes_trapezoid(x_float, y)
    return CumulativeIntegral1D(x_float, y, cumulative, itp)
end

function CumulativeIntegral1D(x::AbstractVector{<:Real}, f)
    n = length(x)
    n >= 2 || throw(ArgumentError("CumulativeIntegral1D requires at least 2 grid points"))
    x_float = x isa AbstractVector{Float64} ? x : collect(Float64, x)
    y = map(f, x_float)
    return _cumulative_integral_from_values(x_float, y)
end

"""
    interpolate(c::CumulativeIntegral1D, x0) -> Real

Linear interpolation of `c.y` at `x0`. Only defined for
`x0 ∈ [c.x[1], c.x[end]]`; otherwise throws `BoundsError`.
"""
interpolate(c::CumulativeIntegral1D, x0::Real) = c.itp(x0)

"""
    cdf(c::CumulativeIntegral1D, x0) -> Real

Evaluate the antiderivative of the linear interpolant from `x[1]` to `x0` using
the analytic (piecewise-trapezoidal) integral. Clamps to `0` for `x0 <= x[1]`
and to `last(cumulative)` for `x0 >= x[end]`.
"""
function cdf(c::CumulativeIntegral1D, x0::Real)
    x_lo = @inbounds c.x[1]
    if x0 <= x_lo
        return zero(eltype(c.cumulative))
    end
    x_hi = @inbounds c.x[end]
    if x0 >= x_hi
        return @inbounds c.cumulative[end]
    end
    return integral(c.itp, x_lo, x0)
end

"""
    normalizer(c::CumulativeIntegral1D) -> Real

Total integral of `f` over the grid, `last(c.cumulative)`.
"""
normalizer(c::CumulativeIntegral1D) = @inbounds c.cumulative[end]
