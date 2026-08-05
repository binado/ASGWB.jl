using DataInterpolations: LinearInterpolation

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
    cumtrapz(x, y) -> AbstractVector

Cumulative trapezoidal integral of `y` over nodes `x`, evaluated at each node:
`out[1] = 0` and `out[i+1] = out[i] + (x[i+1] - x[i]) * (y[i] + y[i+1]) / 2`. Exact for
the antiderivative of the linear interpolant through `(x, y)` at the nodes.

`y` may carry `ForwardDiff.Dual` values; the output element type follows `y`.

Shares its accumulation with [`trapz`](@ref), so `trapz(x, y) === last(cumtrapz(x, y))`
bit-for-bit — the two are one formula, not two.
"""
function cumtrapz(x::AbstractVector{<:Real}, y::AbstractVector)
    n = length(x)
    length(y) == n || throw(ArgumentError("x and y must have the same length"))
    n >= 1 || throw(ArgumentError("cumtrapz requires at least one grid point"))
    cumulative = similar(y)
    @inbounds cumulative[1] = zero(y[1])
    acc = @inbounds cumulative[1]
    @inbounds for i in 1:(n - 1)
        dx = x[i + 1] - x[i]
        acc = acc + dx * (y[i] + y[i + 1]) * 0.5
        cumulative[i + 1] = acc
    end
    return cumulative
end

"""
    trapz(x, y) -> Real

Trapezoidal integral of `y` over nodes `x`. Accumulates left-to-right in exactly the
order [`cumtrapz`](@ref) does, so `trapz(x, y) === last(cumtrapz(x, y))` bit-for-bit.

That identity is load-bearing: the importance-model normalizer comes from `trapz` while
`CumulativeIntegral1D`'s `normalizer` comes from `cumtrapz`, and a difference in
summation order between them would show up as a spurious per-sample offset in the
log-weights.
"""
function trapz(x::AbstractVector{<:Real}, y::AbstractVector)
    n = length(x)
    length(y) == n || throw(ArgumentError("x and y must have the same length"))
    n >= 1 || throw(ArgumentError("trapz requires at least one grid point"))
    acc = zero(@inbounds y[1])
    @inbounds for i in 1:(n - 1)
        dx = x[i + 1] - x[i]
        acc = acc + dx * (y[i] + y[i + 1]) * 0.5
    end
    return acc
end

# Struct-facing alias, so `CumulativeIntegral1D.cumulative` and the free `cumtrapz`
# cannot drift apart.
function _cumulative_at_nodes_trapezoid(x::AbstractVector{Float64}, y::AbstractVector)
    cumtrapz(x, y)
end

@inline function _linear_cell_integral(cumulative_at_left, y_lo, y_hi, dx, t)
    return cumulative_at_left + dx * (y_lo * t + 0.5 * (y_hi - y_lo) * t^2)
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

"""
    CumulativeIntegral1D(x, f)

Build a [`CumulativeIntegral1D`](@ref) by evaluating `f` at each node of `x`,
building a `LinearInterpolation`, and computing nodal cumulative integrals with
[`cumtrapz`](@ref) (O(n), identical to the linear interpolant's antiderivative on the
nodes). `x` must be strictly increasing with length ≥ 2.

Off-grid [`cdf`](@ref) queries use a direct analytic trapezoid lookup on the
cached nodal values.
"""
function CumulativeIntegral1D(x::AbstractVector{<:Real}, f)
    n = length(x)
    n >= 2 || throw(ArgumentError("CumulativeIntegral1D requires at least 2 grid points"))
    x_float = x isa AbstractVector{Float64} ? x : collect(Float64, x)
    y = map(f, x_float)
    return _cumulative_integral_from_values(x_float, y)
end

"""
    CumulativeIntegral1D(x, y::AbstractVector{<:Real})

Build a [`CumulativeIntegral1D`](@ref) directly from precomputed nodal values `y`
(rather than evaluating a function). `x` must be strictly increasing with length ≥ 2
and `length(y) == length(x)`. `y` may carry `ForwardDiff.Dual` values so the cumulative
integral differentiates through whatever produced `y`.
"""
function CumulativeIntegral1D(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    n = length(x)
    n >= 2 || throw(ArgumentError("CumulativeIntegral1D requires at least 2 grid points"))
    length(y) == n || throw(ArgumentError("x and y must have the same length"))
    x_float = x isa AbstractVector{Float64} ? x : collect(Float64, x)
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
    idx = searchsortedlast(c.x, x0)
    @inbounds begin
        dx = c.x[idx + 1] - c.x[idx]
        t = (x0 - c.x[idx]) / dx
        y_lo = c.y[idx]
        y_hi = c.y[idx + 1]
        return _linear_cell_integral(c.cumulative[idx], y_lo, y_hi, dx, t)
    end
end

"""
    normalizer(c::CumulativeIntegral1D) -> Real

Total integral of `f` over the grid, `last(c.cumulative)`.
"""
normalizer(c::CumulativeIntegral1D) = @inbounds c.cumulative[end]

"""
    GridInterpolator(points, grid; check_bounds = false)

Precomputed linear-interpolation plan for a fixed set of `points` on a fixed `grid`.
Calling it on any grid-valued vector `y` (`length(y) == length(grid)`) returns
`y` linearly interpolated at `points`:

```julia
interp = GridInterpolator(z_samples, z_grid)
d_l    = interp(luminosity_distance_grid)
p      = interp(dN_dz_grid)
```

One verb for one operation, applied to as many tabulated quantities as needed — the
per-point `searchsortedlast` is paid once at construction and reused across every
call and every likelihood evaluation sharing the grid.

Out-of-grid points are **clamped** (both the cell index and the within-cell fraction),
matching the Python stack; they do not extrapolate. Pass `check_bounds = true` to throw
instead. General-purpose callers get the clamping primitive; setup paths that want loud
failure do their own range check (see `prepare_bns_madau_dickinson_model`).

`y` may carry `ForwardDiff.Dual` values; the output element type is
`promote_type(eltype(y), Float64)` so the `Float64` fractions promote correctly and an
empty `points` set still yields a concretely-typed empty vector.
"""
struct GridInterpolator
    idx::Vector{Int}
    t::Vector{Float64}
    # Grid length, so the `@inbounds` loop below can be justified by an O(1) check
    # instead of trusting the caller to pass a vector of the right length.
    n_grid::Int
end

function GridInterpolator(
        points::AbstractVector{<:Real},
        grid::AbstractVector{<:Real};
        check_bounds::Bool = false
)
    n_grid = length(grid)
    n_grid >= 2 || throw(ArgumentError("grid must contain at least two points"))
    n = length(points)
    idx = Vector{Int}(undef, n)
    t = Vector{Float64}(undef, n)
    x_min = @inbounds grid[1]
    x_max = @inbounds grid[end]
    @inbounds for k in 1:n
        z = points[k]
        if check_bounds && !(x_min <= z <= x_max)
            throw(ArgumentError(
                "query point $(z) lies outside grid support [$x_min, $x_max]"))
        end
        i = clamp(searchsortedlast(grid, z), 1, n_grid - 1)
        dx = grid[i + 1] - grid[i]
        idx[k] = i
        # Clamping `t` as well as `i` is what makes an out-of-grid point clamp rather
        # than extrapolate linearly off the end of the grid.
        t[k] = clamp(Float64((z - grid[i]) / dx), 0.0, 1.0)
    end
    return GridInterpolator(idx, t, n_grid)
end

function (g::GridInterpolator)(y::AbstractVector)
    length(y) == g.n_grid || throw(DimensionMismatch(
        "GridInterpolator was built on a grid of $(g.n_grid) nodes but got $(length(y)) values"))
    out = similar(y, promote_type(eltype(y), Float64), length(g.idx))
    @inbounds for k in eachindex(g.idx)
        i, t = g.idx[k], g.t[k]
        out[k] = y[i] + t * (y[i + 1] - y[i])
    end
    return out
end
