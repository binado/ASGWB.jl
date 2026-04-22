using Interpolations: linear_interpolation

"""
    RadialInterpolant(x, f; companion=nothing)

Uniform-grid interpolant of a scalar function `f` sampled on the node vector `x`,
combined with a composite-Simpson cumulative antiderivative.

The struct doubles as a PDF bundle (via [`integrand`](@ref) + `.norm`) and as a
cumulative-distance interpolant (via [`integrate`](@ref)). A PDF interpolant can
carry a pointer to the distance interpolant it was derived from through
`companion`, so downstream consumers (e.g. `compute_importance_weights`) can pull
cached distance values without recomputing.

[`integrand`](@ref) evaluates linear interpolation via Interpolations.jl only on
`[x[1], x[end]]`; out-of-domain queries throw `BoundsError`.

# Fields
- `x`            : uniform grid nodes (`Float64`)
- `y`            : `f` evaluated at each node
- `cumulative`   : composite-Simpson antiderivative at each node, `cumulative[1] = 0`
- `norm`         : `cumulative[end]` (stored for zero-overhead access)
- `h`            : uniform step size
- `companion`    : optional linked `RadialInterpolant` (or `nothing`)
- `itp`          : cached `linear_interpolation(x, y)` object (throws outside the grid)
"""
struct RadialInterpolant{TX<:AbstractVector{<:Real},
                         TY<:AbstractVector,
                         TC<:AbstractVector,
                         TN<:Real,
                         TP,
                         TI}
    x::TX
    y::TY
    cumulative::TC
    norm::TN
    h::Float64
    companion::TP
    itp::TI
end

"""
    RadialInterpolant(x, f; companion=nothing)

Build a [`RadialInterpolant`](@ref) by evaluating `f` at each node of `x` and at
each interior midpoint `x[k] + h/2`, folding both into a composite-Simpson
cumulative antiderivative. Midpoint values are consumed in-place and not
retained. `x` must be a (uniformly spaced) vector of length ≥ 2.
"""
function RadialInterpolant(x::AbstractVector{<:Real}, f; companion=nothing)
    n = length(x)
    n >= 2 || throw(ArgumentError("RadialInterpolant requires at least 2 grid points"))
    x_float = x isa AbstractVector{Float64} ? x : collect(Float64, x)
    h = (x_float[end] - x_float[1]) / (n - 1)
    y = map(f, x_float)
    cumulative = similar(y)
    cumulative[1] = zero(eltype(y))
    @inbounds for k in 1:n-1
        y_mid_k = f(x_float[k] + h / 2)
        cumulative[k+1] = cumulative[k] + (h / 6) * (y[k] + 4 * y_mid_k + y[k+1])
    end
    itp = linear_interpolation(x_float, y)
    return RadialInterpolant(x_float, y, cumulative, cumulative[end], Float64(h), companion, itp)
end

"""
    integrand(r::RadialInterpolant, x0) -> Real

Linear interpolation of `r.y` at `x0` using Interpolations.jl. Only defined for
`x0 ∈ [r.x[1], r.x[end]]`; otherwise throws `BoundsError`.
"""
integrand(r::RadialInterpolant, x0::Real) = r.itp(x0)

"""
    integrate(r::RadialInterpolant, x0, f) -> Real

Evaluate the antiderivative of the interpolated function at `x0` using a local
Simpson correction on top of the precomputed cumulative table. `f` must be the
same integrand used to build `r` (or a consistent extension). Requires two fresh
`f` evaluations (at `x0` and `(x[k] + x0)/2`).

Values of `x0` outside `[x[1], x[end]]` are clamped to the boundary cumulative.
"""
function integrate(r::RadialInterpolant, x0::Real, f)
    n = length(r.x)
    x_lo = @inbounds r.x[1]
    if x0 <= x_lo
        return zero(eltype(r.cumulative))
    elseif x0 >= @inbounds(r.x[n])
        return r.cumulative[n]
    end
    k = floor(Int, (x0 - x_lo) / r.h) + 1
    k = clamp(k, 1, n - 1)
    @inbounds begin
        while k > 1 && r.x[k] > x0
            k -= 1
        end
        while k < n - 1 && r.x[k+1] <= x0
            k += 1
        end
        δ = x0 - r.x[k]
        mid = r.x[k] + δ / 2
        y_start = r.y[k]
    end
    y_mid = f(mid)
    y_end = f(x0)
    return @inbounds r.cumulative[k] + (δ / 6) * (y_start + 4 * y_mid + y_end)
end
