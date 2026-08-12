module Trapezoid

export trapz, cumtrapz

"""
    cumtrapz(x, y) -> AbstractVector

Cumulative trapezoidal integral of `y` over nodes `x`, evaluated at each node.
`out[1] = 0` and `out[i+1] = out[i] + (x[i+1] - x[i]) * (y[i] + y[i+1]) / 2`.
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

Trapezoidal integral of `y` over nodes `x`, using the same accumulation order as
[`cumtrapz`](@ref).
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

end # module
