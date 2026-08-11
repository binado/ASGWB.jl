const JULIAN_YEAR_SEC = 365.25 * 24 * 3600.0

year_to_second(yr::Real) = Float64(yr) * JULIAN_YEAR_SEC
second_to_year(sec::Real) = Float64(sec) / JULIAN_YEAR_SEC

@inline function _planck_unit_exponent(t::Real)
    return inv(t) - inv(one(t) - t)
end

@inline function _planck_unit_taper(t::Real)
    T = typeof(t)
    t <= 0 && return zero(T)
    t >= 1 && return one(T)
    a = _planck_unit_exponent(t)
    if a > 0
        ea = exp(-a)
        return ea / (one(ea) + ea)
    end
    return inv(one(a) + exp(a))
end

@inline function _log_planck_unit_taper(t::Real)
    T = typeof(t)
    t <= 0 && return T(-Inf)
    t >= 1 && return zero(T)
    a = _planck_unit_exponent(t)
    if a > 0
        return -a - log1p(exp(-a))
    end
    return -log1p(exp(a))
end

@inline function _log_planck_taper(m::Real, low::Real, δ::Real)
    δ >= 0 || throw(ArgumentError("δ must be non-negative"))
    T = promote_type(typeof(m), typeof(low), typeof(δ))
    δ == 0 && return m < low ? T(-Inf) : zero(T)
    m <= low && return T(-Inf)
    m >= low + δ && return zero(T)
    return _log_planck_unit_taper((m - low) / δ)
end

"""
    planck_taper(m, low, δ)

Planck taper used by the DEFAULT BBH mass model. It is zero at or below `low`, rises as
`1 / (1 + exp(1/t - 1/(1 - t)))` with `t = (m - low)/δ` over `(low, low + δ)`, and is one
at or above `low + δ`. `δ == 0` is treated as a hard step to one at `low`.
"""
planck_taper(m::Real, low::Real, δ::Real) = exp(_log_planck_taper(m, low, δ))
