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
