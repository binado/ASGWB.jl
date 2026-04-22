# ORF following GWFast (arXiv:astro-ph/9305029), matching Python `asgwb.detector.overlap`.

const _R_EARTH_KM = 6371.0
const _C_LIGHT_KM_S = 299792.458
const _LOW_ALPHA_THRESHOLD = 2e-3

function _chord_distance_km(lat1, lon1, lat2, lon2)
    lat1r = deg2rad(lat1)
    lon1r = deg2rad(lon1)
    lat2r = deg2rad(lat2)
    lon2r = deg2rad(lon2)
    x1 = cos(lat1r) * cos(lon1r)
    y1 = cos(lat1r) * sin(lon1r)
    z1 = sin(lat1r)
    x2 = cos(lat2r) * cos(lon2r)
    y2 = cos(lat2r) * sin(lon2r)
    z2 = sin(lat2r)
    return _R_EARTH_KM * sqrt((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)
end

function _initial_course_deg(lat1, lat2, lon1, lon2)
    lat1r = deg2rad(lat1)
    lat2r = deg2rad(lat2)
    dlon = deg2rad(lon2 - lon1)
    x = cos(lat2r) * sin(dlon)
    y = cos(lat1r) * sin(lat2r) - sin(lat1r) * cos(lat2r) * cos(dlon)
    return rad2deg(atan(x, y)) % 360
end

function _final_course_deg(lat1, lat2, lon1, lon2)
    return (_initial_course_deg(lat2, lat1, lon2, lon1) + 180.0) % 360
end

function _opening_angle_rad(az1, az2)
    diff = ((az1 - az2 + 180.0) % 360.0) - 180.0
    return deg2rad(abs(diff))
end

function _azimuth_bisector_rad(az1, az2)
    a1 = deg2rad(az1)
    a2 = deg2rad(az2)
    return atan(sin(a1) + sin(a2), cos(a1) + cos(a2))
end

function _g1(alpha::AbstractVector{Float64})
    return map(_g1_scalar, alpha)
end

function _g1_scalar(alpha::Float64)
    if abs(alpha) < 1e-14
        return 3.0 / 56.0
    end
    return (5.0 / 16.0) * (
        (
        -9 * alpha * cos(alpha) - 6 * alpha^3 * cos(alpha) +
        9 * sin(alpha) +
        3 * alpha^2 * sin(alpha) +
        alpha^4 * sin(alpha)
    ) / alpha^5
    )
end

function _g2(alpha::AbstractVector{Float64})
    return map(_g2_scalar, alpha)
end

function _g2_scalar(alpha::Float64)
    if abs(alpha) < 1e-14
        return -1.0 / 168.0
    end
    return (5.0 / 16.0) * (
        (
        45 * alpha * cos(alpha) + 6 * alpha^3 * cos(alpha) - 45 * sin(alpha) +
        9 * alpha^2 * sin(alpha) +
        3 * alpha^4 * sin(alpha)
    ) / alpha^5
    )
end

function _g3(alpha::AbstractVector{Float64})
    return map(_g3_scalar, alpha)
end

function _g3_scalar(alpha::Float64)
    if abs(alpha) < 1e-14
        return 1.0 / 168.0
    end
    return (5.0 / 4.0) * (
        (
        15 * alpha * cos(alpha) - 4 * alpha^3 * cos(alpha) - 15 * sin(alpha) +
        9 * alpha^2 * sin(alpha) - alpha^4 * sin(alpha)
    ) / alpha^5
    )
end

function _get_orf(
        alpha::AbstractVector{Float64},
        beta::Float64,
        delta::Float64,
        big_delta::Float64,
        ang_btw_arms_1::Float64,
        ang_btw_arms_2::Float64
)
    sin1 = sin(ang_btw_arms_1)
    sin2 = sin(ang_btw_arms_2)
    g1 = _g1(alpha)
    g2 = _g2(alpha)
    g3 = _g3(alpha)
    cb = cos(0.5 * beta)
    sb = sin(0.5 * beta)
    theta_1 = (cb^4) .* g1
    theta_2 = (cb^4) .* g2 .+ g3 .- (sb^4) .* (g2 .+ g1)
    high = (cos(4 * delta) .* theta_1 .+ cos(4 * big_delta) .* theta_2) .* (sin1 * sin2)
    low = cos(4 * delta) * sin1 * sin2
    return ifelse.(alpha .> _LOW_ALPHA_THRESHOLD, high, low)
end

"""
    overlap_reduction_function(frequencies, detector_1, detector_2)

Stochastic isotropic ORF Γ(f) for one detector pair (same length as `frequencies`).
"""
function overlap_reduction_function(
        frequencies::AbstractVector{<:Real},
        detector_1::Detector,
        detector_2::Detector
)
    f = Float64.(collect(frequencies))
    lat1 = detector_1.latitude
    lon1 = detector_1.longitude
    lat2 = detector_2.latitude
    lon2 = detector_2.longitude
    d = _chord_distance_km(lat1, lon1, lat2, lon2)
    alpha = @. 2 * π * f * d / _C_LIGHT_KM_S
    xax_1 = _azimuth_bisector_rad(detector_1.xarm_azimuth, detector_1.yarm_azimuth)
    xax_2 = _azimuth_bisector_rad(detector_2.xarm_azimuth, detector_2.yarm_azimuth)
    ang_1 = deg2rad(_initial_course_deg(lat1, lat2, lon1, lon2) - 90.0)
    ang_2 = deg2rad(_final_course_deg(lat1, lat2, lon1, lon2) - 90.0)
    delta = 0.5 * ((xax_1 + ang_1) - (xax_2 + ang_2))
    big_delta = 0.5 * ((xax_1 + ang_1) + (xax_2 + ang_2))
    asin_arg = clamp(0.5 * d / _R_EARTH_KM, -1.0, 1.0)
    beta = 2 * asin(asin_arg)
    ang_btw_arms_1 = _opening_angle_rad(detector_1.xarm_azimuth, detector_1.yarm_azimuth)
    ang_btw_arms_2 = _opening_angle_rad(detector_2.xarm_azimuth, detector_2.yarm_azimuth)
    return _get_orf(alpha, beta, delta, big_delta, ang_btw_arms_1, ang_btw_arms_2)
end

"""
    pairwise_overlap_reduction_function(frequencies, detectors)

Returns `Γ[i,j,f]` with shape `(n_detector, n_detector, n_freq)` (symmetric in `i,j`).
"""
function pairwise_overlap_reduction_function(
        frequencies::AbstractVector{<:Real},
        detectors::AbstractVector{<:Detector}
)
    det_list = collect(detectors)
    n = length(det_list)
    n_freq = length(frequencies)
    out = zeros(Float64, n, n, n_freq)
    for i in 1:n
        for j in i:n
            d1 = det_list[i]
            d2 = det_list[j]
            orf = overlap_reduction_function(frequencies, d1, d2)
            out[i, j, :] .= orf
            out[j, i, :] .= orf
        end
    end
    return out
end
