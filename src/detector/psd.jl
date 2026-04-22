"""
Tabulated power spectral density S(f) loaded from a two-column text file (Hz, PSD or ASD).
"""
struct PowerSpectralDensity
    frequency::Vector{Float64}
    psd::Vector{Float64}
end

function _read_psd_table(path::AbstractString)
    rows = Vector{Tuple{Float64, Float64}}()
    open(path, "r") do io
        for line in eachline(io)
            s = strip(line)
            isempty(s) && continue
            startswith(s, '#') && continue
            parts = split(s)
            length(parts) >= 2 || continue
            push!(rows, (parse(Float64, parts[1]), parse(Float64, parts[2])))
        end
    end
    isempty(rows) && throw(ArgumentError("no numeric rows in PSD file $(repr(path))"))
    f = [r[1] for r in rows]
    v = [r[2] for r in rows]
    return f, v
end

"""
    PowerSpectralDensity(path; curve_type=:psd)

`curve_type` is `:psd` (file column is S_n) or `:asd` (column is √(S_n); values are squared).
Out-of-range frequencies evaluate to `Inf` (matches the Python stack).
"""
function PowerSpectralDensity(path::AbstractString; curve_type::Symbol = :psd)
    curve_type in (:psd, :asd) || throw(ArgumentError("curve_type must be :psd or :asd"))
    f, v = _read_psd_table(path)
    psd = curve_type == :asd ? v .^ 2 : copy(v)
    return PowerSpectralDensity(f, psd)
end

function _linear_extrapolate_inf(
        xq::AbstractVector{<:Real},
        x::AbstractVector{Float64},
        y::AbstractVector{Float64}
)
    n = length(xq)
    out = Vector{Float64}(undef, n)
    x_min, x_max = first(x), last(x)
    @inbounds for i in 1:n
        q = Float64(xq[i])
        if q < x_min || q > x_max
            out[i] = Inf
            continue
        end
        k = searchsortedlast(x, q)
        if k <= 0
            out[i] = Inf
        elseif k >= length(x)
            out[i] = y[end]
        else
            x0, x1 = x[k], x[k + 1]
            y0, y1 = y[k], y[k + 1]
            t = (q - x0) / (x1 - x0)
            out[i] = y0 + t * (y1 - y0)
        end
    end
    return out
end

function (p::PowerSpectralDensity)(frequencies::AbstractVector{<:Real})
    return _linear_extrapolate_inf(frequencies, p.frequency, p.psd)
end

function (p::PowerSpectralDensity)(f::Real)
    v = p([Float64(f)])
    return v[1]
end
