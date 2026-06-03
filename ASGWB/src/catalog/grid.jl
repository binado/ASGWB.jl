"""
    FrequencyGrid

Frequency-axis specification mirroring the Python `FrequencyGrid` dataclass in
`scripts/generate_waveforms.py`. `frequencies` and `in_band_mask` are derived
properties; the five scalars are the canonical on-disk representation.
"""
struct FrequencyGrid
    duration::Float64
    sampling_frequency::Float64
    reference_frequency::Float64
    minimum_frequency::Float64
    maximum_frequency::Float64

    function FrequencyGrid(
            duration::Real,
            sampling_frequency::Real,
            reference_frequency::Real,
            minimum_frequency::Real,
            maximum_frequency::Union{Nothing, Real} = nothing
    )
        duration = Float64(duration)
        sampling_frequency = Float64(sampling_frequency)
        reference_frequency = Float64(reference_frequency)
        minimum_frequency = Float64(minimum_frequency)
        maximum_frequency = isnothing(maximum_frequency) ?
                            sampling_frequency / 2 : Float64(maximum_frequency)
        _validate_frequency_grid(
            duration,
            sampling_frequency,
            minimum_frequency,
            maximum_frequency
        )
        return new(
            duration,
            sampling_frequency,
            reference_frequency,
            minimum_frequency,
            maximum_frequency
        )
    end
end

function _validate_frequency_grid(
        duration::Float64,
        sampling_frequency::Float64,
        minimum_frequency::Float64,
        maximum_frequency::Float64
)
    duration > 0 || throw(ArgumentError("duration must be positive"))
    sampling_frequency > 0 ||
        throw(ArgumentError("sampling_frequency must be positive"))
    minimum_frequency >= 0 ||
        throw(ArgumentError("minimum_frequency must be nonnegative"))
    minimum_frequency < maximum_frequency ||
        throw(ArgumentError("minimum_frequency must be less than maximum_frequency"))
    maximum_frequency <= sampling_frequency / 2 ||
        throw(ArgumentError("maximum_frequency must not exceed Nyquist frequency"))
    return nothing
end

"""
    frequencies(g::FrequencyGrid) -> Vector{Float64}

Full frequency axis: `linspace(0, f_nyquist, nsamples÷2+1)`.
"""
function frequencies(g::FrequencyGrid)
    nsamples = round(Int, g.duration * g.sampling_frequency)
    nfreq = nsamples ÷ 2 + 1
    return collect(range(0.0, g.sampling_frequency / 2.0; length = nfreq))
end

"""
    in_band_mask(g::FrequencyGrid) -> BitVector

`true` for frequency bins satisfying `f_min ≤ f ≤ f_max`.
"""
function in_band_mask(g::FrequencyGrid)
    f = frequencies(g)
    return BitVector(@. g.minimum_frequency <= f <= g.maximum_frequency)
end
