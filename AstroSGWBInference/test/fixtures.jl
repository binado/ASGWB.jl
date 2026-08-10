using Distributions: Uniform

# S1: the model contract is a callable, so an ad-hoc model is three lines of arithmetic --
# no struct, no method definitions on foreign generics, no import.
const LOCAL_MODEL = function (Λ, samples)
    return (1.0e-7 * Λ.rate_scale, fill(Λ.weight_shift, length(samples.redshift)))
end
# The fixtures are already band-restricted: the DC row of the underlying parity
# catalog is sliced off before reaching the model, as a real caller would.
const LOCAL_FLUXES = Float64[1.0 1.5; 2.0 2.5]
const LOCAL_SAMPLES = (redshift = [0.1, 0.2],)
const LOCAL_FIDUCIALS = (rate_scale = 1.0, weight_shift = 0.0)
const LOCAL_THETA = (rate_scale = 1.1, weight_shift = 0.05)
const LOCAL_PRIOR = (
    rate_scale = Uniform(0.5, 1.5),
    weight_shift = Uniform(-0.2, 0.2)
)
const LOCAL_FREQUENCIES = [20.0, 40.0]
const LOCAL_EFFECTIVE_PSD = [1.0, 1.0]
const LOCAL_OBSERVATION_TIME = 1.0

function local_problem_context()
    return (;
        model = LOCAL_MODEL,
        fluxes = LOCAL_FLUXES,
        samples = LOCAL_SAMPLES,
        fiducials = LOCAL_FIDUCIALS,
        theta = LOCAL_THETA,
        prior = LOCAL_PRIOR,
        frequencies = LOCAL_FREQUENCIES,
        effective_psd = LOCAL_EFFECTIVE_PSD,
        observation_time = LOCAL_OBSERVATION_TIME
    )
end
