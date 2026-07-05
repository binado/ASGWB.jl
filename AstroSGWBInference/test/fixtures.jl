using Distributions: Uniform, product_distribution
import AstroSGWBInference: hyperparameters, merger_rate_and_log_weights

struct LocalImportanceModel end

hyperparameters(::LocalImportanceModel) = (:rate_scale, :weight_shift)

function merger_rate_and_log_weights(::LocalImportanceModel, Λ::NamedTuple, samples)
    rate = 1.0e-7 * Λ.rate_scale
    return rate, fill(Λ.weight_shift, length(samples.redshift))
end

const LOCAL_MODEL = LocalImportanceModel()
const LOCAL_FLUXES = Float64[0.0 0.0; 1.0 1.5; 2.0 2.5]
const LOCAL_SAMPLES = (redshift = [0.1, 0.2],)
const LOCAL_FIDUCIALS = (rate_scale = 1.0, weight_shift = 0.0)
const LOCAL_THETA = (rate_scale = 1.1, weight_shift = 0.05)
const LOCAL_PRIOR = product_distribution((
    rate_scale = Uniform(0.5, 1.5),
    weight_shift = Uniform(-0.2, 0.2)
))
const LOCAL_OBSERVATION = ObservationContext(
    [0.0, 20.0, 40.0],
    [Inf, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    BitVector([false, true, true]),
    1.0
)

function local_problem_context()
    return (;
        model = LOCAL_MODEL,
        fluxes = LOCAL_FLUXES,
        samples = LOCAL_SAMPLES,
        fiducials = LOCAL_FIDUCIALS,
        theta = LOCAL_THETA,
        prior = LOCAL_PRIOR,
        observation = LOCAL_OBSERVATION
    )
end
