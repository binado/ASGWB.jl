using AdvancedHMC
using Bijectors
using Distributions: ProductNamedTupleDistribution
using FiniteDiff
using ForwardDiff
using LogDensityProblems
using LogDensityProblemsAD

"""
    ASGWBLogDensity(problem, prior)

A struct representing the log-density of the ASGWB importance sampling model,
conforming to the `LogDensityProblems.jl` interface. It handles the
transformation between unconstrained parameters (where the sampler operates)
and constrained physical parameters.
"""
struct ASGWBLogDensity{
    C <: ImportanceSamplingProblem, P <: ProductNamedTupleDistribution, B}
    problem::C
    prior::P
    transform::B
end

function ASGWBLogDensity(
        problem::ImportanceSamplingProblem,
        prior::ProductNamedTupleDistribution
)
    return ASGWBLogDensity(problem, prior, bijector(prior))
end

"""
    constrained_parameters(ld::ASGWBLogDensity, z) -> (theta_nt, logabsdet)

Transform unconstrained parameters `z` back to the physical parameter space
defined by the prior. Returns a named tuple of parameters and the log-absolute-determinant
of the Jacobian of the inverse transformation.
"""
function constrained_parameters(ld::ASGWBLogDensity, z::AbstractVector{<:Real})
    theta_nt, logabsdet = with_logabsdet_jacobian(inverse(ld.transform), z)
    return theta_nt, logabsdet
end

"""
    unconstrained_initial_point(ld::ASGWBLogDensity, theta0::NamedTuple) -> Vector{Float64}

Transform a set of physical hyperparameters `theta0` into the unconstrained
parameter space.
"""
function unconstrained_initial_point(ld::ASGWBLogDensity, theta0::NamedTuple)
    ordered_theta0 = (; (k => theta0[k] for k in hyperparameter_order(ld.prior))...)
    return collect(Bijectors.link(ld.prior, ordered_theta0))
end

LogDensityProblems.dimension(ld::ASGWBLogDensity) = length(keys(ld.prior.dists))
function LogDensityProblems.capabilities(::Type{<:ASGWBLogDensity})
    LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.logdensity(ld::ASGWBLogDensity, z::AbstractVector{<:Real})
    theta_nt, logabsdet = constrained_parameters(ld, z)
    return logposterior(theta_nt, ld.problem, ld.prior) + logabsdet
end

"""
    ad_logdensity(ld::ASGWBLogDensity)

Wrap an `ASGWBLogDensity` with ForwardDiff-based automatic differentiation
using `LogDensityProblemsAD.jl`.
"""
function ad_logdensity(ld::ASGWBLogDensity)
    return LogDensityProblemsAD.ADgradient(:ForwardDiff, ld)
end

"""
    finite_difference_logdensity_and_gradient(ld::ASGWBLogDensity, z) -> (logd, grad)

Compute the log-density and its gradient at `z` using finite differences.
Useful for verifying AD gradients.
"""
function finite_difference_logdensity_and_gradient(
        ld::ASGWBLogDensity,
        z::AbstractVector{<:Real}
)
    zf = collect(Float64, z)
    gradient = similar(zf)
    FiniteDiff.finite_difference_gradient!(
        gradient,
        x -> LogDensityProblems.logdensity(ld, x),
        zf
    )
    return LogDensityProblems.logdensity(ld, zf), gradient
end

"""
    sample_with_advancedhmc(problem, prior, theta0; kwargs...) -> (samples, stats, ld)

Sample from the ASGWB posterior using `AdvancedHMC.jl` directly (without Turing).
"""
function sample_with_advancedhmc(
        problem::ImportanceSamplingProblem,
        prior::ProductNamedTupleDistribution,
        theta0::NamedTuple;
        n_adapts::Int = 25,
        n_samples::Int = 25,
        target_acceptance::Float64 = 0.8
)
    ld = ASGWBLogDensity(problem, prior)
    z0 = unconstrained_initial_point(ld, theta0)
    ad_problem = ad_logdensity(ld)

    metric = DiagEuclideanMetric(length(z0))
    hamiltonian = Hamiltonian(metric, ad_problem)
    step_size = find_good_stepsize(hamiltonian, z0)
    integrator = Leapfrog(step_size)
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = StanHMCAdaptor(
        MassMatrixAdaptor(metric),
        StepSizeAdaptor(target_acceptance, integrator)
    )

    samples_unconstrained,
    stats = sample(hamiltonian, kernel, z0, n_samples, adaptor, n_adapts; progress = false)

    samples_constrained = map(samples_unconstrained) do z
        theta_nt, _ = constrained_parameters(ld, z)
        coerce_hyperparameters(theta_nt)
    end

    return samples_constrained, stats, ld
end
