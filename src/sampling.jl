using AdvancedHMC
using Bijectors
using Distributions: ProductNamedTupleDistribution
using FiniteDiff
using ForwardDiff
using LogDensityProblems
using LogDensityProblemsAD

struct ASGWBLogDensity{
    C<:ImportanceSamplingProblem,
    P<:ProductNamedTupleDistribution,
    B,
}
    problem::C
    prior::P
    transform::B
end

function ASGWBLogDensity(
    problem::ImportanceSamplingProblem,
    prior::ProductNamedTupleDistribution,
)
    return ASGWBLogDensity(problem, prior, bijector(prior))
end

function constrained_parameters(ld::ASGWBLogDensity, z::AbstractVector{<:Real})
    theta_nt, logabsdet = with_logabsdet_jacobian(inverse(ld.transform), z)
    return theta_nt, logabsdet
end

function unconstrained_initial_point(ld::ASGWBLogDensity, theta0::HyperParameters)
    return collect(Bijectors.link(ld.prior, theta0))
end

LogDensityProblems.dimension(ld::ASGWBLogDensity) = length(keys(ld.prior.dists))
LogDensityProblems.capabilities(::Type{<:ASGWBLogDensity}) =
    LogDensityProblems.LogDensityOrder{0}()

function LogDensityProblems.logdensity(
    ld::ASGWBLogDensity,
    z::AbstractVector{<:Real},
)
    theta_nt, logabsdet = constrained_parameters(ld, z)
    return logposterior(theta_nt, ld.problem, ld.prior) + logabsdet
end

function ad_logdensity(ld::ASGWBLogDensity)
    return LogDensityProblemsAD.ADgradient(:ForwardDiff, ld)
end

function finite_difference_logdensity_and_gradient(
    ld::ASGWBLogDensity,
    z::AbstractVector{<:Real},
)
    zf = collect(Float64, z)
    gradient = similar(zf)
    FiniteDiff.finite_difference_gradient!(
        gradient,
        x -> LogDensityProblems.logdensity(ld, x),
        zf,
    )
    return LogDensityProblems.logdensity(ld, zf), gradient
end

function sample_with_advancedhmc(
    problem::ImportanceSamplingProblem,
    prior::ProductNamedTupleDistribution,
    theta0::HyperParameters;
    n_adapts::Int=25,
    n_samples::Int=25,
    target_acceptance::Float64=0.8,
)
    ld = ASGWBLogDensity(problem, prior)
    z0 = unconstrained_initial_point(ld, theta0)
    ad_problem = ad_logdensity(ld)

    metric = DiagEuclideanMetric(length(z0))
    hamiltonian = Hamiltonian(metric, ad_problem)
    step_size = find_good_stepsize(hamiltonian, z0)
    integrator = Leapfrog(step_size)
    kernel = HMCKernel(
        Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()),
    )
    adaptor = StanHMCAdaptor(
        MassMatrixAdaptor(metric),
        StepSizeAdaptor(target_acceptance, integrator),
    )

    samples_unconstrained, stats = sample(
        hamiltonian,
        kernel,
        z0,
        n_samples,
        adaptor,
        n_adapts;
        progress=false,
    )

    samples_constrained = map(samples_unconstrained) do z
        theta_nt, _ = constrained_parameters(ld, z)
        HyperParameters(theta_nt)
    end

    return samples_constrained, stats, ld
end
