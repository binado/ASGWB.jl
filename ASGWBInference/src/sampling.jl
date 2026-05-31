using AdvancedHMC
using Bijectors
using Distributions: ProductNamedTupleDistribution, logpdf
using FiniteDiff
using ForwardDiff
using LogDensityProblems
using LogDensityProblemsAD

"""
    ASGWBLogDensity(problem, C, ctx, prior; observed)

A struct representing the log-density of the ASGWB importance sampling model,
conforming to the `LogDensityProblems.jl` interface. It handles the
transformation between unconstrained parameters (where the sampler operates)
and constrained physical parameters.

The cosmology family `C` is carried as a type parameter (a compile-time constant for the
hot path); `ctx` holds the `Λ`-independent caches and the default observed data.
"""
struct ASGWBLogDensity{
    C, M <: PopulationModel, P <: ProductNamedTupleDistribution, B}
    problem::ImportanceSamplingProblem{M}
    cosmology_type::Type{C}
    ctx::ModelContext
    prior::P
    transform::B
    order::Tuple{Vararg{Symbol}}
    observed::Vector{Float64}
end

function validate_hyperprior(order::Tuple{Vararg{Symbol}}, prior::ProductNamedTupleDistribution)
    keys(prior.dists) == order || throw(
        ArgumentError("hyperprior must match order $(order), got $(keys(prior.dists))"),
    )
    return nothing
end

function logposterior(
        Λ::NamedTuple,
        problem::ImportanceSamplingProblem,
        ::Type{C},
        ctx::ModelContext,
        prior::ProductNamedTupleDistribution;
        observed::AbstractVector{<:Real} = ctx.fiducial_spectral_density
) where {C <: AbstractCosmology}
    return logpdf(prior, Λ) +
           loglikelihood(Λ, problem, C, ctx; observed = observed)
end

function ASGWBLogDensity(
        problem::ImportanceSamplingProblem,
        ::Type{C},
        ctx::ModelContext,
        prior::ProductNamedTupleDistribution;
        observed::AbstractVector{<:Real} = ctx.fiducial_spectral_density
) where {C <: AbstractCosmology}
    order = full_hyperparameters(C, problem.population_model)
    validate_hyperprior(order, prior)
    return ASGWBLogDensity(
        problem, C, ctx, prior, bijector(prior), order, collect(Float64, observed))
end

"""
    constrained_parameters(ld::ASGWBLogDensity, z) -> (Λ, logabsdet)

Transform unconstrained parameters `z` back to the physical parameter space
defined by the prior. Returns a named tuple of parameters and the log-absolute-determinant
of the Jacobian of the inverse transformation.
"""
function constrained_parameters(ld::ASGWBLogDensity, z::AbstractVector{<:Real})
    Λ, logabsdet = with_logabsdet_jacobian(inverse(ld.transform), z)
    return Λ, logabsdet
end

"""
    unconstrained_initial_point(ld::ASGWBLogDensity, theta0::NamedTuple) -> Vector{Float64}

Transform a set of physical hyperparameters `theta0` into the unconstrained
parameter space.
"""
function unconstrained_initial_point(ld::ASGWBLogDensity, theta0::NamedTuple)
    validate_hyperparameters(ld.order, theta0; context = "initial hyperparameters")
    ordered_theta0 = (; (k => theta0[k] for k in ld.order)...)
    return collect(Bijectors.link(ld.prior, ordered_theta0))
end

LogDensityProblems.dimension(ld::ASGWBLogDensity) = length(ld.order)
function LogDensityProblems.capabilities(::Type{<:ASGWBLogDensity})
    LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.logdensity(ld::ASGWBLogDensity, z::AbstractVector{<:Real})
    Λ, logabsdet = constrained_parameters(ld, z)
    return logposterior(
        Λ, ld.problem, ld.cosmology_type, ld.ctx, ld.prior; observed = ld.observed) +
           logabsdet
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
    sample_with_advancedhmc(problem, C, ctx, prior, theta0; kwargs...) -> (samples, stats, ld)

Sample from the ASGWB posterior using `AdvancedHMC.jl` directly (without Turing).
"""
function sample_with_advancedhmc(
        problem::ImportanceSamplingProblem,
        ::Type{C},
        ctx::ModelContext,
        prior::ProductNamedTupleDistribution,
        theta0::NamedTuple;
        n_adapts::Int = 25,
        n_samples::Int = 25,
        target_acceptance::Float64 = 0.8
) where {C <: AbstractCosmology}
    ld = ASGWBLogDensity(problem, C, ctx, prior)
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
        Λ, _ = constrained_parameters(ld, z)
        canonical_hyperparameters(ld.order, Λ; context = "sampled hyperparameters")
    end

    return samples_constrained, stats, ld
end
