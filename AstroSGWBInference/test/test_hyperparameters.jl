using Test
using Distributions: Uniform, product_distribution
using AstroSGWBInference: build_turing_model, hyperparameters
import AstroSGWBInference: hyperparameters, merger_rate_and_log_weights

struct InvalidNamesModel{Names}
    names::Names
end

hyperparameters(model::InvalidNamesModel) = model.names
function merger_rate_and_log_weights(::InvalidNamesModel, Λ, samples)
    return (1.0, zeros(length(samples.redshift)))
end

@testset "caller-owned model hyperparameter contract" begin
    problem = local_problem_context()
    expected = Set(hyperparameters(problem.model))

    reversed_prior = product_distribution((;
        (name => problem.prior.dists[name]
    for name in reverse(keys(problem.prior.dists)))...))
    reversed_fiducials = (;
        (name => problem.fiducials[name] for name in reverse(keys(problem.fiducials)))...)

    @test Set(keys(reversed_prior.dists)) == expected
    @test keys(reversed_prior.dists) != hyperparameters(problem.model)
    @test build_turing_model(
        problem.model,
        problem.fluxes,
        problem.samples,
        reversed_fiducials,
        problem.observation,
        reversed_prior
    ) !== nothing

    missing_prior = product_distribution((rate_scale = problem.prior.dists.rate_scale,))
    extra_prior = product_distribution(merge(problem.prior.dists, (extra = Uniform(0, 1),)))
    missing_fiducials = (rate_scale = problem.fiducials.rate_scale,)
    extra_fiducials = merge(problem.fiducials, (extra = 1.0,))

    @test_throws ArgumentError build_turing_model(
        problem.model, problem.fluxes, problem.samples, problem.fiducials,
        problem.observation, missing_prior)
    @test_throws ArgumentError build_turing_model(
        problem.model, problem.fluxes, problem.samples, problem.fiducials,
        problem.observation, extra_prior)
    @test_throws ArgumentError build_turing_model(
        problem.model, problem.fluxes, problem.samples, missing_fiducials,
        problem.observation, problem.prior)
    @test_throws ArgumentError build_turing_model(
        problem.model, problem.fluxes, problem.samples, extra_fiducials,
        problem.observation, problem.prior)

    for invalid in (
        InvalidNamesModel((:x, :x)),
        InvalidNamesModel((:x, "y")),
        InvalidNamesModel([:x])
    )
        err = try
            build_turing_model(
                invalid, problem.fluxes, problem.samples, (x = 1.0,),
                problem.observation, product_distribution((x = Uniform(0, 2),)))
            nothing
        catch exception
            exception
        end
        @test err isa ArgumentError
        @test occursin("hyperparameters(model)", sprint(showerror, err))
    end
end
