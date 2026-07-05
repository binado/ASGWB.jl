using Test
using Turing
using Turing.DynamicPPL: VarInfo, getsym
using FlexiChains
using AstroSGWB
using AstroSGWBInference: build_turing_model, condition_turing_model,
                          fiducial_spectral_density, logposterior,
                          merger_rate_and_log_weights

_varinfo_symbols(vi) = Set(getsym(vn) for vn in keys(vi))

@testset "Turing model smoke test with local adapter" begin
    problem = local_problem_context()
    model = build_turing_model(
        problem.model,
        problem.fluxes,
        problem.samples,
        problem.fiducials,
        problem.observation,
        problem.prior;
        track = false
    )
    observed = fiducial_spectral_density(
        problem.model, problem.fluxes, problem.samples, problem.fiducials)
    rate,
    log_weights = merger_rate_and_log_weights(
        problem.model, problem.fiducials, problem.samples)
    @test observed ≈ spectral_density(problem.fluxes, rate; weights = exp.(log_weights))
    @test Turing.logjoint(model, problem.theta) ≈ logposterior(
        problem.theta,
        problem.model,
        problem.fluxes,
        problem.samples,
        problem.observation,
        problem.prior,
        observed
    ) rtol = 1.0e-6

    tracked = build_turing_model(
        problem.model,
        problem.fluxes,
        problem.samples,
        problem.fiducials,
        problem.observation,
        problem.prior;
        track = true
    )
    returned_nt = Turing.returned(tracked, problem.theta)
    @test 0 < returned_nt.effective_sample_size <= 1
    @test isfinite(returned_nt.spectral_snr)
    @test returned_nt.spectral_snr^2 ≈ returned_nt.spectral_snr_squared

    chain = sample(
        model,
        Turing.NUTS(3, 0.8),
        3;
        progress = false,
        chain_type = FlexiChains.VNChain,
        initial_params = InitFromPrior()
    )
    @test chain isa FlexiChains.VNChain
    @test size(chain, 1) == 3
    @test sort(collect(Symbol.(FlexiChains.parameters(chain)))) ==
          sort(collect(keys(problem.theta)))
    @test all(isfinite, vec(Array(chain[:logjoint])))
end

@testset "flat submodel and conditioning boundary" begin
    problem = local_problem_context()
    model = build_turing_model(
        problem.model,
        problem.fluxes,
        problem.samples,
        problem.fiducials,
        problem.observation,
        problem.prior
    )
    present = _varinfo_symbols(VarInfo(model))
    @test present == Set((:rate_scale, :weight_shift))

    @test condition_turing_model(
        model, problem.theta, problem.prior, nothing) === model
    conditioned = condition_turing_model(
        model, problem.theta, problem.prior, (:rate_scale,))
    @test _varinfo_symbols(VarInfo(conditioned)) == Set((:rate_scale,))

    @test_throws ArgumentError condition_turing_model(
        model, problem.theta, problem.prior, ())
    @test_throws ArgumentError condition_turing_model(
        model, problem.theta, problem.prior, (:unknown,))
    @test_throws ArgumentError condition_turing_model(
        model, problem.theta, problem.prior, (:rate_scale, :rate_scale))
end
