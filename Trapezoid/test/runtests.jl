using Test
using ForwardDiff
using Trapezoid

@testset "trapz / cumtrapz" begin
    # Linear integrand: the trapezoidal rule is exact, so compare to the antiderivative.
    x = [0.0, 1.0, 2.0]
    y = @. 2.0 + 3.0 * x
    @test cumtrapz(x, y) ≈ [0.0, 2.0 + 1.5, 4.0 + 6.0]
    @test trapz(x, y) ≈ 2.0 * 2.0 + 1.5 * 2.0^2

    # The load-bearing identity: same accumulation order, so bit-for-bit equal.
    xr = collect(LinRange(1e-3, 20.0, 256))
    yr = @. exp(-xr) * (2 + sin(7xr))
    @test trapz(xr, yr) === last(cumtrapz(xr, yr))
    @test cumtrapz(xr, yr)[1] == 0.0

    @test_throws ArgumentError trapz([0.0, 1.0], [1.0])
    @test_throws ArgumentError cumtrapz([0.0, 1.0], [1.0])

    # Duals propagate and the eltype follows `y`.
    yd = ForwardDiff.Dual{Nothing}.(yr, 1.0)
    @test eltype(cumtrapz(xr, yd)) <: ForwardDiff.Dual
    @test trapz(xr, yd) isa ForwardDiff.Dual
end
