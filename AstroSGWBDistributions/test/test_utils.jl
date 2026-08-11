using AstroSGWBDistributions
using Test

@testset "time conversions" begin
    @test year_to_second(1.0) ≈ 365.25 * 24 * 3600
    @test second_to_year(year_to_second(2.5)) ≈ 2.5
    @test year_to_second(1.0) == JULIAN_YEAR_SEC
end
