module GWDistributions

using DataInterpolations: LinearInterpolation
using Trapezoid: trapz, cumtrapz

export MadauDickinsonSourceFrame, source_frame_distribution, DEFAULT_Z_GRID
export AbstractSourceFrame, Interpolated1DDistribution, normalizer,
       RedshiftInterpolatedDistribution
export DefaultBBHPrimaryMass, DefaultBBHMassPair, planck_taper
export JULIAN_YEAR_SEC, year_to_second, second_to_year

include("utils.jl")
include("base/truncated_power_law.jl")
include("base/broken_power_law.jl")
include("base/interpolated_1d_distribution.jl")
include("mass/uniform.jl")
include("mass/broken_power_law_plus_two_peaks.jl")
include("spins/aligned.jl")
include("redshift/base.jl")
include("redshift/madau_dickinson.jl")

end # module
