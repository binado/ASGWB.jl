module CBCDistributions

using DataInterpolations: LinearInterpolation
import Cosmology
import Cosmology: AbstractCosmology,
                  cumtrapz, distance_and_volume_grid, trapz

export PopulationModel, single_event_prior,
       canonical_hyperparameters, validate_hyperparameters
export MadauDickinsonSourceFrame, source_frame_distribution, redshift_prior, DEFAULT_Z_GRID
export DefaultBBHPrimaryMass, DefaultBBHMassPair, planck_taper
export JULIAN_YEAR_SEC, year_to_second, second_to_year

include("utils.jl")
include("mass/uniform.jl")
include("mass/broken_power_law_plus_two_peaks.jl")
include("spins/aligned.jl")
include("redshift.jl")
include("population_model.jl")

end # module
