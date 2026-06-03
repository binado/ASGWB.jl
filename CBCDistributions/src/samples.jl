"""
    stack_source_masses(mass_1_source, mass_2_source) -> Matrix{Float64}

Pack two same-length mass vectors into a `2 × n` matrix (row 1 = `mass_1_source`,
row 2 = `mass_2_source`), the layout expected under the `mass` field of a
proposal-sample `NamedTuple`.
"""
function stack_source_masses(
        mass_1_source::AbstractVector{<:Real},
        mass_2_source::AbstractVector{<:Real}
)::Matrix{Float64}
    n = length(mass_1_source)
    length(mass_2_source) == n ||
        throw(ArgumentError("mass_1_source and mass_2_source must have matching lengths"))
    return permutedims(
        hcat(collect(Float64, mass_1_source), collect(Float64, mass_2_source)),
    )
end
