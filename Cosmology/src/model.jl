"""Abstract supertype for flat FLRW cosmology models."""
abstract type AbstractCosmology end

"""Flat ΛCDM cosmology (w=-1, radiation-free)."""
struct LambdaCDM{TH0 <: Real, TΩm <: Real} <: AbstractCosmology
    H0::TH0
    Ωm::TΩm
end

"""Flat wCDM cosmology with constant dark-energy equation of state w0."""
struct W0CDM{TH0 <: Real, TΩm <: Real, Tw0 <: Real} <: AbstractCosmology
    H0::TH0
    Ωm::TΩm
    w0::Tw0
end

"""Flat w0waCDM (CPL) cosmology: w(z) = w0 + wa·z/(1+z)."""
struct W0WaCDM{TH0 <: Real, TΩm <: Real, Tw0 <: Real, Twa <: Real} <: AbstractCosmology
    H0::TH0
    Ωm::TΩm
    w0::Tw0
    wa::Twa
end

H0(c::AbstractCosmology) = c.H0
Ωm(c::AbstractCosmology) = c.Ωm

Base.broadcastable(c::AbstractCosmology) = Ref(c)

"""Supported configurable cosmology subtypes (registration order)."""
const SUPPORTED_COSMOLOGIES = (
    LambdaCDM,
    W0CDM,
    W0WaCDM
)

"""
    hyperparameters(::Type{C}) -> Tuple{Vararg{Symbol}}

Hyperparameter symbols for cosmology subtype `C`, in struct field order.
"""
hyperparameters(::Type{C}) where {C <: AbstractCosmology} = fieldnames(C)

const cosmology_parameters = hyperparameters

"""
    cosmology(::Type{C}, h::NamedTuple) -> C

Build cosmology subtype `C` from hyperparameter state `h` (fields must match [`hyperparameters`](@ref)(`C`)).
"""
function cosmology(::Type{C}, h::NamedTuple) where {C <: AbstractCosmology}
    fn = fieldnames(C)
    return C(ntuple(i -> h[fn[i]], Val(length(fn)))...)
end

"""
    cosmology(h::NamedTuple) -> AbstractCosmology

Infer cosmology subtype from keys in `h` (`:wa` → [`W0WaCDM`](@ref), `:w0` → [`W0CDM`](@ref), else [`LambdaCDM`](@ref)).
"""
function cosmology(h::NamedTuple)
    :wa in keys(h) && return cosmology(W0WaCDM, h)
    :w0 in keys(h) && return cosmology(W0CDM, h)
    return cosmology(LambdaCDM, h)
end

function (::Type{C})(h::NamedTuple) where {C <: AbstractCosmology}
    return cosmology(C, h)
end

cosmology_config_name(::Type{LambdaCDM}) = "LambdaCDM"
cosmology_config_name(::Type{W0CDM}) = "W0CDM"
cosmology_config_name(::Type{W0WaCDM}) = "W0WaCDM"

const _COSMOLOGY_BY_CONFIG_NAME = Dict(
    cosmology_config_name(C) => C for C in SUPPORTED_COSMOLOGIES
)

"""
    cosmology_type(name::AbstractString) -> Type{<:AbstractCosmology}

Resolve a config/TOML cosmology name to a concrete subtype.
"""
function cosmology_type(name::AbstractString)
    C = get(_COSMOLOGY_BY_CONFIG_NAME, String(name), nothing)
    C === nothing && throw(
        ArgumentError(
        "unknown cosmology \"$(name)\"; valid choices: $(sort(collect(keys(_COSMOLOGY_BY_CONFIG_NAME))))",
    ),
    )
    return C
end

# ---------------------------------------------------------------------------
# GW propagation: an axis orthogonal to the FLRW background. `d_L^GW(z) =
# Ξ(z) · d_L^EM(z)`, and `Ξ(z)` never touches `Ωm`/`E(z)`/any distance integral,
# so propagation is threaded as its own type token `P` alongside the cosmology
# token `C` rather than wrapping a cosmology.
# ---------------------------------------------------------------------------

"""Abstract supertype for gravitational-wave propagation models."""
abstract type AbstractPropagation end

"""General relativity propagation: `Ξ(z) ≡ 1` (GW and EM distances coincide)."""
struct GR <: AbstractPropagation end

"""
Modified gravitational-wave propagation with `Ξ(z) = Ξ₀ + (1 - Ξ₀)/(1 + z)^Ξₙ`.
Independent of the FLRW background; combined with a cosmology only when forming
the GW luminosity distance.
"""
struct ModifiedPropagation{T <: Real} <: AbstractPropagation
    Ξ₀::T
    Ξₙ::T
end

# One type parameter for two fields means the default constructor rejects mixed eltypes.
# That is the *normal* case under ForwardDiff: a run that samples `Ξ₀` while holding `Ξₙ`
# fixed hands this constructor a `Dual` and a `Float64`. Promote instead of erroring.
ModifiedPropagation(Ξ₀::Real, Ξₙ::Real) = ModifiedPropagation(promote(Ξ₀, Ξₙ)...)

Base.broadcastable(p::AbstractPropagation) = Ref(p)

"""Supported configurable propagation subtypes (registration order)."""
const SUPPORTED_PROPAGATIONS = (GR, ModifiedPropagation)

"""
    propagation_hyperparameters(::Type{P}) -> Tuple{Vararg{Symbol}}

Hyperparameter symbols owned by propagation subtype `P`.
"""
propagation_hyperparameters(::Type{GR}) = ()
propagation_hyperparameters(::Type{<:ModifiedPropagation}) = (:Ξ₀, :Ξₙ)

"""
    propagation(::Type{P}, h::NamedTuple) -> AbstractPropagation

Build propagation subtype `P` from hyperparameter state `h`.
"""
propagation(::Type{GR}, h::NamedTuple) = GR()
propagation(::Type{<:ModifiedPropagation}, h::NamedTuple) = ModifiedPropagation(h.Ξ₀, h.Ξₙ)

propagation_config_name(::Type{GR}) = "GR"
propagation_config_name(::Type{<:ModifiedPropagation}) = "ModifiedPropagation"

const _PROPAGATION_BY_CONFIG_NAME = Dict(
    propagation_config_name(P) => P for P in SUPPORTED_PROPAGATIONS
)

"""
    propagation_type(name::AbstractString) -> Type{<:AbstractPropagation}

Resolve a config/TOML propagation name to a concrete subtype.
"""
function propagation_type(name::AbstractString)
    P = get(_PROPAGATION_BY_CONFIG_NAME, String(name), nothing)
    P === nothing && throw(
        ArgumentError(
        "unknown propagation \"$(name)\"; valid choices: $(sort(collect(keys(_PROPAGATION_BY_CONFIG_NAME))))",
    ),
    )
    return P
end

"""
    dark_energy_eos(c::AbstractCosmology, z) -> Real

Dark energy equation of state w(z).
"""
dark_energy_eos(::LambdaCDM, z) = -one(z)
dark_energy_eos(c::W0CDM, z) = c.w0
dark_energy_eos(c::W0WaCDM, z) = c.w0 + c.wa * z / (1 + z)

"""
    de_density_ratio(c::AbstractCosmology, z) -> Real

Ratio ρ_DE(z)/ρ_DE(0): closed-form integral of `dark_energy_eos` through the Friedmann equation.
"""
de_density_ratio(::LambdaCDM, z) = one(z)
de_density_ratio(c::W0CDM, z) = (1 + z)^(3 * (1 + c.w0))
function de_density_ratio(c::W0WaCDM, z)
    (1 + z)^(3 * (1 + c.w0 + c.wa)) * exp(-3 * c.wa * z / (1 + z))
end

"""
    E(z, c::AbstractCosmology) -> Real

Hubble parameter ratio E(z) = H(z)/H₀ for flat FLRW cosmology.
"""
function E(z::Real, c::AbstractCosmology)
    return sqrt(Ωm(c) * (1 + z)^3 + (1 - Ωm(c)) * de_density_ratio(c, z))
end
