using CBCDistributions: AbstractCosmology

"""
Madau-Dickinson population with modified gravitational-wave propagation.

Type parameter `C <: AbstractCosmology` selects the cosmology model.
`MadauDickinsonModifiedPropagation()` defaults to `LambdaCDM`.
"""
struct MadauDickinsonModifiedPropagation{C <: AbstractCosmology} <: AbstractASGWBModel end

MadauDickinsonModifiedPropagation() = MadauDickinsonModifiedPropagation{LambdaCDM}()

"""
    model_parameters(::Type{<:MadauDickinsonModifiedPropagation}) -> Tuple{Vararg{Symbol}}

Hyperparameter symbols owned by the Madau–Dickinson modified-propagation forward model
(excluding cosmology parameters).
"""
model_parameters(::Type{<:MadauDickinsonModifiedPropagation}) = (:Ξ₀, :Ξₙ, :γ, :κ, :zpeak)

"""
    cosmology_type(model::MadauDickinsonModifiedPropagation{C}) -> Type{C}

Return the cosmology subtype baked into the forward model's type parameter.
"""
cosmology_type(::MadauDickinsonModifiedPropagation{C}) where {C <: AbstractCosmology} = C

function external_model_parameter_names(::MadauDickinsonModifiedPropagation)
    return (
        Ξ₀ = "Xi_0",
        Ξₙ = "Xi_n",
        γ = "gamma",
        κ = "kappa",
        zpeak = "z_peak"
    )
end

function model_section_dict(model::MadauDickinsonModifiedPropagation{C}) where {C <:
                                                                                AbstractCosmology}
    return Dict{String, Any}(
        "name" => MADAU_DICKINSON_MODIFIED_PROPAGATION_CONFIG_NAME,
        "cosmology" => cosmology_config_name(C)
    )
end

redshift_prior_family(::MadauDickinsonModifiedPropagation) = MadauDickinson

"""
    gravitational_wave_distance(m::MadauDickinsonModifiedPropagation, z, d_l, Λ)

Modified GW luminosity distance: destructures `Λ.Ξ₀`/`Λ.Ξₙ` and delegates to the
scalar CBCDistributions hook.
"""
function gravitational_wave_distance(
        ::MadauDickinsonModifiedPropagation,
        z::Real,
        d_l::Real,
        Λ::NamedTuple
)
    return gravitational_wave_distance(z, d_l, Λ.Ξ₀, Λ.Ξₙ)
end
