using Statistics

function normalized_ess(weights::AbstractVector{<:Real})
    w = weights ./ sum(weights)
    return inv(sum(abs2, w)) / length(w)
end

function max_normalized_weight(weights::AbstractVector{<:Real})
    return maximum(weights) / sum(weights)
end

log_ratio_variance(log_ratio::AbstractVector{<:Real}) = var(log_ratio; corrected = false)
