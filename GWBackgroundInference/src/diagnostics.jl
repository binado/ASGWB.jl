function normalized_ess(weights::AbstractVector{<:Real})
    w = weights ./ sum(weights)
    return inv(sum(abs2, w)) / length(w)
end
