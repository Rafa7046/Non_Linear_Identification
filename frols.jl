# frols.jl
using LinearAlgebra
using Statistics

"""
    frols(y, candidates; M0, threshold)
    ... (docstring) ...
"""
function frols(y::Vector{T}, candidates::Matrix{T}; M0=nothing, threshold=1e-6) where T<:AbstractFloat
    N, M = size(candidates)
    M0 = isnothing(M0) ? M : min(M0, M)

    selected_indices = Int[]
    remaining_indices = collect(1:M)
    Q = zeros(T, N, M0)
    σy = var(y) * (N - 1) / N

    for s in 1:M0
        best_err = -Inf
        best_idx_in_rem = -1
        best_q = zeros(T, N)

        for (i, idx) in enumerate(remaining_indices)
            p_m = candidates[:, idx]
            q_m = copy(p_m)
            # Orthogonalization Step
            for r in 1:(s-1)
                qr = Q[:, r]
                α = dot(qr, p_m) / dot(qr, qr)
                q_m .-= α .* qr
            end

            den = dot(q_m, q_m)
            if den < 1e-12
                continue
            end

            g_m = dot(y, q_m) / den
            err = (g_m^2 * den) / (N * σy)

            if err > best_err
                best_err = err
                best_idx_in_rem = i
                best_q = q_m
            end
        end

        if best_idx_in_rem == -1 || best_err < threshold
            break
        end

        push!(selected_indices, remaining_indices[best_idx_in_rem])
        Q[:, s] .= best_q
        deleteat!(remaining_indices, best_idx_in_rem)
    end

    A = candidates[:, selected_indices]
    theta = A \ y
    return selected_indices, theta
end
