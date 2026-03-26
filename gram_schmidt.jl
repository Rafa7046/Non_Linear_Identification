using LinearAlgebra

# --- 1. Candidate Generation ---
function build_narx_dictionary(u, y; nu=2, ny=2)
    N = length(y)
    max_lag = max(nu, ny)

    # Target vector (shifted by max_lag)
    Y_target = y[max_lag+1:end]

    # Base linear regressors: [y(k-1), y(k-2), ..., u(k-1), u(k-2), ...]
    P_linear = zeros(N - max_lag, nu + ny)

    for k in (max_lag+1):N
        idx = k - max_lag
        for i in 1:ny
            P_linear[idx, i] = y[k-i]
        end
        for i in 1:nu
            P_linear[idx, ny+i] = u[k-i]
        end
    end

    # Create Nonlinear dictionary (Linear + Squared + Cubic terms)
    # This acts as a proxy for the `sysid.candidate_matrix(dm, l=3)`
    P_nonlinear = hcat(P_linear, P_linear .^ 2, P_linear .^ 3)

    return Y_target, P_nonlinear
end

# --- 2. Gram-Schmidt ERR Algorithm ---
function gram_schmidt_err(Y::Vector{Float64}, P::Matrix{Float64}, n_theta::Int)
    N, M = size(P)
    selected = Int[]
    Y_Y = dot(Y, Y)
    W = zeros(N, n_theta)

    for k in 1:n_theta
        ERRi = zeros(M)

        for i in 1:M
            if i in selected
                continue
            end

            w_i = copy(P[:, i])

            # Orthogonalize against W
            for j in 1:(k-1)
                w_j = W[:, j]
                α = dot(w_j, w_i) / dot(w_j, w_j)
                w_i .-= α .* w_j
            end

            w_norm_sq = dot(w_i, w_i)
            if w_norm_sq > 1e-12
                g_i = dot(w_i, Y) / w_norm_sq
                ERRi[i] = (g_i^2 * w_norm_sq) / Y_Y
            end
        end

        best_idx = argmax(ERRi)
        push!(selected, best_idx)

        # Recompute winning vector for W
        w_best = copy(P[:, best_idx])
        for j in 1:(k-1)
            w_j = W[:, j]
            α = dot(w_j, w_best) / dot(w_j, w_j)
            w_best .-= α .* w_j
        end
        W[:, k] = w_best

        println("Step $k: Selected candidate index $best_idx, ERR: $(round(ERRi[best_idx], digits=6))")
    end

    return selected
end
