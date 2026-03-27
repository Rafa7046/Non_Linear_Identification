function fit(algo::GSERR, u, y, Y_target, P)
    N, M = size(P)
    selected = Int[]
    Y_Y = dot(Y_target, Y_target)
    W = zeros(N, algo.max_terms)

    for k in 1:algo.max_terms
        ERRi = zeros(M)
        for i in 1:M
            if i in selected
                continue
            end
            w_i = copy(P[:, i])
            for j in 1:(k-1)
                w_j = W[:, j]
                w_i .-= (dot(w_j, P[:, i]) / dot(w_j, w_j)) .* w_j
            end
            norm_sq = dot(w_i, w_i)
            if norm_sq > 1e-12
                ERRi[i] = ((dot(w_i, Y_target) / norm_sq)^2 * norm_sq) / Y_Y
            end
        end
        best_idx = argmax(ERRi)
        push!(selected, best_idx)
        w_best = copy(P[:, best_idx])
        for j in 1:(k-1)
            w_j = W[:, j]
            w_best .-= (dot(w_j, P[:, best_idx]) / dot(w_j, w_j)) .* w_j
        end
        W[:, k] = w_best
    end
    return selected, P[:, selected] \ Y_target
end
