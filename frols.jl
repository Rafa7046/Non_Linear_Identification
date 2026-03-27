function fit(algo::FROLS, u, y, Y_target, P)
    N, M = size(P)
    selected = Int[]
    rem_idx = collect(1:M)
    Q = zeros(N, algo.max_terms)
    σy = var(Y_target) * (N - 1) / N

    for s in 1:algo.max_terms
        best_err = -Inf
        best_i = -1
        best_q = zeros(N)
        for (i, idx) in enumerate(rem_idx)
            q_m = copy(P[:, idx])
            for r in 1:(s-1)
                qr = Q[:, r]
                q_m .-= (dot(qr, P[:, idx]) / dot(qr, qr)) .* qr
            end
            den = dot(q_m, q_m)
            if den < 1e-12
                continue
            end
            err = ((dot(Y_target, q_m) / den)^2 * den) / (N * σy)
            if err > best_err
                best_err = err
                best_i = i
                best_q = q_m
            end
        end
        push!(selected, rem_idx[best_i])
        Q[:, s] .= best_q
        deleteat!(rem_idx, best_i)
    end
    return selected, P[:, selected] \ Y_target
end
