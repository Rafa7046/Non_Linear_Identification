function fit(algo::SEMP, u, y, Y_target, P)
    N, M = size(P)
    max_lag = length(y) - N
    selected = Int[]
    best_J_global = Inf
    best_theta_global = Float64[]

    for step in 1:algo.max_terms
        best_c = -1
        best_J_step = Inf
        best_theta_step = Float64[]

        for c in 1:M
            if c in selected
                continue
            end
            cand_set = [selected; c]
            theta = P[:, cand_set] \ Y_target
            y_sim = simulate_narx(u, y, cand_set, theta; max_lag=max_lag)
            J = sum((Y_target .- y_sim[max_lag+1:end]) .^ 2) + algo.lambda * length(cand_set)

            if J < best_J_step
                best_J_step = J
                best_c = c
                best_theta_step = theta
            end
        end

        current_set = [selected; best_c]
        current_theta = best_theta_step
        if best_J_step < best_J_global
            best_J_global = best_J_step
            selected = current_set
            best_theta_global = current_theta
        else
            break
        end
    end
    return selected, best_theta_global
end
