using LinearAlgebra

# --- 1. Free-Run Simulator ---
"""
    simulate_narx(u, y_true, selected_indices, theta; nu=2, ny=2)

Simulates the NARX model recursively (Free-Run).
It uses the true `y` only for the initial conditions, then feeds its own predictions back.
"""
function simulate_narx(u::Vector{Float64}, y_true::Vector{Float64}, selected_indices::Vector{Int}, theta::Vector{Float64}; nu=2, ny=2)
    N = length(u)
    max_lag = max(nu, ny)

    # Initialize simulation array with exact initial conditions
    y_sim = copy(y_true)

    for k in (max_lag+1):N
        # 1. Build the base linear regressor using PREDICTED y and measured u
        row_linear = [y_sim[k-1], y_sim[k-2], u[k-1], u[k-2]]

        # 2. Expand to nonlinear dictionary (Linear + Squared + Cubic)
        row_full = vcat(row_linear, row_linear .^ 2, row_linear .^ 3)

        # 3. Filter only the terms selected by the algorithm
        row_selected = row_full[selected_indices]

        # 4. Predict the current step and feed it back into y_sim
        y_sim[k] = dot(row_selected, theta)
    end

    return y_sim
end


# --- 2. SEMP Algorithm ---
"""
    semp_algorithm(u, y, P_osa, max_terms, lambda)

Implements Forward Selection with Pruning based on Simulation Error.
- `P_osa`: The One-Step-Ahead candidate matrix for estimating parameters.
- `lambda`: The complexity penalty (λ).
"""
function semp_algorithm(u::Vector{Float64}, y::Vector{Float64}, P_osa::Matrix{Float64}; max_terms::Int=5, lambda::Float64=0.01)
    N, M = size(P_osa)
    max_lag = length(y) - N
    y_target = y[max_lag+1:end]

    selected = Int[]
    best_J_global = Inf
    best_theta_global = Float64[]

    for step in 1:max_terms
        best_c = -1
        best_J_step = Inf
        best_theta_step = Float64[]

        # --- FORWARD SELECTION ---
        for c in 1:M
            if c in selected
                continue
            end

            # Temporary candidate set
            candidate_set = [selected; c]

            # 1. Estimate Parameters via Linear Least Squares (OSA)
            P_sub = P_osa[:, candidate_set]
            theta = P_sub \ y_target

            # 2. Simulate Model (Free-Run)
            y_sim = simulate_narx(u, y, candidate_set, theta)
            y_sim_target = y_sim[max_lag+1:end]

            # 3. Compute Simulation Error & Cost Function J
            sse_sim = sum((y_target .- y_sim_target) .^ 2)
            J = sse_sim + lambda * length(candidate_set)

            if J < best_J_step
                best_J_step = J
                best_c = c
                best_theta_step = theta
            end
        end

        # --- PRUNING (BACKWARD ELIMINATION) ---
        current_set = [selected; best_c]
        current_theta = best_theta_step

        if length(current_set) > 2
            for p_idx in 1:length(current_set)-1 # Try removing previously selected terms
                test_set = copy(current_set)
                deleteat!(test_set, p_idx)

                # Fit and Simulate pruned model
                P_sub = P_osa[:, test_set]
                theta_prune = P_sub \ y_target
                y_sim_prune = simulate_narx(u, y, test_set, theta_prune)

                sse_prune = sum((y_target .- y_sim_prune[max_lag+1:end]) .^ 2)
                J_prune = sse_prune + lambda * length(test_set)

                # If removing a term lowers the cost, keep it pruned!
                if J_prune < best_J_step
                    best_J_step = J_prune
                    current_set = test_set
                    current_theta = theta_prune
                    println("  [Pruned term $(selected[p_idx])!]")
                    break # Prune one at a time
                end
            end
        end

        # --- UPDATE & TERMINATION ---
        if best_J_step < best_J_global
            best_J_global = best_J_step
            selected = current_set
            best_theta_global = current_theta
            println("Step $step: Selected terms $selected | J = $(round(best_J_global, digits=4))")
        else
            println("Stopping: Cost function J did not improve. Complexity penalty prevents adding more terms.")
            break
        end
    end

    return selected, best_theta_global
end
