using PythonCall
using LinearAlgebra
using Statistics
using Plots
using Plots.Measures

# ==========================================================
# 1. CORE DATA STRUCTURES
# ==========================================================

struct ExperimentData
    name::String
    u_train::Vector{Float64}
    y_train::Vector{Float64}
    u_test::Vector{Float64}
    y_test::Vector{Float64}
end

struct ModelResult
    algo_name::String
    selected_indices::Vector{Int}
    theta::Vector{Float64}
    y_hat_test::Vector{Float64}
    mse::Float64
    mae::Float64
end

# ==========================================================
# 2. STRATEGY INTERFACE & ALGORITHM STRUCTS
# ==========================================================

abstract type SysIdAlgorithm end

Base.@kwdef struct FROLS <: SysIdAlgorithm
    max_terms::Int = 5
end

Base.@kwdef struct GSERR <: SysIdAlgorithm
    max_terms::Int = 5
end

Base.@kwdef struct SEMP <: SysIdAlgorithm
    max_terms::Int = 5
    lambda::Float64 = 0.05
end

# ==========================================================
# 3. HELPER FUNCTIONS
# ==========================================================

function load_dataset(dataset_sym::Symbol)
    nb = pyimport("nonlinear_benchmarks")
    if dataset_sym == :CascadedTanks
        train_val, test = nb.Cascaded_Tanks()
        u_tr, y_tr = pyconvert(Vector{Float64}, train_val.u), pyconvert(Vector{Float64}, train_val.y)
        u_te, y_te = pyconvert(Vector{Float64}, test.u), pyconvert(Vector{Float64}, test.y)
        return ExperimentData("Cascaded Tanks", u_tr, y_tr, u_te, y_te)
    elseif dataset_sym == :Silverbox
        train_val, test = nb.Silverbox()
        test_ms = test[0]
        u_tr, y_tr = pyconvert(Vector{Float64}, train_val.u), pyconvert(Vector{Float64}, train_val.y)
        u_te, y_te = pyconvert(Vector{Float64}, test_ms.u), pyconvert(Vector{Float64}, test_ms.y)
        return ExperimentData("Silverbox", u_tr, y_tr, u_te, y_te)
    else
        error("Unknown dataset")
    end
end

function build_narx_dictionary(u, y; max_lag=2)
    N = length(y)
    Y_target = y[max_lag+1:end]
    P_linear = zeros(N - max_lag, max_lag * 2)

    for k in (max_lag+1):N
        idx = k - max_lag
        for i in 1:max_lag
            P_linear[idx, i] = y[k-i]
            P_linear[idx, max_lag+i] = u[k-i]
        end
    end
    P_nonlinear = hcat(P_linear, P_linear .^ 2, P_linear .^ 3)
    return Y_target, P_nonlinear
end

function simulate_narx(u, y_true, selected_indices, theta; max_lag=2)
    N = length(u)
    y_sim = copy(y_true)

    for k in (max_lag+1):N
        row_linear = Float64[]
        for i in 1:max_lag
            push!(row_linear, y_sim[k-i])
        end
        for i in 1:max_lag
            push!(row_linear, u[k-i])
        end

        row_full = vcat(row_linear, row_linear .^ 2, row_linear .^ 3)
        y_sim[k] = dot(row_full[selected_indices], theta)
    end
    return y_sim
end

# ==========================================================
# 4. ALGORITHM IMPLEMENTATIONS (Multiple Dispatch)
# ==========================================================

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

# ==========================================================
# 5. TEMPLATE METHOD: EXPERIMENT RUNNER & PLOTTER
# ==========================================================

function run_benchmark(data::ExperimentData, algorithms::Vector{SysIdAlgorithm}; max_lag=2)
    println("\n=== Starting Benchmark: $(data.name) ===")

    Y_tr, P_tr = build_narx_dictionary(data.u_train, data.y_train; max_lag=max_lag)
    results = ModelResult[]

    for algo in algorithms
        algo_name = string(typeof(algo))
        println("-> Training $algo_name...")

        # Dispatch dynamically calls the correct fit() based on the struct type
        indices, theta = fit(algo, data.u_train, data.y_train, Y_tr, P_tr)

        # Predict using Free-Run Simulation to ensure fair comparison
        y_hat = simulate_narx(data.u_test, data.y_test, indices, theta; max_lag=max_lag)

        Y_target = data.y_test[max_lag+1:end]
        y_hat_target = y_hat[max_lag+1:end]

        mse = mean((Y_target .- y_hat_target) .^ 2)
        mae = mean(abs.(Y_target .- y_hat_target))

        println("   Test MSE: $(round(mse, digits=6))")
        push!(results, ModelResult(algo_name, indices, theta, y_hat_target, mse, mae))
    end

    plot_comparison(data, results; max_lag=max_lag)
    return results
end

function plot_comparison(data::ExperimentData, results::Vector{ModelResult}; max_lag=2)
    Y_target = data.y_test[max_lag+1:end]
    len = min(1024, length(Y_target))
    steps = 0:len-1

    p = plot(steps, Y_target[1:len],
        label="Measured", linecolor=:black, lw=2,
        xlabel="timestep", ylabel="amplitude",
        title="$(data.name) - Algorithm Comparison",
        grid=true, framestyle=:box,
        legend=:outerbottomright,
        left_margin=15mm, bottom_margin=15mm, right_margin=45mm)

    colors = [:red, :blue, :green, :orange, :purple]
    for (i, res) in enumerate(results)
        plot!(p, steps, res.y_hat_test[1:len],
            label="$(res.algo_name) (MSE: $(round(res.mse, digits=4)))",
            linecolor=colors[i], linestyle=:dash, lw=1.5)
    end

    display(plot(p, size=(1400, 500)))
end

# ==========================================================
# 6. EXECUTION SCRIPT
# ==========================================================

function main()
    # 1. Define the algorithms to compare
    algos = SysIdAlgorithm[
        FROLS(max_terms=5),
        GSERR(max_terms=5),
        SEMP(max_terms=5, lambda=0.05)
    ]

    # 2. Run on Cascaded Tanks
    tanks_data = load_dataset(:CascadedTanks)
    run_benchmark(tanks_data, algos)

    # 3. Run on Silverbox (Uncomment to run both sequentially)
    # silverbox_data = load_dataset(:Silverbox)
    # run_benchmark(silverbox_data, algos)

    println("\nPress [Enter] to exit.")
    readline()
end

main()
