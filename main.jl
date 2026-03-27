using PythonCall
using Plots
using Plots.Measures
using LinearAlgebra
using Statistics

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

include("utils.jl")
include("frols.jl")
include("gram_schmidt.jl")
include("semp.jl")

# ==========================================================
# TOP-LEVEL CONFIGURATION
# Set DATASET to :CascadedTanks or :Silverbox
# ==========================================================
const DATASET = :CascadedTanks
const N_TERMS = 5
const LAMBDA = 0.05

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

function main()
    algorithms = SysIdAlgorithm[
        FROLS(max_terms=N_TERMS),
        GSERR(max_terms=N_TERMS),
        SEMP(max_terms=N_TERMS, lambda=LAMBDA)
    ]

    data = load_dataset(DATASET)
    run_benchmark(data, algorithms)

    println("\nPress [Enter] to exit.")
    readline()
end

main()
