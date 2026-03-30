using PythonCall
using Plots
using Plots.Measures
using Statistics

# =============================================================================
# Template Method Pattern — Benchmark framework
# =============================================================================

abstract type BenchmarkRunner end

"""
    run_benchmark(runner)

Template method: fixed benchmark skeleton that delegates each phase to an
overridable hook. Concrete runners may override individual hooks via multiple
dispatch without altering the overall flow.
"""
function run_benchmark(runner::BenchmarkRunner)
    println("=== Starting Benchmark: $(get_dataset_name(runner)) ===")

    u_train, y_train, u_test, y_test = load_data(runner)
    plot_input_output(runner, u_train, y_train)

    result = identify(runner, u_train, y_train, u_test, y_test)

    print_metrics(runner, result)
    plot_results(runner, result)

    println("Experiment finished.")
end

# --- Overridable hooks ---

"""Return the dataset name for this runner."""
function get_dataset_name(runner::BenchmarkRunner)::String
    error("get_dataset_name not implemented for $(typeof(runner))")
end

"""Return the algorithm strategy for this runner."""
function get_algorithm(runner::BenchmarkRunner)::SysIdAlgorithm
    error("get_algorithm not implemented for $(typeof(runner))")
end

"""Return a short label used in result paths (e.g. \"FROLS\")."""
function get_algorithm_label(runner::BenchmarkRunner)::String
    return string(typeof(get_algorithm(runner)))
end

"""Load and return (u_train, y_train, u_test, y_test) for this runner."""
function load_data(runner::BenchmarkRunner)
    return load_normalized_dataset(get_dataset_name(runner))
end

"""Plot raw input/output signals."""
function plot_input_output(runner::BenchmarkRunner, u, y)
    name = get_dataset_name(runner)
    plot_io(u, y, "$(uppercasefirst(name)) - IO", "src/results/$name/io.png")
end

"""Run identification via the runner's strategy and return a SysIdResult."""
function identify(runner::BenchmarkRunner, u_train, y_train, u_test, y_test)
    return fit(get_algorithm(runner), u_train, y_train, u_test, y_test)
end

"""
    print_metrics(runner, result)

Print algorithm-specific diagnostics followed by the common validation MSE.
Override `print_algorithm_metrics` in `sysid.jl` to customise per-algorithm output.
"""
function print_metrics(runner::BenchmarkRunner, result::SysIdResult)
    print_algorithm_metrics(get_algorithm(runner), result)
    mse = mean((result.y_pred_test .- result.Y_test) .^ 2)
    println("\nValidation MSE = $mse")
end

"""Save and display train/validation prediction plots."""
function plot_results(runner::BenchmarkRunner, result::SysIdResult)
    name  = get_dataset_name(runner)
    label = get_algorithm_label(runner)
    plot_y(result.Y_train, result.y_pred_train,
           "$(uppercasefirst(name)) - Train",
           "src/results/$name/$label/train.png")
    plot_y(result.Y_test, result.y_pred_test,
           "$(uppercasefirst(name)) - Validation",
           "src/results/$name/$label/test.png")
end

# =============================================================================
# Data loading utilities
# =============================================================================

function load_normalized_dataset(dataset_name::String)
    nb = pyimport("nonlinear_benchmarks")
    if dataset_name == "cascadedTanks"
        train_val, test = nb.Cascaded_Tanks()
        u_tr, y_tr = pyconvert(Vector{Float64}, train_val.u), pyconvert(Vector{Float64}, train_val.y)
        u_te, y_te = pyconvert(Vector{Float64}, test.u), pyconvert(Vector{Float64}, test.y)

        mu_train, mu_test = maximum(abs.(u_tr)), maximum(abs.(u_te))
        my_train, my_test = maximum(abs.(y_tr)), maximum(abs.(y_te))

        return u_tr ./ mu_train, y_tr ./ my_train, u_te ./ mu_test, y_te ./ my_test
    elseif dataset_name == "silverbox"
        train_val, test = nb.Silverbox()
        test_ms = test[0]
        u_tr, y_tr = pyconvert(Vector{Float64}, train_val.u), pyconvert(Vector{Float64}, train_val.y)
        u_te, y_te = pyconvert(Vector{Float64}, test_ms.u), pyconvert(Vector{Float64}, test_ms.y)

        mu, my = maximum(abs.(u_tr)), maximum(abs.(y_tr))
        n_train = Int(floor(0.2 * length(u_tr)))
        return (u_tr[end-n_train+1:end] ./ mu, y_tr[end-n_train+1:end] ./ my,
                u_tr[1:end-n_train] ./ mu, y_tr[1:end-n_train] ./ my)
    else
        error("Dataset mapping not implemented for: $dataset_name")
    end
end

# =============================================================================
# Plotting utilities
# =============================================================================

function plot_io(u, y, title_str, save_path)
    p = plot(layout=(2, 1), size=(800, 500), plot_title=title_str)
    plot!(p[1], u, label="u", linecolor=:red, ylabel="Input")
    plot!(p[2], y, label="y", linecolor=:black, ylabel="Output", xlabel="Samples")

    mkpath(dirname(save_path))
    savefig(p, save_path)
    display(p)

    println(">>> Plot displayed: $title_str")
    println(">>> Press [Enter] to continue execution...")
    readline()
end

function plot_y(y, y_pred, title_str, save_path)
    p = plot(y, label="True (y)", linecolor=:black, lw=1.5, title=title_str)
    plot!(p, y_pred, label="Predicted (y_hat)", linecolor=:red, linestyle=:dash, lw=1.5)

    mkpath(dirname(save_path))
    savefig(p, save_path)
    display(p)

    println(">>> Plot displayed: $title_str")
    println(">>> Press [Enter] to continue execution...")
    readline()
end
