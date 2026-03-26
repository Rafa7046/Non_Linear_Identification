using PythonCall
using Plots
using Plots.Measures

include("frols.jl")

# ==========================================================
# TOP-LEVEL CONFIGURATION
# Set DATASET to :CascadedTanks or :Silverbox
# ==========================================================
const DATASET = :CascadedTanks
const SHOW_PLOT = true

# 1. Import Python library
nb = pyimport("nonlinear_benchmarks")

function load_data()
    println("--- Loading Dataset: $DATASET ---")

    if DATASET == :CascadedTanks
        train_val, test = nb.Cascaded_Tanks()
        u_tr, y_tr = pyconvert(Vector{Float64}, train_val.u), pyconvert(Vector{Float64}, train_val.y)
        u_te, y_te = pyconvert(Vector{Float64}, test.u), pyconvert(Vector{Float64}, test.y)
        return (u_tr, y_tr, u_te, y_te, "Cascaded Tanks")

    elseif DATASET == :Silverbox
        train_val, test = nb.Silverbox()
        # Using the multisine test set (test[0]) as per paper excerpt
        test_ms = test[0]
        u_tr, y_tr = pyconvert(Vector{Float64}, train_val.u), pyconvert(Vector{Float64}, train_val.y)
        u_te, y_te = pyconvert(Vector{Float64}, test_ms.u), pyconvert(Vector{Float64}, test_ms.y)
        return (u_tr, y_tr, u_te, y_te, "Silverbox (Multisine)")
    end
end

function run_identification()
    u_tr, y_tr, u_te, y_te, label = load_data()

    # 3. Construct a simple Candidate Dictionary
    # Example: [u(k), u(k)^2, y(k-1)]
    # Note: Using y_tr[1:end-1] and u_tr[2:end] to align dimensions for y(k-1)
    N = length(u_tr)
    y_prev = vcat(0.0, y_tr[1:end-1])
    candidates = hcat(u_tr, u_tr .^ 2, y_prev)

    # 4. Call the function from frols.jl
    println("Starting FROLS identification...")
    indices, θ = frols(y_tr, candidates, M0=3)

    println("Selected Terms: ", indices)
    println("Parameters: ", θ)
end

function evaluate_model()
    # 1. Load Data
    u_tr, y_tr, u_te, y_te, label = load_data()

    # 2. Build Training Dictionary (same as before)
    y_tr_prev = vcat(0.0, y_tr[1:end-1])
    X_tr = hcat(u_tr, u_tr .^ 2, y_tr_prev)

    # 3. Train with FROLS
    indices, θ = frols(y_tr, X_tr, M0=3)

    # 4. Build Test Dictionary (One-Step Ahead)
    # Note: Using y_te (measured) for the regressor
    y_te_prev = vcat(0.0, y_te[1:end-1])
    X_te = hcat(u_te, u_te .^ 2, y_te_prev)

    # 5. Reconstruct Time Series (OSA)
    # Filter only the columns selected by FROLS
    y_hat_osa = X_te[:, indices] * θ

    # 6. Metrics (MSE)
    mse = mean((y_te .- y_hat_osa) .^ 2)
    println("\n--- Evaluation ---")
    println("Model MSE: ", mse)

    # 7. Plot Comparison
    steps = 1:min(1024, length(y_te))

    p = plot(steps, y_te[steps],
        linecolor=:black,
        label="Measured (y_test)",
        title="$label - Model Reconstruction (OSA)",
        xlabel="timestep",
        ylabel="amplitude",
        lw=1.5,
        framestyle=:box,
        left_margin=15mm, bottom_margin=15mm)

    plot!(p, steps, y_hat_osa[steps],
        linecolor=:blue,
        linestyle=:dash,
        label="Predicted (y_hat)",
        lw=1.5)

    # Annotate MSE on the plot
    annotate!(p, steps[end] * 0.8, maximum(y_te) * 0.9,
        text("MSE: $(round(mse, digits=6))", :black, :right, 10))

    display(p)
    readline()
end

# --- Execution ---

run_identification()
evaluate_model()

if SHOW_PLOT
    u_tr, y_tr, u_te, y_te, label = load_data()

    println("Generating Wide Overlaid Plots...")

    function create_benchmark_plot(u, y, title_suffix)
        len = min(1024, length(u))
        steps = 0:len-1

        p = plot(steps, u[1:len],
            linecolor=:red,
            label="input (u)",
            xlabel="timestep",
            ylabel="amplitude",
            title="$label - $title_suffix",
            grid=true,
            framestyle=:box,
            lw=1.2,
            # LEGEND SETTINGS:
            legend=:outertopright,  # Moves legend outside to the bottom right
            left_margin=15mm,
            bottom_margin=15mm,
            right_margin=15mm)         # Added right margin so the external legend fits

        plot!(p, steps, y[1:len],
            linecolor=:black,
            label="output (y)",
            lw=1.0)
        return p
    end

    p_train = create_benchmark_plot(u_tr, y_tr, "Train Set")
    p_test = create_benchmark_plot(u_te, y_te, "Test Set")

    # 1. We make the width 1400 and height 500 for a "Wider" look
    # 2. link=:y ensures both plots use the same scale for easier comparison
    final_fig = plot(p_train, p_test,
        layout=(1, 2),
        size=(1800, 500),
        margin=15mm) # General padding between subplots

    display(final_fig)

    # Save a high-res version for your paper
    # savefig("benchmark_results.png")

    println("Press [Enter] to close.")
    readline()
end
