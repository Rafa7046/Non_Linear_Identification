using PythonCall
using Plots
using Plots.Measures

include("frols.jl")
include("gram_schmidt.jl")
include("semp.jl")

# ==========================================================
# TOP-LEVEL CONFIGURATION
# Set DATASET to :CascadedTanks or :Silverbox
# ==========================================================
const DATASET = :Silverbox
const SHOW_PLOT = false
const N_TERMS = 5

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

function run_frols_identification()
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

# --- 3. Execution & Evaluation ---
function run_gram_identification()
    u_tr, y_tr, u_te, y_te, label = load_data()

    println("Building Candidate Matrices...")
    Y_tr, P_tr = build_narx_dictionary(u_tr, y_tr, nu=2, ny=2)
    Y_te, P_te = build_narx_dictionary(u_te, y_te, nu=2, ny=2)

    println("Running Modified Gram-Schmidt with ERR...")
    selected_indices = gram_schmidt_err(Y_tr, P_tr, N_TERMS)

    # Filter dictionaries to only selected columns
    P_tr_sel = P_tr[:, selected_indices]
    P_te_sel = P_te[:, selected_indices]

    # Compute Parameters using Least Squares (\ operator)
    θ_hat = P_tr_sel \ Y_tr
    println("\nEstimated Parameters (Θ): ", θ_hat)

    # Predict on Test Set (One-Step Ahead)
    y_hat_te = P_te_sel * θ_hat

    # Compute Metrics
    mae = mean(abs.(Y_te .- y_hat_te))
    mse = mean((Y_te .- y_hat_te) .^ 2)
    println("Test MAE: ", mae)
    println("Test MSE: ", mse)

    # Plotting (Paper Style)
    len = min(1024, length(Y_te))
    steps = 0:len-1

    p = plot(steps, Y_te[1:len],
        linecolor=:black,
        label="Measured (y)",
        xlabel="timestep",
        ylabel="amplitude",
        title="$label - Test Set (GS-ERR)",
        grid=true,
        framestyle=:box,
        lw=1.5,
        legend=:outerbottomright,
        left_margin=15mm,
        bottom_margin=15mm,
        right_margin=25mm)

    plot!(p, steps, y_hat_te[1:len],
        linecolor=:red,
        linestyle=:dash,
        label="Predicted (y_hat)",
        lw=1.5)

    display(plot(p, size=(1200, 500)))

    println("\nPress [Enter] to exit.")
    readline()
end

function run_semp_identification()
    u_tr, y_tr, u_te, y_te, label = load_data()

    println("Building Candidate Matrices...")
    # nu=2, ny=2 gives us a max lag of 2
    Y_tr, P_tr = build_narx_dictionary(u_tr, y_tr, nu=2, ny=2)

    # 1. Run SEMP Algorithm
    # lambda controls complexity. Higher lambda = fewer terms selected.
    lambda_val = 0.05
    println("\nRunning SEMP (lambda=$lambda_val)...")
    selected_indices, θ_hat = semp_algorithm(u_tr, y_tr, P_tr, max_terms=6, lambda=lambda_val)

    println("\nFinal Selected Term Indices: ", selected_indices)
    println("Estimated Parameters (Θ): ", θ_hat)

    # 2. Evaluate on Test Set using Free-Run Simulation
    # Note: simulate_narx takes the full arrays and handles the initial lags internally
    y_hat_te = simulate_narx(u_te, y_te, selected_indices, θ_hat)

    # 3. Compute Test Metrics
    max_lag = 2
    Y_target_te = y_te[max_lag+1:end]
    y_hat_target_te = y_hat_te[max_lag+1:end]

    mse = mean((Y_target_te .- y_hat_target_te) .^ 2)
    mae = mean(abs.(Y_target_te .- y_hat_target_te))
    println("Test Free-Run MSE: ", mse)
    println("Test Free-Run MAE: ", mae)

    # 4. Plotting (Paper Style)
    println("Generating Plot...")
    len = min(1024, length(Y_target_te))
    steps = 0:len-1

    p = plot(steps, Y_target_te[1:len],
        linecolor=:black,
        label="Measured (y)",
        xlabel="timestep",
        ylabel="amplitude",
        title="$label - Test Set (SEMP Free-Run)",
        grid=true,
        framestyle=:box,
        lw=1.5,
        legend=:outerbottomright,
        left_margin=15mm,
        bottom_margin=15mm,
        right_margin=25mm)

    plot!(p, steps, y_hat_target_te[1:len],
        linecolor=:red,
        linestyle=:dash,
        label="Predicted (y_hat)",
        lw=1.5)

    # Add MSE annotation inside the plot for easy reading in the paper
    annotate!(p, steps[end] * 0.8, maximum(Y_target_te[1:len]) * 0.9,
        text("MSE: $(round(mse, digits=6))", :black, :right, 10))

    display(plot(p, size=(1400, 500)))

    println("\nPress [Enter] to exit.")
    readline()
end

# --- Execution ---

# run_frols_identification()
# evaluate_model()
# run_gram_identification()
run_semp_identification()

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
