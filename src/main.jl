using PythonCall
using Plots

# ==========================================================
# TOP-LEVEL CONFIGURATION
# Set DATASET to :CascadedTanks or :Silverbox
# ==========================================================
const DATASET = :Silverbox  # Options: :CascadedTanks, :Silverbox
const SHOW_PLOT = true

# 1. Import the Python library
nb = pyimport("nonlinear_benchmarks")

function load_data()
    println("--- Selected Dataset: $DATASET ---")

    if DATASET == :CascadedTanks
        # Load Cascaded Tanks
        train_val, test = nb.Cascaded_Tanks()

        # Unpack Data
        u_train, y_train = pyconvert(Vector{Float64}, train_val.u), pyconvert(Vector{Float64}, train_val.y)
        u_test,  y_test  = pyconvert(Vector{Float64}, test.u),      pyconvert(Vector{Float64}, test.y)

        ts = pyconvert(Float64, train_val.sampling_time)
        init_win = pyconvert(Int, test.state_initialization_window_length)

        println("Initialization Window Length: $init_win")
        return (u_train, y_train, u_test, y_test, ts, "Cascaded Tanks")

    elseif DATASET == :Silverbox
        # Load Silverbox
        train_val, test = nb.Silverbox()

        # Silverbox 'test' is a list of datasets
        # test[0] = multisine, test[1] = arrow_full, test[2] = arrow_no_extrapolation
        test_multisine = test[0]

        u_train, y_train = pyconvert(Vector{Float64}, train_val.u), pyconvert(Vector{Float64}, train_val.y)
        u_test,  y_test  = pyconvert(Vector{Float64}, test_multisine.u), pyconvert(Vector{Float64}, test_multisine.y)

        ts = pyconvert(Float64, train_val.sampling_time)
        init_win = pyconvert(Int, test_multisine.state_initialization_window_length)

        println("Initialization Window Length (Multisine Test): $init_win")
        return (u_train, y_train, u_test, y_test, ts, "Silverbox (Multisine)")

    else
        error("Unknown dataset selected in TOP-LEVEL macros.")
    end
end

# --- Execution Logic ---

# 1. Extract the data
u_tr, y_tr, u_te, y_te, ts, label = load_data()

# 2. Plotting
if SHOW_PLOT
    println("Generating plots...")

    # Create time vectors
    t_tr = (0:length(u_tr)-1) .* ts
    t_te = (0:length(u_te)-1) .* ts

    # Plot Training Data (Subset for visibility if data is huge)
    limit = min(5000, length(u_tr))
    p1 = plot(t_tr[1:limit], [u_tr[1:limit] y_tr[1:limit]],
              layout=(2,1), title="$label - Train (Subset)", label=["u" "y"])

    # Plot Test Data
    limit_test = min(5000, length(u_te))
    p2 = plot(t_te[1:limit_test], [u_te[1:limit_test] y_te[1:limit_test]],
              layout=(2,1), title="$label - Test (Subset)", label=["u" "y"])

    display(plot(p1, p2, layout=(1,2), size=(1200, 700)))

    println("\nPlot rendered. Press [Enter] to exit.")
    readline()
end
