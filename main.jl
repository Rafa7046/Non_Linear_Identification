using LinearAlgebra
using Statistics

include("sysid.jl")
include("utils.jl")

function main()
    dataset_name = "cascadedTanks" # "cascadedTanks" or "silverbox"
    println("=== Starting Benchmark: $dataset_name ===")

    u_train, y_train, u_test, y_test = load_normalized_dataset(dataset_name)

    plot_io(u_train, y_train, "$(uppercasefirst(dataset_name)) - IO", "src/results/$dataset_name/io.png")

    nu, ny, ne = 2, 2, 0
    nlin = 2
    tol = 0.0
    max_iter = 10
    n_max = max(nu, ny, ne)

    dm_train = data_matrix(u_train, y_train, nu=nu, ny=ny, ne=ne)
    cm_train, comb = candidate_matrix(dm_train, nlin)
    Y_train = y_train[1:(end-n_max)]

    println("\nRunning FROLS...")
    selected, ERR = frols(cm_train, Y_train, tol, max_iter)

    P_train = cm_train[:, selected]
    T = P_train \ Y_train
    y_pred_train = P_train * T

    plot_y(Y_train, y_pred_train, "$(uppercasefirst(dataset_name)) - Train", "src/results/$dataset_name/FROLS/train.png")

    esr = 1.0 - sum(ERR)
    println("\nERRi = ", round.(ERR, digits=6))
    println("ESR = $esr")
    println("Selected terms: $(length(selected)) of $(length(comb)) with tol = $tol")

    for (i, t) in zip(selected, T)
        term_str = get_model_term(comb[i], nu, ny, ne)
        println("  $term_str \t\t Weight: $(round(t, digits=4))")
    end

    println("\nValidating on Test Set...")
    dm_test = data_matrix(u_test, y_test, nu=nu, ny=ny, ne=ne)
    cm_test, _ = candidate_matrix(dm_test, nlin)

    Y_test = y_test[1:(end-n_max)]
    P_test = cm_test[:, selected]
    y_pred_test = P_test * T

    plot_y(Y_test, y_pred_test, "$(uppercasefirst(dataset_name)) - Validation", "src/results/$dataset_name/FROLS/test.png")

    mse = mean((y_pred_test .- Y_test) .^ 2)
    println("\nValidation MSE = $mse")
    println("Experiment finished.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
