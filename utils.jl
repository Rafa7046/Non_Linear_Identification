using PythonCall
using Plots
using Plots.Measures

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
