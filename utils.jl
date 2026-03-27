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
