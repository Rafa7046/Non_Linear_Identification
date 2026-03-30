using Random
using Combinatorics
using LinearAlgebra

# =============================================================================
# Strategy Pattern — Algorithm interface & types
# =============================================================================

abstract type SysIdAlgorithm end

"""
    FROLSStrategy

Encapsulates all hyperparameters for the FROLS (Forward Regression Orthogonal
Least Squares) identification algorithm.
"""
Base.@kwdef struct FROLSStrategy <: SysIdAlgorithm
    nu::Int
    ny::Int
    ne::Int
    nlin::Int
    tol::Float64
    max_iter::Int
end

"""
    SysIdResult

Holds the outputs produced by any `SysIdAlgorithm` after fitting.
"""
struct SysIdResult
    selected::Vector{Int}
    ERR::Vector{Float64}
    theta::Vector{Float64}
    comb::Vector{Vector{Int}}
    nu::Int
    ny::Int
    ne::Int
    y_pred_train::Vector{Float64}
    y_pred_test::Vector{Float64}
    Y_train::Vector{Float64}
    Y_test::Vector{Float64}
end

"""
    fit(alg, u_train, y_train, u_test, y_test) -> SysIdResult

Strategy interface: runs the identification algorithm and returns a
`SysIdResult` with predictions and diagnostics for both splits.
"""
function fit(alg::FROLSStrategy, u_train, y_train, u_test, y_test)::SysIdResult
    nu, ny, ne = alg.nu, alg.ny, alg.ne
    n_max = max(nu, ny, ne)

    dm_train = data_matrix(u_train, y_train; nu=nu, ny=ny, ne=ne)
    cm_train, comb = candidate_matrix(dm_train, alg.nlin)
    Y_train = y_train[1:(end-n_max)]

    println("\nRunning FROLS...")
    selected, ERR = frols(cm_train, Y_train, alg.tol, alg.max_iter)

    P_train = cm_train[:, selected]
    theta = P_train \ Y_train
    y_pred_train = P_train * theta

    println("\nValidating on Test Set...")
    dm_test = data_matrix(u_test, y_test; nu=nu, ny=ny, ne=ne)
    cm_test, _ = candidate_matrix(dm_test, alg.nlin)
    Y_test = y_test[1:(end-n_max)]
    P_test = cm_test[:, selected]
    y_pred_test = P_test * theta

    return SysIdResult(selected, ERR, theta, comb, nu, ny, ne,
                       y_pred_train, y_pred_test, Y_train, Y_test)
end

"""
    print_algorithm_metrics(alg, result)

Print diagnostics specific to the algorithm. Dispatches on algorithm type so
each strategy can report its own relevant metrics.
"""
function print_algorithm_metrics(alg::FROLSStrategy, result::SysIdResult)
    esr = 1.0 - sum(result.ERR)
    println("\nERRi = ", round.(result.ERR, digits=6))
    println("ESR = $esr")
    println("Selected terms: $(length(result.selected)) of $(length(result.comb)) with tol = $(alg.tol)")
    for (i, t) in zip(result.selected, result.theta)
        term_str = get_model_term(result.comb[i], result.nu, result.ny, result.ne)
        println("  $term_str \t\t Weight: $(round(t, digits=4))")
    end
end

# =============================================================================
# Low-level system identification primitives
# =============================================================================

"""
    generate_combinations(elements, degree)

Generates all combinations with replacement of `elements` up to `degree`.
"""
function generate_combinations(elements, degree::Int)
    combs = Vector{Int}[]
    for d in 1:degree
        for c in with_replacement_combinations(elements, d)
            push!(combs, c)
        end
    end
    return combs
end

"""
    data_matrix(u, y; nu=1, ny=1, ne=0)

Computes the data matrix Ψ from the input and output vectors.

# Arguments
- `u::AbstractVector`: Input vector.
- `y::AbstractVector`: Output vector.
- `nu::Int`: Number of inputs. Default is 1.
- `ny::Int`: Number of outputs. Default is 1.
- `ne::Int`: Number of moving averages. Default is 0.

# Returns
- `Ψ::Matrix`: Data matrix with nu + ny (+ ne) columns.
"""
function data_matrix(u::AbstractVector, y::AbstractVector; nu::Int=1, ny::Int=1, ne::Int=0)
    N = length(u)
    n = max(nu, ny, ne)

    U = zeros(eltype(u), N - n, nu)
    Y = zeros(eltype(y), N - n, ny)

    for i in 0:(nu-1)
        U[:, nu-i] = u[(i+2):(N-n+i+1)]
    end

    for i in 0:(ny-1)
        Y[:, ny-i] = y[(i+2):(N-n+i+1)]
    end

    if ne == 0
        return hcat(Y, U)
    end

    E = zeros(Float64, N - n, ne)
    e = rand(MersenneTwister(), N)

    for i in 0:(ne-1)
        E[:, ne-i] = e[(i+2):(N-n+i+1)]
    end

    return hcat(Y, U, E)
end

"""
    get_model_term(idxs, nu, ny, ne=0)

Returns the model term string for the given column indices of the data matrix.

# Arguments
- `idxs::AbstractVector{Int}`: Column indices (1-based).
- `nu::Int`: Number of inputs.
- `ny::Int`: Number of outputs.
- `ne::Int`: Number of moving averages. Default is 0.
"""
function get_model_term(idxs::AbstractVector{Int}, nu::Int, ny::Int, ne::Int=0)
    ans_str = ""
    for i in idxs
        if i > ny + nu
            ans_str *= " e[k-$(nu + ny + ne - i + 1)]"
        elseif i > ny
            ans_str *= " u[k-$(nu + ny - i + 1)]"
        else
            ans_str *= " y[k-$(ny - i + 1)]"
        end
    end
    return strip(ans_str)
end

"""
    candidate_matrix(dm, nl)

Returns the candidate matrix of all monomial combinations of `dm` columns up
to degree `nl`, along with the list of index combinations used.
"""
function candidate_matrix(dm::AbstractMatrix, nl::Int)
    nrows, ncols = size(dm)
    elements = 1:ncols
    combinations = generate_combinations(elements, nl)

    cm = zeros(eltype(dm), nrows, length(combinations))
    for (j, comb) in enumerate(combinations)
        cm[:, j] = prod(dm[:, comb], dims=2)
    end

    return cm, combinations
end

"""
    frols(cm, y, tol, max_iter)

Forward Regression Orthogonal Least Squares. Greedily selects the most
significant terms from the candidate matrix using the ERR criterion.

# Arguments
- `cm::Matrix{Float64}`: Candidate matrix.
- `y::Vector{Float64}`: Output vector.
- `tol::Float64`: Stops when unexplained variance ≤ tol.
- `max_iter::Int`: Maximum number of terms to select.

# Returns
- `Vector{Int}`: Indices of selected columns in `cm`.
- `Vector{Float64}`: ERR values for each selected term.
"""
function frols(cm::Matrix{Float64}, y::Vector{Float64}, tol::Float64, max_iter::Int)
    N, M = size(cm)
    y_y = dot(y, y)

    selected = Int[]
    ERRi_selected = Float64[]
    W = zeros(N, max_iter)

    for k in 1:max_iter
        if 1.0 - sum(ERRi_selected) <= tol && k > 1
            break
        end

        best_err = -Inf
        best_i = -1
        best_w = zeros(N)

        for i in 1:M
            if i in selected
                continue
            end

            w_i = copy(cm[:, i])
            for j in 1:(k-1)
                w_j = W[:, j]
                w_i .-= (dot(w_j, cm[:, i]) / dot(w_j, w_j)) .* w_j
            end

            norm_sq = dot(w_i, w_i)
            if norm_sq > 1e-12
                err = (dot(w_i, y) / norm_sq)^2 * norm_sq / y_y
                if err > best_err
                    best_err = err
                    best_i = i
                    best_w = w_i
                end
            end
        end

        push!(selected, best_i)
        push!(ERRi_selected, best_err)
        W[:, k] = best_w
    end

    return selected, ERRi_selected
end
