using Random
using Combinatorics
using LinearAlgebra

"""
    generate_combinations(elements, degree)

Generates all possible combinations (with replacement) of the given elements
up to the specified degree of nonlinearity.
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

Computes the data matrix Ψ, given the input and output vector.

# Arguments
- `u::AbstractVector`: Input vector.
- `y::AbstractVector`: Output vector.
- `nu::Int`: Number of inputs. Default is 1.
- `ny::Int`: Number of outputs. Default is 1.
- `ne::Int`: Number of moving averages. Default is 0.

# Returns
- `Ψ::Matrix`: Data matrix consisting of nu + ny + ne columns.
"""
function data_matrix(u::AbstractVector, y::AbstractVector; nu::Int=1, ny::Int=1, ne::Int=0)
    N = length(u)
    n = max(nu, ny, ne)

    U = zeros(eltype(u), N - n, nu)
    Y = zeros(eltype(y), N - n, ny)

    # Adjusting for 1-based indexing:
    # Python: U[:, -(i+1)] = u[i+1 : N-n+i+1]
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
    e = rand(MersenneTwister(), N) # Fixed seed generator can be used here if needed

    for i in 0:(ne-1)
        E[:, ne-i] = e[(i+2):(N-n+i+1)]
    end

    return hcat(Y, U, E)
end

"""
    get_model_term(idxs, nu, ny, ne=0)

Returns the model term corresponding to the given indices.

# Arguments
- `idxs::AbstractVector{Int}`: Column indices of the data matrix (1-based).
- `nu::Int`: Number of inputs.
- `ny::Int`: Number of outputs.
- `ne::Int`: Number of moving averages. Default is 0.

# Returns
- `String`: Model term corresponding to the given indices.
"""
function get_model_term(idxs::AbstractVector{Int}, nu::Int, ny::Int, ne::Int=0)
    ans_str = ""
    for i in idxs
        # 1-based index adjustments
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

Returns the candidate matrix with all possible combinations of columns
with a degree of nonlinearity given by nl.

# Arguments
- `dm::AbstractMatrix`: Data matrix.
- `nl::Int`: Degree of nonlinearity.

# Returns
- `Matrix`: Candidate matrix.
- `Vector{Vector{Int}}`: List of all possible combinations of columns.
"""
function candidate_matrix(dm::AbstractMatrix, nl::Int)
    nrows, ncols = size(dm)

    # Generate 1-based column indices
    elements = 1:ncols
    combinations = generate_combinations(elements, nl)

    # Pre-allocate the candidate matrix for performance
    cm = zeros(eltype(dm), nrows, length(combinations))

    for (j, comb) in enumerate(combinations)
        # Multiply the selected columns element-wise across the rows
        cm[:, j] = prod(dm[:, comb], dims=2)
    end

    return cm, combinations
end

"""
    frols(cm, y, tol, max_iter)

Forward Regression Orthogonal Least Squares algorithm.
Greedily selects the most significant terms from the candidate matrix `cm`
using the Error Reduction Ratio (ERR) criterion.

# Arguments
- `cm::Matrix{Float64}`: Candidate matrix.
- `y::Vector{Float64}`: Output vector.
- `tol::Float64`: Stopping tolerance (stops when unexplained variance ≤ tol).
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
