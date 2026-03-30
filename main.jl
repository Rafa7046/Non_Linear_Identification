using LinearAlgebra
using Statistics

include("sysid.jl")
include("utils.jl")

# =============================================================================
# Concrete runner — wires a dataset to an algorithm
# =============================================================================

"""
    FROLSBenchmark

Concrete `BenchmarkRunner` that pairs a dataset with a `FROLSStrategy`.
"""
struct FROLSBenchmark <: BenchmarkRunner
    dataset_name::String
    algorithm::FROLSStrategy
end

get_dataset_name(r::FROLSBenchmark)    = r.dataset_name
get_algorithm(r::FROLSBenchmark)       = r.algorithm
get_algorithm_label(r::FROLSBenchmark) = "FROLS"

# =============================================================================
# Entry point
# =============================================================================

function main()
    alg    = FROLSStrategy(nu=2, ny=2, ne=0, nlin=2, tol=0.0, max_iter=10)
    runner = FROLSBenchmark("cascadedTanks", alg)
    run_benchmark(runner)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
