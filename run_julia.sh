#!/bin/bash
#
# Path to the python executable in conda env
export JULIA_PYTHONCALL_EXE=$(conda run -n nl_ident which python)
echo "Using Python from: $JULIA_PYTHONCALL_EXE"

# Run julia using the local project environment
julia --project=. main.jl
