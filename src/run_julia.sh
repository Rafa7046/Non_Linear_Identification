#!/bin/bash

# 2. Tell PythonCall exactly which Python to use
# This points directly to the python executable inside your conda env
export JULIA_PYTHONCALL_EXE=$(conda run -n nl_ident which python)

# 3. Execute Julia with your script
julia src/main.jl
