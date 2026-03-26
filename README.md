# Nonlinear System Identification Benchmarks

This project provides a Julia-based interface to load and visualize classic nonlinear system identification datasets (Cascaded Tanks and Silverbox) using the `nonlinear-benchmarks` Python library.

## Prerequisites

1. **Julia**: [Download and install Julia](https://julialang.org/downloads/).
2. **Conda**: Ensure you have [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html) installed.

## Setup Instructions

### 1. Python Environment
Create and configure the Conda environment named `nl_ident`:

```bash
conda create -n nl_ident python -y
conda activate nl_ident
pip install -r requirements.txt
```

### 2. Julia Packages
Install the required Julia dependencies. Open your terminal in the project root and run:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

_This command reads the Project.toml and installs the exact versions of PythonCall, Plots, etc._

### How to Run
The project uses a shell script to ensure Julia connects to the correct Conda environment.

1. Make the script executable:

```bash
chmod +x run_julia.sh
```

2. Run the project:

```bash
bash run_julia.sh
```
