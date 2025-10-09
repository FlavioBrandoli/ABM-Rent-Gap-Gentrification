
# ABM-Rent-Gap-Gentrification

Agent-based model developed in Julia to study gentrification, developed for my MSc thesis in the PCS program (Physics of Complex Systems), Politecnico di Torino – thesis project hosted at Complexity Science Hub Vienna.

## Overview

This project implements an Agent-Based Model (ABM) to explore the dynamics of gentrification, particularly inspired by rent-gap theory. The model is written in Julia and uses the [Agents.jl](https://juliadynamics.github.io/Agents.jl/stable/) library.

The core of the model includes agent definitions, environment setup, simulation logic, and utility functions. A Jupyter notebook is provided to demonstrate how to run the model and visualize basic results.

## Repository Structure

- `agents.jl` – defines the agent types used in the simulation.
- `model.jl` – contains the model initialization function and the step functions.
- `utils.jl` – includes helper functions used to run and analyze the simulation.
- `examples.ipynb` – Jupyter notebook with a simple simulation run and example plots.
- `Project.toml` / `Manifest.toml` – define the Julia environment for reproducibility.
- `README.md` – project description and usage instructions.

## Getting Started

### Requirements

- Julia (>= 1.6 recommended)
- [Agents.jl](https://github.com/JuliaDynamics/Agents.jl)
- Other dependencies are listed in `Project.toml`

### Installation

Clone the repository:

```bash
git clone https://github.com/FlavioBrandoli/ABM-Rent-Gap-Gentrification.git
cd ABM-Rent-Gap-Gentrification
````

Open Julia and activate the environment:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Now you can include the files and run your simulations. For example:

```julia
include("model.jl")
# define and run your model...
```

Or explore the `examples.ipynb` notebook for a basic simulation and some output plots.

## Theoretical Background

The model is based on the rent-gap theory (Smith, 1979), which explains gentrification as a process driven by the difference between actual and potential ground rent. Agents interact spatially, with behaviors designed to reflect investment dynamics, affordability, and displacement mechanisms.

> A more detailed explanation of the model can be found in the associated MSc thesis (link coming soon).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

Flavio Brandoli
Politecnico di Torino – MSc Physics of Complex Systems
Complexity Science Hub Vienna – Thesis Host Institution

```

