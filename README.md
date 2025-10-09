
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

Open Julia and activate the project environment:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

This will install all required dependencies listed in the `Project.toml` and `Manifest.toml` files.

To run simulations or explore the model:

👉 Refer to the [`examples.ipynb`](examples.ipynb) notebook, which includes all necessary `using` and `include` statements, as well as an example workflow for initializing and running the model.


## Theoretical Background

The model is based on the rent-gap theory (Smith, 1979), which explains inner-city gentrification (particularly in U.S. cities) as a process driven by the difference between actual and potential ground rent. According to this theory, as cities expand, new buildings are constructed on the outskirts, while little to no investment is made in the inner-city housing stock, which gradually deteriorates.
As a result, the urban population becomes spatially divided: high-income residents settle in the newly developed suburbs, while low-income residents are increasingly concentrated in the neglected inner-city areas. This segregation persists until the gap between the potential ground rent and the actual rent being capitalized in the inner city becomes wide enough to justify profitable reinvestment. At that point, capital begins to flow back into these areas, attracting high-income residents and triggering gentrification.

For an explanation of the model's structure and agent behavior, see the `examples.ipynb` notebook.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

Flavio Brandoli
Politecnico di Torino – MSc Physics of Complex Systems
Complexity Science Hub Vienna – Thesis Host Institution
