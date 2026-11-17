# ABM-Rent-Gap-Gentrification

Agent-based rent-gap model of gentrification with public-housing interventions, in Julia
with [Agents.jl](https://juliadynamics.github.io/Agents.jl/stable/).

Companion code of F. Brandoli and V. D. P. Servedio, *Small Interventions, Big Effects:
Targeted Public Housing Against Gentrification* (submitted to *Cities*). The model was
developed for Flavio Brandoli's MSc thesis (Physics of Complex Systems, Politecnico di
Torino), hosted at the Complexity Science Hub Vienna.

This repository contains what is needed to rebuild the simulation summaries, the figures and
Table 2 of the paper from the saved results, and to re-run the simulations that produce them.
Two items of the paper are not produced here: the thermalization schematic (Fig. 1) and the
empirical validation (Fig. 2). See [What is not included](#what-is-not-included).

## Contents

- [Quick start](#quick-start)
- [Reproducing the figures and Table 2 from the saved results](#reproducing-the-figures-and-table-2-from-the-saved-results)
- [Re-running the simulations](#re-running-the-simulations)
- [Environments and the random stream](#environments-and-the-random-stream)
- [The model in one paragraph](#the-model-in-one-paragraph)
- [Repository layout](#repository-layout)
- [Switches](#switches)
- [What is not included](#what-is-not-included)
- [License](#license)

## Quick start

Requirements: Julia 1.10 or later for the root environment (verified with Julia 1.13.3). To
reproduce the paper's runs exactly, use `env-paper/` with Julia 1.9 (see
[Environments and the random stream](#environments-and-the-random-stream)).

```bash
git clone <this repository> && cd ABM-Rent-Gap-Gentrification
julia --project -e 'import Pkg; Pkg.instantiate()'     # root environment (Julia >= 1.10)
julia --project test/runtests.jl                        # smoke tests: all should pass
julia -t auto --project scripts/compare_scenarios.jl --fraction=0.07 --tag=base   # produces results/compare_base_summary.csv
julia --project scripts/plot_fig9.jl --tag=base         # a first figure, from that summary
```

The CSV outputs are not versioned in this repository. They are produced by the simulation
scripts; see [Re-running the simulations](#re-running-the-simulations) for the full list.

Every script takes `--key=value` options and has a header that shows its usage. Run them from
the repository root. Tables go to `results/`, figures to `figures/` (PDF and PNG).

## Reproducing the figures and Table 2 from the saved results

The plotting scripts read the summaries that the simulation scripts write to `results/`. Those
CSV files are not versioned in this repository. Produce them first with the commands in
[Re-running the simulations](#re-running-the-simulations); after that, the commands below need no
further simulation and take seconds to a few minutes each.

**Collected datasets.** The summaries are not published with the repository yet. If a collected
archive is deposited, its link and checksum will be given here and the files go into `results/`
unchanged. Until then, reproduce them with the commands below.

The CSV outputs can be stored gzipped (`results/**/*.csv.gz`, about a third of the size). The scripts
read plain `.csv` files, so decompress them before use, keeping the archives if you want:

```bash
find results -name '*.csv.gz' -exec gunzip -k {} +
```

| Manuscript | Content | Command | Output |
|---|---|---|---|
| Fig. 3 | City maps under the three scenarios | `julia --project scripts/city_maps.jl --seed=580 --fraction=0.07` | `figures/PH_random_seed580_f0.07.pdf`, `figures/PH_rentgap_seed580_f0.07.pdf`, `figures/city_maps_seed580_f0.07.pdf` |
| Fig. 4 | Inner-city median income and agent counts, f = 0.07 | `julia --project scripts/plot_fig4.jl --tag=base` | `figures/median_income_comparison_base.pdf`, `figures/number_agents_base.pdf` |
| Fig. 5 | Final inner-city income vs f (dose-response) | `julia --project scripts/plot_sweep.jl --tags=base` | `figures/final_median_income_base.pdf` |
| Fig. 6 | Gentrification clusters before and after the intervention | `julia --project scripts/city_maps.jl --seed=580 --fraction=0.07` | `figures/cluster_before_seed580_f0.06.pdf`, `figures/cluster_after_seed580_f0.06.pdf` |
| Fig. 7 | Sensitivity: spillover and activation time | `julia --project scripts/plot_sweep.jl --tags=base,spill0,spill4,trig025,trig075 --out=sensitivity_spillover_timing` | `figures/final_median_income_sensitivity_spillover_timing.pdf` |
| Fig. 8 | Sensitivity: vacancy, eligibility, grid | `julia --project scripts/plot_sweep.jl --tags=vac005,vac010,inc20,inc45,grid60 --out=sensitivity_vacancy_eligibility_grid` | `figures/final_median_income_sensitivity_vacancy_eligibility_grid.pdf` |
| Fig. 9 | Neighbor segregation and dissimilarity index | `julia --project scripts/plot_fig9.jl --tag=base` | `figures/neighbor_segregation_base.pdf`, `figures/dissimilarity_index_base.pdf` |
| Fig. 10 | Pairwise dissimilarity between the three income tiers | `julia --project scripts/plot_fig10.jl --tag=base` | `figures/dissimilarity_pair_base.pdf` |
| Fig. 11 | Decomposed Theil index | `julia --project scripts/plot_fig11.jl --tag=base` | `figures/theil_base.pdf` |
| Graphical abstract | Result panel | `julia --project scripts/graphical_abstract_panel.jl` | `figures/ga_result_panel.pdf` |
| Table 2 | Saturation point f*, minimum income, income at f = 0.06, eligible pool | `julia --project scripts/table_sensitivity.jl` | `results/table2_sensitivity.csv` |

Figs. 4 and 9–11 read `results/compare_base_summary.csv`. Figs. 5, 7, 8 and the graphical abstract
read `results/sweep_<tag>_summary.csv`. Table 2 reads the sweep summaries and
`results/candidate_profile_<tag>.csv`. All CSV files in `results/` are git-ignored. The per-step tables
`results/*_steps.csv` (tens of MB each) are written by the sweeps, and none of the commands above
reads them.

## Re-running the simulations

The summaries above come from the runs listed below. The defaults of the scripts are the paper's
settings; each row changes one ingredient. Every run uses the 15 paper seeds (`--seeds=paper`,
defined as `PAPER_SEEDS` in `src/parameters.jl`) and 170 dynamic steps, and writes
`results/sweep_<tag>_{steps,final,summary}.csv`. Add `-t auto` to run seeds in parallel.

| Tag | Arguments (in addition to the defaults) | Used in |
|---|---|---|
| `base` | (none) | Figs. 4, 5, 9–11, graphical abstract, Table 2 |
| `spill0` | `--spillover=0` | Fig. 7, Table 2 |
| `spill4` | `--spillover=4` | Fig. 7, Table 2 |
| `trig025` | `--trigger=0.25` | Figs. 7, 8, Table 2 |
| `trig075` | `--trigger=0.75` | Figs. 7, 8, Table 2 |
| `vac005` | `--vacancy=0.05` | Fig. 8, Table 2 |
| `vac010` | `--vacancy=0.10` | Fig. 8, Table 2 |
| `inc20` | `--income_threshold=20` | Fig. 8, Table 2 |
| `inc45` | `--income_threshold=45` | Fig. 8, Table 2 |
| `grid60` | `--dims=60 --steps=400 --dev_invest=2 --dev_sample=68` | Fig. 8, Table 2 |

```bash
julia -t auto --project scripts/sweep_fraction.jl --tag=base
julia -t auto --project scripts/sweep_fraction.jl --tag=spill0 --spillover=0
julia -t auto --project scripts/sweep_fraction.jl --tag=grid60 --dims=60 --steps=400 --dev_invest=2 --dev_sample=68
# ... one line per tag in the table above
julia -t auto --project scripts/compare_scenarios.jl --fraction=0.07 --tag=base    # inputs of Figs. 4 and 9–11
julia -t auto --project scripts/candidate_profile.jl --tag=base                    # eligible houses (Table 2)
```

For the 60×60 grid, the developer's capacity is scaled with the stock (2 renovations per step and
a sample of 68 vacant houses), as in the Methods. `--dev_invest` takes an integer.

The default `--tag` of `sweep_fraction.jl` and `compare_scenarios.jl` is `paper`. Pass `--tag=base`
to get the file names used in the table above.

## Environments and the random stream

The random stream of a seed depends on the Julia version: the same seed gives a different city on
Julia 1.9 and on Julia 1.13, even with the same `StatsBase` and `Distributions` versions. The 15
paper seeds were selected as "successful gentrification" runs under the original setup, so they
reproduce the paper's results under that setup. Two environments are provided.

* `env-paper/` (Julia 1.9.4, Agents 6.2.10, StatsBase 0.34.4, Distributions 0.25.118) reproduces
  the paper. With [juliaup](https://github.com/JuliaLang/juliaup): `juliaup add 1.9`.

  ```bash
  julia +1.9 --project=env-paper -e 'import Pkg; Pkg.instantiate()'
  julia +1.9 --project=env-paper test/runtests.jl
  julia +1.9 -t auto --project=env-paper scripts/sweep_fraction.jl --tag=base
  ```

* The root `Project.toml` (Julia 1.10 and later) is a lean environment with the same model and
  switches. Its random stream differs, so only part of the paper seeds gentrify in it.
  `results/v1/julia1.13/` is where the first-submission sweeps run in this environment on Julia 1.13
  are written (with the first-submission switches, see below).

## The model in one paragraph

A 40×40 grid holds a central business district, amenity patches and houses. Each house has an age
(older towards the centre, as in Smith's rent-gap picture), a potential rent set by nearby amenities
and a capitalized rent that decays with age; the rent gap is their difference. Households (incomes
sampled from the 2013 ACS distribution, split at the 60th percentile into low and high income)
relocate Schelling-style when their utility (affordability, amenities, housing age, centrality)
drops below a threshold. After 15 thermalization steps a developer starts renovating the vacant
houses with the largest rent gap, raising the potential rent of the neighbours (positive spillover)
and adapting its investment rate to the occupancy of past renovations. When the inner-city median
rent has grown by 50%, a public-housing policy converts a fraction `f` of the stock, either at
random or targeting the highest rent-gap eligible houses. Public units get a fixed low rent, are
closed to the developer and lower the potential rent of their neighbours (negative spillover).

## Repository layout

```
src/RentGapCity.jl     module entry point (include it, then `using .RentGapCity`)
src/parameters.jl      ModelParams (paper values as defaults) and PAPER_SEEDS
src/agents.jl          Household agent
src/city.jl            City state (patch matrices, developer and policy state)
src/init.jl            initialize_model
src/utility.jl         household utility, vacancy lists, sampling helpers
src/dynamics.jl        agent_step!, model_step! (developer, spillovers, ageing)
src/policy.jl          public-housing conversion (random / rent-gap)
src/metrics.jl         per-step indicators, dissimilarity, Theil, cluster sizes
src/run.jl             run_scenario, run_multi_seed (threaded), summaries
scripts/common.jl      shared preamble: options, output folders, seeds
scripts/sweep_fraction.jl          dose-response sweep over f (simulations)
scripts/compare_scenarios.jl       three-scenario time series (simulations)
scripts/candidate_profile.jl       eligible houses at activation (simulations)
scripts/city_maps.jl               city maps and clusters of one run (simulations and figures)
scripts/plot_sweep.jl              dose-response figures (Figs. 5, 7, 8)
scripts/plot_fig4.jl               Fig. 4
scripts/plot_fig9.jl               Fig. 9
scripts/plot_fig10.jl              Fig. 10
scripts/plot_fig11.jl              Fig. 11
scripts/graphical_abstract_panel.jl   result panel of the graphical abstract
scripts/table_sensitivity.jl       Table 2
test/runtests.jl       smoke tests
env-paper/             paper environment (Julia 1.9)
results/               CSV outputs written by the scripts (git-ignored, not versioned)
figures/               figures written by the scripts
```

## Switches

The switches are fields of `ModelParams`, with matching options in `sweep_fraction.jl` and
`compare_scenarios.jl`. The defaults are the revised paper's settings.

* `public_housing_spillover` (`--spillover`, default 2.0): decrease of the potential rent of the Moore
  neighbours of a public unit. 0 switches it off; a negative value gives a positive spillover.
* `public_housing_spillover_mode` (`--mode`, default `once`): `once` applies the decrease once, at
  conversion (the paper). `every_step` applies it at every step after activation, cumulatively, down
  to a floor of `0.5 * base_potential_rent` (the first submission).
* `exclude_renovated_from_conversion` (`--exclude_renovated`, default `true`): renovated houses cannot
  be converted (the paper). In the first submission, renovated houses could be converted (`false`).
* `renovation_age_max` (`--renovation_age_max`, default 15): upper bound of the age of renovated houses.
  `Inf` reproduces the first submission.
* `median_rent_growth_trigger` (`--trigger`, default 0.5): activation threshold of the policy.
* `target_vacancy_rate` (`--vacancy`, default 0.075), `public_housing_income_threshold`
  (`--income_threshold`, default 30), `developer_investments_per_step` (`--dev_invest`, default 1),
  `developer_search_sample_size` (`--dev_sample`, default 30), `dims` (`--dims`, default 40).

The first-submission settings are `--mode=every_step --exclude_renovated=false --renovation_age_max=Inf`.
The outputs of those runs are in `results/v1/`.

## What is not included

* **Fig. 1** (thermalization schematic): no script in this repository produces it.
* **Fig. 2** (empirical validation): it compares the model with ACS rent and income series of gentrifying
  ZIP codes and of control ZIP codes in the same cities. The data are public, but the code that assembles
  them is not part of this repository. Sources:

  * **Data**: American Community Survey 5-year estimates, US Census Bureau, for each year 2011–2020, at the
    ZIP Code Tabulation Area level. Variables: median gross rent (`B25064_001E`) and median household income
    (`B19013_001E`). Access through [data.census.gov](https://data.census.gov) (table IDs B25064 and B19013) or
    the Census API (`https://api.census.gov/data/<year>/acs/acs5?get=B25064_001E,B19013_001E&for=zip%20code%20tabulation%20area:*`).
    Rents and incomes were inflation-adjusted to 2020 dollars with CPI-U and indexed to 2011.
  * **Gentrifying ZIP codes**: the list of the most gentrified US ZIP codes in Shawn M. Carter, CNBC, 6 March 2018
    ("This LA Neighborhood Is the Most Gentrified in the US—and These Other Zip Codes Made the Top 10").
  * **Control ZIP codes** (same cities, not gentrifying): *the list and its source are to be added here*; the
    repository does not yet document them.

  The Census API response format and the availability of ZCTA tables for each year should be checked against
  the current Census documentation before use.
* **Graphical abstract**: the figure is assembled in Keynote and TikZ around the result panel above. The
  Keynote file is not part of this repository.
* The notebooks of the first version of the code (`ABM-original-backup`) are not included.

## License

MIT, see `LICENSE`.
