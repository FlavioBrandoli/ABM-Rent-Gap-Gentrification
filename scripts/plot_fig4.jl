# Fig. 4 of the manuscript ("Public housing intervention stabilizes inner-city
# demographics"), panels (a) inner-city median income and (b) agent counts,
# drawn from the seed-averaged summary written by compare_scenarios.jl.
# No simulation is rerun: the figure reads results/compare_<tag>_summary.csv.
#
#   julia --project scripts/plot_fig4.jl --tag=base
#
# Layout: panels are drawn at 420x380 pt with large fonts, so that they stay
# legible when scaled to the 0.48-column subfigures of the manuscript. Panel (a)
# carries its legend in the empty lower-right corner; panel (b) has no legend
# (the line styles are described in the caption).

include(joinpath(@__DIR__, "common.jl"))
using CairoMakie

tag = getarg(parse_args(), "tag", "base")
summ = CSV.read(joinpath(RESULTS, "compare_$(tag)_summary.csv"), DataFrame)
const N_THERM = 15

const SIZE = (420, 380)
const LABEL_SIZE = 20
const TICK_SIZE = 18
const LEGEND_SIZE = 18
const LINE_WIDTH = 3

const COL_NONE = :seagreen
const COL_RANDOM = :dodgerblue3
const COL_RENTGAP = :darkorange2
const COL_LOW = :dodgerblue3
const COL_HIGH = :firebrick

function curve!(ax, scenario, col; color, label = nothing, linestyle = :solid)
    d = sort(summ[summ.scenario .== scenario, :], :step)
    m = d[!, Symbol("mean_", col)]
    s = d[!, Symbol("sem_", col)]
    band!(ax, d.step, m .- s, m .+ s; color = (color, 0.2))
    lines!(ax, d.step, m; color, linestyle, linewidth = LINE_WIDTH, label)
end

function styled_axis(pos, ylabel)
    return Axis(pos; xlabel = "Step", ylabel,
                xlabelsize = LABEL_SIZE, ylabelsize = LABEL_SIZE,
                xticklabelsize = TICK_SIZE, yticklabelsize = TICK_SIZE)
end

# ---- (a) median income -------------------------------------------------------
let fig = Figure(size = SIZE)
    ax = styled_axis(fig[1, 1], "Inner-city median income")
    curve!(ax, "none", :median_income_inner_city; color = COL_NONE, label = "No intervention")
    curve!(ax, "random", :median_income_inner_city; color = COL_RANDOM, label = "Random PH (f=0.07)")
    curve!(ax, "rentgap", :median_income_inner_city; color = COL_RENTGAP, label = "Rent-gap PH (f=0.07)")
    vlines!(ax, [N_THERM]; color = :black, linestyle = :dash, linewidth = 2)
    axislegend(ax; position = :rb, labelsize = LEGEND_SIZE, framevisible = true)
    save(joinpath(FIGURES, "median_income_comparison_$(tag).pdf"), fig)
end

# ---- (b) agents: low income (blue), high income (red); line style = scenario ----
let fig = Figure(size = SIZE)
    ax = styled_axis(fig[1, 1], "Agents in the inner city")
    for (sc, ls) in (("none", :solid), ("random", :dash), ("rentgap", :dot))
        curve!(ax, sc, :n_low_inner; color = COL_LOW, linestyle = ls)
        curve!(ax, sc, :n_high_inner; color = COL_HIGH, linestyle = ls)
    end
    vlines!(ax, [N_THERM]; color = :black, linestyle = :dash, linewidth = 2)
    save(joinpath(FIGURES, "number_agents_$(tag).pdf"), fig)
end

println("Fig. 4 panels written to $(FIGURES) (tag=$(tag))")
