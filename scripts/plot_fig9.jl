# Fig. 9 of the manuscript ("Rent-gap selected public housing slows down and stabilizes
# segregation during gentrification dynamics"), panels (a) neighbor segregation index N and
# (b) dissimilarity index D (low vs high income), drawn from the seed-averaged summary written
# by compare_scenarios.jl. No simulation is rerun.
#
#   julia --project scripts/plot_fig9.jl --tag=base
#
# Panels are drawn at 420x380 with large fonts, so they stay legible at the 0.48-column
# subfigure width of the manuscript. Each panel has its legend in the lower-right corner,
# where no curve runs.

include(joinpath(@__DIR__, "common.jl"))
using CairoMakie

tag = getarg(parse_args(), "tag", "base")
summ = CSV.read(joinpath(RESULTS, "compare_$(tag)_summary.csv"), DataFrame)

# Canvas is small relative to the fonts: the panel is printed at about 150 pt wide, so a
# 320 px canvas with 20 px labels reads at about 10 pt.
const SIZE = (320, 320)
const LABEL_SIZE = 20
const TICK_SIZE = 18
const LEGEND_SIZE = 16
const LINE_WIDTH = 3

const SCENARIOS = (("none", :seagreen, "No intervention"),
                   ("rentgap_0.06", :firebrick, "Rent-gap, f = 0.06"),
                   ("rentgap_0.10", :purple, "Rent-gap, f = 0.10"))

function curve!(ax, scenario, col; color, label)
    d = sort(summ[summ.scenario .== scenario, :], :step)
    m = d[!, Symbol("mean_", col)]
    s = d[!, Symbol("sem_", col)]
    band!(ax, d.step, m .- s, m .+ s; color = (color, 0.2))
    lines!(ax, d.step, m; color, label, linewidth = LINE_WIDTH)
end

function panel(ylabel, col, file)
    fig = Figure(size = SIZE)
    ax = Axis(fig[1, 1]; xlabel = "Step", ylabel,
              xlabelsize = LABEL_SIZE, ylabelsize = LABEL_SIZE,
              xticklabelsize = TICK_SIZE, yticklabelsize = TICK_SIZE)
    for (sc, color, label) in SCENARIOS
        curve!(ax, sc, col; color, label)
    end
    # lower right: the curves are all above the legend box there
    axislegend(ax; position = :rb, labelsize = LEGEND_SIZE, framevisible = true)
    save(joinpath(FIGURES, file), fig)
end

panel("Neighbor segregation", :neighbour_segregation_index, "neighbor_segregation_$(tag).pdf")
panel("Dissimilarity index", :D_low_high, "dissimilarity_index_$(tag).pdf")

println("Fig. 9 panels written to $(FIGURES) (tag=$(tag))")
