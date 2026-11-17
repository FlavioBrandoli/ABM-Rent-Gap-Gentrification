# Fig. 10 of the manuscript ("Middle-income agents drive inner-city gentrification, not top
# earners"): pairwise dissimilarity indices between the three income tiers in the free-market
# scenario, drawn from the seed-averaged summary written by compare_scenarios.jl. No simulation
# is rerun.
#
#   julia --project scripts/plot_fig10.jl --tag=base
#
# The panel is drawn at 420x380 with large fonts, so it stays legible at the 0.9-column width
# of the manuscript. The legend is in the lower-right corner, where no curve runs.

include(joinpath(@__DIR__, "common.jl"))
using CairoMakie

tag = getarg(parse_args(), "tag", "base")
summ = CSV.read(joinpath(RESULTS, "compare_$(tag)_summary.csv"), DataFrame)

const SIZE = (420, 380)
const LABEL_SIZE = 20
const TICK_SIZE = 18
const LEGEND_SIZE = 16
const LINE_WIDTH = 3

const PAIRS = ((:D_low_middle, :purple, "low - middle"),
               (:D_middle_top, :dodgerblue3, "middle - top"),
               (:D_low_top, :darkorange2, "low - top"))

function curve!(ax, col; color, label)
    d = sort(summ[summ.scenario .== "none", :], :step)
    m = d[!, Symbol("mean_", col)]
    s = d[!, Symbol("sem_", col)]
    band!(ax, d.step, m .- s, m .+ s; color = (color, 0.2))
    lines!(ax, d.step, m; color, label, linewidth = LINE_WIDTH)
end

let fig = Figure(size = SIZE)
    ax = Axis(fig[1, 1]; xlabel = "Step", ylabel = "Pairwise dissimilarity",
              xlabelsize = LABEL_SIZE, ylabelsize = LABEL_SIZE,
              xticklabelsize = TICK_SIZE, yticklabelsize = TICK_SIZE)
    for (col, color, label) in PAIRS
        curve!(ax, col; color, label)
    end
    axislegend(ax; position = :rb, labelsize = LEGEND_SIZE, framevisible = true)
    save(joinpath(FIGURES, "dissimilarity_pair_$(tag).pdf"), fig)
end

println("Fig. 10 panel written to $(FIGURES) (tag=$(tag))")
