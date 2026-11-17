# Fig. 11 of the manuscript ("Decomposed Theil entropy reveals segregation paradox"): total
# Theil entropy and its between-group and within-group (inner city, suburbs) components in the
# free-market scenario, drawn from the seed-averaged summary written by compare_scenarios.jl.
# No simulation is rerun.
#
#   julia --project scripts/plot_fig11.jl --tag=base
#
# The panel is drawn at 420x380 with large fonts. The legend is at the upper centre, in the
# band above the curves between about step 55 and step 120, where no curve runs.

include(joinpath(@__DIR__, "common.jl"))
using CairoMakie

tag = getarg(parse_args(), "tag", "base")
summ = CSV.read(joinpath(RESULTS, "compare_$(tag)_summary.csv"), DataFrame)

const SIZE = (420, 380)
const LABEL_SIZE = 20
const TICK_SIZE = 18
const LEGEND_SIZE = 16
const LINE_WIDTH = 3

const COMPONENTS = ((:H_total, :black, "H total"),
                    (:H_between, :firebrick, "H between"),
                    (:H_within_inner, :dodgerblue3, "H inner city"),
                    (:H_within_suburbs, :seagreen, "H suburbs"))

function curve!(ax, col; color, label)
    d = sort(summ[summ.scenario .== "none", :], :step)
    m = d[!, Symbol("mean_", col)]
    s = d[!, Symbol("sem_", col)]
    band!(ax, d.step, m .- s, m .+ s; color = (color, 0.2))
    lines!(ax, d.step, m; color, label, linewidth = LINE_WIDTH)
end

let fig = Figure(size = SIZE)
    ax = Axis(fig[1, 1]; xlabel = "Step", ylabel = "Theil index",
              xlabelsize = LABEL_SIZE, ylabelsize = LABEL_SIZE,
              xticklabelsize = TICK_SIZE, yticklabelsize = TICK_SIZE)
    for (col, color, label) in COMPONENTS
        curve!(ax, col; color, label)
    end
    # headroom above the curves (max 0.075) for the legend, which sits in two columns at the top centre
    ylims!(ax; high = 0.10)
    axislegend(ax; position = :ct, nbanks = 2, labelsize = LEGEND_SIZE, framevisible = true)
    save(joinpath(FIGURES, "theil_$(tag).pdf"), fig)
end

println("Fig. 11 panel written to $(FIGURES) (tag=$(tag))")
