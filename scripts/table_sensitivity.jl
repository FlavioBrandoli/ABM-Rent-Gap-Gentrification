# Table 2 of the manuscript: saturation point f* of the rent-gap strategy, its minimum final
# inner-city median income, and the income at f = 0.06, for each sensitivity setting.
# Computed from the saved sweep summaries (results/sweep_<tag>_summary.csv) and candidate
# profiles (results/candidate_profile_<tag>.csv); no simulation is run.
#
#   julia --project scripts/table_sensitivity.jl
#
# Definitions:
#   eligible pool  share of the stock that is in the inner city and eligible for conversion
#                  at activation (mean over seeds). The spillover settings share the baseline pool.
#   f*             smallest f at which the rent-gap income is within one SEM of its minimum over
#                  0 <= f <= 0.15; ">0.15" if the minimum is at f = 0.15; "n/a" if the whole rent-gap
#                  curve is within one SEM of its minimum (no saturation can be identified).
#   minimum        the minimum over f of the rent-gap mean income (the value of the curve at its
#                  lowest point, which is what the manuscript reports in the column "income at f*").
#   free market    mean income at f = 0 (random strategy, zero public housing).
#   rent-gap, random at f = 0.06: mean income at f = 0.06 for each strategy.

include(joinpath(@__DIR__, "common.jl"))

# (label, sweep tag, candidate-profile tag for the eligible pool)
SETTINGS = [
    ("baseline (\$\\Delta PR^- = 2\$, once)", "base", "base"),
    ("no public-housing spillover", "spill0", "base"),
    ("spillover doubled (\$\\Delta PR^- = 4\$)", "spill4", "base"),
    ("activation at 25% rent growth", "trig025", "trig025"),
    ("activation at 75% rent growth", "trig075", "trig075"),
    ("vacancy 5%", "vac005", "vac005"),
    ("vacancy 10%", "vac010", "vac010"),
    ("eligibility 20k USD", "inc20", "inc20"),
    ("eligibility 45k USD", "inc45", "inc45"),
    ("60x60 grid (400 steps, developer capacity x2.25)", "grid60", "grid60"),
]

"Fractions, mean and SEM of the final income, for one policy (sorted by fraction)."
function curve(summary, policy)
    d = sort(summary[summary.policy .== policy, :], :ph_fraction)
    return d.ph_fraction, d.mean_final_income, d.sem_final_income
end

"Mean over seeds of (eligible houses in the inner city) / (houses)."
function eligible_pool(tag)
    cp = CSV.read(joinpath(RESULTS, "candidate_profile_$(tag).csv"), DataFrame)
    return mean(count(g.inner) / first(g.n_houses) for g in groupby(cp, :seed))
end

"Value of v at the fraction x0 (fractions in f)."
value_at(x0, f, v) = v[findfirst(x -> isapprox(x, x0; atol = 1e-9), f)]

function row(label, tag, pool_tag)
    s = CSV.read(joinpath(RESULTS, "sweep_$(tag)_summary.csv"), DataFrame)
    f, m, e = curve(s, "rentgap")
    _, mr, _ = curve(s, "random")
    imin = argmin(m)
    fstar = if m[1] - m[imin] <= e[imin]
        "n/a"
    elseif imin == length(m)
        "> 0.15"
    else
        fidx = findfirst(i -> m[i] <= m[imin] + e[imin], eachindex(m))
        @sprintf("%.2f", f[fidx])
    end
    return (setting = label, pool = @sprintf("%.1f%%", 100 * eligible_pool(pool_tag)),
            fstar = fstar, minimum = @sprintf("%.1f", m[imin]),
            free_market = @sprintf("%.1f", mr[1]),
            rentgap_006 = @sprintf("%.1f", value_at(0.06, f, m)),
            random_006 = @sprintf("%.1f", value_at(0.06, f, mr)))
end

tbl = DataFrame([row(l, t, p) for (l, t, p) in SETTINGS])
rename!(tbl, :setting => "Setting", :pool => "eligible inner-city pool", :fstar => "f*",
        :minimum => "minimum income", :free_market => "free market",
        :rentgap_006 => "rent-gap, f=0.06", :random_006 => "random, f=0.06")
show(stdout, tbl, allrows = true, allcols = true); println()
CSV.write(joinpath(RESULTS, "table2_sensitivity.csv"), tbl)
println("saved results/table2_sensitivity.csv")
