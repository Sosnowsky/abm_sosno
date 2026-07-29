# Ensemble of business-investor realizations, comparing several sensing radii.
# For each `sensing_radius` we average three model-level quantities over many
# random landscapes/placements and plot one curve per radius:
#   1. mean investor wealth        (do investors get richer, and how fast?)
#   2. mean *chosen* profit        (do they settle on higher-profit patches?)
#   3. mean *chosen* risk          (do they settle on lower-risk patches?)
#
# "Chosen" profit/risk = the profit/risk of the patch each investor currently
# occupies, averaged over investors — what the population selects, not the
# landscape average. A larger sensing radius lets an investor jump directly to the
# best business within that Chebyshev distance; radius 0 disables repositioning.

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "BusinessInvestor.jl"))
using .BusinessInvestor
using Agents, CairoMakie, Statistics

# --- Ensemble configuration ---
base_params = (
    dims = (19, 19),
    n_investors = 10,
    T = 5,
    profit_range = (-5.0, 10.0),
    risk_range = (0.01, 0.1),
)
radii = [0, 1, 2, 3, 5]
seeds = 1:100
nsteps = 25

# --- Model-level metrics collected each step ---
mean_wealth(model) = mean(a.wealth for a in allagents(model))
mean_profit(model) = mean(model.profit[a.pos...] for a in allagents(model))
mean_risk(model) = mean(model.risk[a.pos...] for a in allagents(model))
mdata = [mean_wealth, mean_profit, mean_risk]

"""Run the seed ensemble for one sensing radius; return the per-step means."""
function run_ensemble(radius)
    models = [initialize(; base_params..., sensing_radius = radius, seed = s) for s in seeds]
    _, mdf, _ = ensemblerun!(models, nsteps; mdata)
    # ensemblerun! stacks realizations end to end (time varying fastest), so a
    # column reshapes to (ntimes, nseeds). Assert the layout to be safe.
    ntimes = length(unique(mdf.time))
    timemat = reshape(mdf.time, ntimes, :)
    @assert all(timemat .== timemat[:, 1]) "unexpected row ordering from ensemblerun!"
    seedmean(col) = vec(mean(reshape(col, ntimes, :); dims = 2))
    return (;
        times = timemat[:, 1],
        wealth = seedmean(mdf.mean_wealth),
        profit = seedmean(mdf.mean_profit),
        risk = seedmean(mdf.mean_risk),
    )
end

results = [run_ensemble(r) for r in radii]
times = results[1].times
colors = cgrad(:viridis, length(radii); categorical = true)

# --- Plot: three stacked panels sharing the time axis, one curve per radius ---
fig = Figure(size = (780, 900))
ax1 = Axis(
    fig[1, 1];
    ylabel = "mean wealth",
    title = "Business-investor ensemble ($(length(seeds)) realizations)",
)
ax2 = Axis(fig[2, 1]; ylabel = "mean chosen profit")
ax3 = Axis(fig[3, 1]; xlabel = "step", ylabel = "mean chosen risk")

for (i, r) in enumerate(radii)
    res = results[i]
    lines!(ax1, res.times, res.wealth; color = colors[i], label = "r = $r")
    lines!(ax2, res.times, res.profit; color = colors[i])
    lines!(ax3, res.times, res.risk; color = colors[i])
end

axislegend(ax1, "sensing radius"; position = :lt)
linkxaxes!(ax1, ax2, ax3)
save("business_investor_ensemble.png", fig)
println("Wrote business_investor_ensemble.png")
