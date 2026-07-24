# Ensemble of business-investor realizations, summarized over time. For each of
# many random landscapes/placements we track three model-level quantities each
# step and plot their mean ± spread across realizations:
#   1. mean investor wealth        (do investors get richer, and how variable?)
#   2. mean *chosen* profit        (do they migrate toward higher-profit patches?)
#   3. mean *chosen* risk          (do they migrate toward lower-risk patches?)
#
# "Chosen" profit/risk = the profit/risk of the patch each investor currently
# occupies, averaged over investors — so it reports what the population selects,
# not the landscape average.

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "BusinessInvestor.jl"))
using .BusinessInvestor
using Agents, CairoMakie, Statistics

# --- Ensemble configuration ---
base_params = (
    dims = (19, 19),
    n_investors = 100,
    T = 5,
    profit_range = (1.0, 10.0),
    risk_range = (0.01, 0.1),
)
seeds = 1:100
nsteps = 25

# --- Model-level metrics collected each step ---
mean_wealth(model) = mean(a.wealth for a in allagents(model))
mean_profit(model) = mean(model.profit[a.pos...] for a in allagents(model))
mean_risk(model) = mean(model.risk[a.pos...] for a in allagents(model))
mdata = [mean_wealth, mean_profit, mean_risk]

models = [initialize(; base_params..., seed = s) for s in seeds]
_, mdf, _ = ensemblerun!(models, nsteps; mdata)

# --- Reshape the long dataframe into [time × realization] matrices ---
# ensemblerun! stacks realizations end to end (time varying fastest), so a column
# reshapes to (ntimes, nseeds). Assert the time layout to be safe.
ntimes = length(unique(mdf.time))
timemat = reshape(mdf.time, ntimes, :)
@assert all(timemat .== timemat[:, 1]) "unexpected row ordering from ensemblerun!"
times = timemat[:, 1]

bandstats(col) = (m = vec(mean(col; dims = 2)), s = vec(std(col; dims = 2)))
W = bandstats(reshape(mdf.mean_wealth, ntimes, :))
P = bandstats(reshape(mdf.mean_profit, ntimes, :))
F = bandstats(reshape(mdf.mean_risk, ntimes, :))

# --- Plot: three stacked panels sharing the time axis ---
fig = Figure(size = (760, 820))

ax1 = Axis(
    fig[1, 1];
    ylabel = "mean wealth",
    title = "Business-investor ensemble ($(length(seeds)) realizations)",
)
band!(ax1, times, W.m .- W.s, W.m .+ W.s; color = (:seagreen, 0.25))
lines!(ax1, times, W.m; color = :seagreen)

ax2 = Axis(fig[2, 1]; ylabel = "mean chosen profit")
band!(ax2, times, P.m .- P.s, P.m .+ P.s; color = (:steelblue, 0.25))
lines!(ax2, times, P.m; color = :steelblue)

ax3 = Axis(fig[3, 1]; xlabel = "step", ylabel = "mean chosen risk")
band!(ax3, times, F.m .- F.s, F.m .+ F.s; color = (:firebrick, 0.25))
lines!(ax3, times, F.m; color = :firebrick)

linkxaxes!(ax1, ax2, ax3)
save("business_investor_ensemble.png", fig)
println("Wrote business_investor_ensemble.png")
