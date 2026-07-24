# Single realization of the business-investor model, rendered to a video:
#   left  — the business landscape (profit heatmap) with investors moving over it
#   right — mean investor wealth over time
#
# The profit/risk landscape is static, so we draw profit once with `heatarray`.
# Mean wealth is an Observable we append to each step from the current investors,
# so the curve builds up as the run proceeds.

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
includet(joinpath(@__DIR__, "..", "src", "BusinessInvestor.jl"))
using .BusinessInvestor
using Agents, CairoMakie, Statistics

# --- Model parameters ---
model_params = (
    dims = (19, 19),
    n_investors = 100,
    T = 5,
    sensing_radius = 1,
    profit_range = (1.0, 10.0),
    risk_range = (0.01, 0.1),
    seed = 1234,
)

model = initialize(; model_params...)

# --- Left panel: profit heatmap + investors ---
profitmap(m) = m.profit
plotkwargs = (;
    agent_color = :white,
    agent_size = 11,
    agent_marker = :circle,
    agentsplotkwargs = (strokewidth = 1.0, strokecolor = :black),
    heatarray = profitmap,
    heatkwargs = (colormap = :viridis,),
)

fig, ax, abmobs = abmplot(
    model;
    add_controls = false,
    figure = (size = (1050, 520),),
    axis = (title = "Business landscape (profit) + investors",),
    plotkwargs...,
)

# --- Right panel: mean wealth over time ---
current_mean_wealth(m) = mean(a.wealth for a in allagents(m))
steps = Observable(Int[0])
mean_wealth = Observable(Float64[current_mean_wealth(model)])
wax = Axis(
    fig[1, 2];
    xlabel = "step",
    ylabel = "mean wealth",
    title = "Mean investor wealth over time",
)
lines!(wax, steps, mean_wealth; color = :seagreen)

# --- Record the video ---
frames = 25
record(fig, "business_investor.mp4"; framerate = 3) do io
    for s = 1:frames
        recordframe!(io)
        Agents.step!(abmobs, 1)                        # reposition + accounting
        push!(steps[], s)
        push!(mean_wealth[], current_mean_wealth(model))
        steps[] = steps[]                              # notify Makie of the change
        mean_wealth[] = mean_wealth[]
        autolimits!(wax)
    end
    recordframe!(io)
end

println("Wrote business_investor.mp4")
