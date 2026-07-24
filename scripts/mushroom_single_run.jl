# Single realization of the mushroom-hunt model, rendered to a video:
# ABM space (left) + cumulative-mushrooms-eaten-by-strategy panel (right).
#
# `abmvideo` only renders the space, so we build the figure ourselves with
# `abmplot(...; adata)` (which keeps `abmobs.adf` updated each step) and drive
# it with Makie's `record`.

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
includet(joinpath(@__DIR__, "..", "src", "MushroomHunt.jl"))
using .MushroomHunt
using Agents, CairoMakie

# Model parameters
model_params = (
    seed = 1234,
    dims = (100, 100),
    speed = 1,
    dt = 1,
    sight_radius = 1,
    max_rotation_searching = π / 16,
    max_rotation_foraging = π / 2,
    n_clusters = 4,
    n_mushrooms_per_cluster = 100,
    time_strategies = (0, 50),
    cluster_spread = 2,
)


# --- Plotting appearance ---
ashape(a) = a isa Forager ? :utriangle : :circle
foragercol(a) = a.id == 1 ? :yellow : :blue
acolor(a) = a isa Forager ? foragercol(a) : :brown

plotkwargs = (;
    agent_color = acolor,
    agent_size = 25,
    agent_marker = ashape,
    agentsplotkwargs = (strokewidth = 1.0, strokecolor = :black),
)

# --- Data to track live: cumulative mushrooms eaten per strategy ---
eaten(a) = a.eaten_mushrooms
fast(a) = a isa Forager && a.time_strategy == 0
slow(a) = a isa Forager && a.time_strategy == 50
adata = [(eaten, sum, fast), (eaten, sum, slow)]

# --- Build the model + interactive plot ---
vmodel = initialize(; model_params...)
vfig, vax, vabmobs = abmplot(
    vmodel;
    adata,
    add_controls = false,
    figure = (size = (1100, 560),),
    plotkwargs...,
)

# Vision circles that follow the foragers (recomputed each step via `lift`).
vision = lift(vabmobs.model) do m
    [Circle(Point2f(a.pos), m.sight_radius) for a in allagents(m) if a isa Forager]
end
poly!(vax, vision; color = (:yellow, 0.1), strokecolor = :blue, strokewidth = 1)

# Right-hand data panel tracking cumulative mushrooms eaten, per strategy.
# Column names come from `dataname`: "sum_eaten_fast" / "sum_eaten_slow".
data_ax = Axis(vfig[1, 2]; xlabel = "time", ylabel = "mushrooms eaten")
fast_curve = @lift(Point2f.($(vabmobs.adf).time, $(vabmobs.adf)[:, "sum_eaten_fast"]))
slow_curve = @lift(Point2f.($(vabmobs.adf).time, $(vabmobs.adf)[:, "sum_eaten_slow"]))
scatterlines!(data_ax, fast_curve; color = :gold, label = "fast (time_strategy=0)")
scatterlines!(data_ax, slow_curve; color = :blue, label = "slow (time_strategy=50)")
axislegend(data_ax; position = :lt)

# --- Record the video ---
frames = 1000
record(vfig, "mushroom_hunt.mp4"; framerate = 50) do io
    for _ = 1:frames
        recordframe!(io)
        Agents.step!(vabmobs, 1)
        autolimits!(data_ax)   # keep the growing curves in view
    end
    recordframe!(io)
end
