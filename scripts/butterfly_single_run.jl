# Single realization of the butterfly virtual-corridor model, rendered to a video:
#   left  — the elevation landscape with the butterflies moving over it
#   right — a cumulative "visitation" heatmap that reveals the emergent corridor
#
# The elevation field is static, so we draw it once with `heatarray`. The
# visitation panel is an Observable matrix we increment each step from the
# current butterfly positions, so the corridor builds up as the run proceeds.

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
includet(joinpath(@__DIR__, "..", "src", "ButterflyCorridor.jl"))
using .ButterflyCorridor
using Agents, CairoMakie

# --- Model parameters ---
model_params = (
    q = 0.4,                 # probability of moving uphill vs. randomly
    dims = (150, 150),
    n_butterflies = 500,
    start = (20, 20),
    start_radius = 4,
    # Two Gaussian hills; butterflies released in the low corner flow uphill.
    hills = ((110, 45, 100.0, 25.0), (55, 120, 80.0, 20.0)),
    seed = 1234,
)

model = initialize(; model_params...)
dims = model_params.dims

# --- Left panel: elevation heatmap + butterflies ---
elevation(m) = m.elevation
heatkwargs = (colormap = :terrain,)

plotkwargs = (;
    agent_color = :black,
    agent_size = 5,
    agent_marker = :circle,
    heatarray = elevation,
    heatkwargs = heatkwargs,
)

fig, ax, abmobs = abmplot(
    model;
    add_controls = false,
    figure = (size = (1150, 560),),
    axis = (title = "Landscape + butterflies (q = $(model_params.q))",),
    plotkwargs...,
)

# --- Right panel: cumulative visitation heatmap ---
visits = Observable(zeros(Float64, dims))
vax, vhm = heatmap(
    fig[1, 2],
    visits;
    colormap = :inferno,
    axis = (
        title = "Cumulative butterfly visits (emergent corridor)",
        aspect = DataAspect(),
    ),
)
Colorbar(fig[1, 3], vhm)

"""Add each butterfly's current patch to the running visitation tally."""
function tally_visits!(v, model)
    for a in allagents(model)
        v[a.pos...] += 1
    end
    return v
end

# --- Record the video ---
frames = 400
record(fig, "butterfly_corridor.mp4"; framerate = 30) do io
    for _ = 1:frames
        recordframe!(io)
        Agents.step!(abmobs, 1)
        tally_visits!(visits[], model)
        visits[] = visits[]        # notify Makie the matrix changed
        vhm.colorrange[] = (0, maximum(visits[]))
    end
    recordframe!(io)
end

println("Wrote butterfly_corridor.mp4")
