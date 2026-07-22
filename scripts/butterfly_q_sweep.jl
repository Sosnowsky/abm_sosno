# Corridor width vs. q, using *spatial* width estimators that correctly report a
# WIDE (or absent) corridor at low q.
#
# A naive estimator would use area / mean-path-length, where "length" is the
# number of steps taken. That misleads at low q: butterflies never reach a
# hilltop, so the step count saturates at `max_steps` and the width is pushed
# DOWN for a purely temporal reason. Here both the area and the length are
# geometric, so a diffuse low-q cloud correctly reads as wide:
#
#   1. spatial area/length = (patches visited) / (max distance reached from start)
#   2. transverse spread   = 2*sqrt(minor-axis variance of the visited patches)
#                            i.e. the lateral thickness of the trail via PCA
#
# Both use a single Gaussian hill so the corridor is unambiguous, and are
# averaged over several seeds.

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "ButterflyCorridor.jl"))
using .ButterflyCorridor
using Agents, CairoMakie, Statistics, LinearAlgebra

# A patch is a hilltop if no neighbour is strictly higher (local maximum).
function is_hilltop(pos, model)
    e = model.elevation
    here = e[pos...]
    for n in nearby_positions(pos, model, 1)
        e[n...] > here && return false
    end
    return true
end

"""Walk all butterflies (freezing on hilltops) and return the visited bitmask."""
function walk_visited(; q, seed, base_params, max_steps)
    model = initialize(; base_params..., q = q, seed = seed)
    dims = size(model.elevation)
    ids = collect(allids(model))
    top = maximum(ids)
    arrived = falses(top)
    visited = falses(dims)
    for id in ids
        visited[model[id].pos...] = true
    end
    for _ = 1:max_steps
        all(@view arrived[ids]) && break
        for id in ids
            arrived[id] && continue
            a = model[id]
            if is_hilltop(a.pos, model)
                arrived[id] = true
                continue
            end
            butterfly_step!(a, model)
            visited[a.pos...] = true
        end
    end
    return visited
end

"""
    width_metrics(visited, start) -> (; spatial, transverse)

Two geometric corridor-width estimates from the set of visited patches:
- `spatial`:    area / max radial distance from `start`
- `transverse`: 2*sqrt(minor eigenvalue) of the visited-patch coordinate spread
"""
function width_metrics(visited, start)
    coords = findall(visited)
    n = length(coords)
    xs = Float64[c[1] for c in coords]
    ys = Float64[c[2] for c in coords]

    maxr = maximum(hypot(x - start[1], y - start[2]) for (x, y) in zip(xs, ys))
    spatial = n / maxr

    mx, my = mean(xs), mean(ys)
    dx, dy = xs .- mx, ys .- my
    C = Symmetric([mean(dx .^ 2) mean(dx .* dy); mean(dx .* dy) mean(dy .^ 2)])
    λmin = minimum(eigvals(C))
    transverse = 2 * sqrt(max(λmin, 0.0))

    return (; spatial, transverse)
end

# --- Sweep configuration (same landscape/start as the temporal sweep) ---
base_params = (
    dims = (150, 150),
    n_butterflies = 400,
    start = (25, 25),
    start_radius = 0,
    hills = ((110, 110, 100.0, 30.0),),
)
qs = 0.0:0.1:1.0
seeds = 1:5
max_steps = 1500

spatial_m = Float64[]
transverse_m = Float64[]
for q in qs
    res = [
        width_metrics(
            walk_visited(;
                q = q,
                seed = s,
                base_params = base_params,
                max_steps = max_steps,
            ),
            base_params.start,
        ) for s in seeds
    ]
    push!(spatial_m, mean(r.spatial for r in res))
    push!(transverse_m, mean(r.transverse for r in res))
    println(
        "q = $(round(q, digits=2))  ->  spatial = $(round(spatial_m[end], digits=1))   transverse = $(round(transverse_m[end], digits=1))",
    )
end

# --- Plot both spatial estimators vs q ---
fig = Figure(size = (740, 470))
ax = Axis(
    fig[1, 1];
    xlabel = "q  (probability of moving uphill)",
    ylabel = "corridor width  (patches)",
    title = "Spatial corridor-width estimators (high at low q)",
)
s = scatterlines!(ax, collect(qs), spatial_m; color = :seagreen, markersize = 10)
t = scatterlines!(ax, collect(qs), transverse_m; color = :purple, markersize = 10)
axislegend(ax, [s, t], ["area / radial reach", "transverse spread (PCA)"]; position = :rt)
save("butterfly_corridor_width_spatial.png", fig)
println("Wrote butterfly_corridor_width_spatial.png")
