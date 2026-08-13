# Single realization of the rock–paper–scissors learning model, rendered to a
# video of the *strategy simplex* — the only space this model has.
#
# Two panels, same seed and same learning rate, differing only in the belief
# memory λ:
#   left  — λ = 1, memoryless: the belief is just the last match's three throws,
#           so the best-response target flickers and the strategies random-walk
#           around Nash.
#   right — λ = 0.2: the belief averages several matches, the target persists,
#           and the pair locks into a clean leapfrogging chase around the centre.
#
# Each player's strategy is drawn in barycentric coordinates with a fading trail;
# the hollow diamond is that player's running *time average*, which converges to
# Nash even though the instantaneous strategy never does.

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "RockPaperScissors.jl"))
using .RockPaperScissors
using Agents, CairoMakie, Statistics

# --- Parameters ---
T = 600                       # matches (= frames)
trail_length = 90             # how many past strategies stay visible
shared = (learning_rate = 0.12, rounds_per_match = 3, seed = 20240812)
runs = (
    (label = "memoryless  (λ = 1)", memory = 1.0),
    (label = "remembering  (λ = 0.2)", memory = 0.2),
)

player_colors = (RGBf(0.85, 0.30, 0.20), RGBf(0.15, 0.35, 0.62))
player_labels = ("player 1", "player 2")

# --- Simplex drawing helpers ---
pt(x) = Point2f(ternary(x)...)

"""Draw the empty simplex: edges, vertex labels, and the Nash point."""
function draw_simplex!(ax)
    corners = [pt(v) for v in VERTEX]
    poly!(ax, corners; color = RGBAf(0.96, 0.96, 0.97, 1.0))
    lines!(ax, push!(copy(corners), corners[1]); color = :grey30, linewidth = 1.5)

    # Vertex labels, nudged outward from the centre of the triangle.
    centre = pt(NASH)
    for (i, c) in enumerate(corners)
        d = (c - centre) / 0.6
        text!(
            ax,
            c + Point2f(0.09 * d[1], 0.09 * d[2]);
            text = ACTION_NAMES[i],
            align = (:center, :center),
            fontsize = 20,
            color = :grey20,
        )
    end

    scatter!(ax, [centre]; marker = :xcross, markersize = 13, color = :grey40)
    text!(
        ax,
        centre + Point2f(0.0, -0.055);
        text = "Nash",
        align = (:center, :top),
        fontsize = 11,
        color = :grey40,
    )
    hidedecorations!(ax)
    hidespines!(ax)
    return ax
end

# --- Figure ---
fig = Figure(size = (1500, 620))

axes_simplex = Axis[]
trails = Observable{Vector{Point2f}}[]
heads = Observable{Vector{Point2f}}[]
averages = Observable{Vector{Point2f}}[]
models = StandardABM[]
histories = Vector{Vector{SVector{3,Float64}}}[]   # [run][player] -> strategies

for (k, run) in enumerate(runs)
    ax = Axis(
        fig[1, k];
        aspect = DataAspect(),
        title = run.label,
        titlesize = 18,
        limits = (-0.14, 1.14, -0.14, 1.02),
    )
    draw_simplex!(ax)
    push!(axes_simplex, ax)

    model = initialize(; shared..., memory = run.memory)
    push!(models, model)
    push!(histories, [[model[j].strategy] for j = 1:2])

    # One trail per player. Fixed length (pre-filled with the starting point) so
    # the per-vertex colour vector never has to be resized.
    for j = 1:2
        start = pt(model[j].strategy)
        trail = Observable(fill(start, trail_length))
        head = Observable([start])
        avg = Observable([start])
        c = player_colors[j]
        lines!(
            ax,
            trail;
            color = range(0, 1; length = trail_length),
            colormap = cgrad([RGBAf(c.r, c.g, c.b, 0.0), RGBAf(c.r, c.g, c.b, 0.9)]),
            linewidth = 2.5,
        )
        scatter!(ax, head; color = c, markersize = 15, strokecolor = :white, strokewidth = 1)
        scatter!(
            ax,
            avg;
            color = :transparent,
            marker = :diamond,
            markersize = 15,
            strokecolor = c,
            strokewidth = 2,
        )
        push!(trails, trail)
        push!(heads, head)
        push!(averages, avg)
    end
end

# Legend, drawn once on the left panel.
legend_entries = [
    MarkerElement(color = player_colors[1], marker = :circle, markersize = 13),
    MarkerElement(color = player_colors[2], marker = :circle, markersize = 13),
    MarkerElement(
        color = :transparent,
        marker = :diamond,
        markersize = 13,
        strokecolor = :grey30,
        strokewidth = 2,
    ),
]
axislegend(
    axes_simplex[1],
    legend_entries,
    ["player 1", "player 2", "time average"];
    position = :lt,
    framevisible = false,
    labelsize = 12,
)

# --- Right panel: distance to Nash over time (player 1 of each run) ---
ax_d = Axis(
    fig[1, 3];
    title = "player 1: distance to Nash",
    xlabel = "match",
    ylabel = "‖x − x*‖",
    limits = (0, T, 0, 0.85),
)
dist_lines = [Observable(Point2f[]) for _ in runs]
for (k, run) in enumerate(runs)
    lines!(
        ax_d,
        dist_lines[k];
        color = k == 1 ? RGBf(0.55, 0.55, 0.58) : RGBf(0.15, 0.35, 0.62),
        linewidth = 1.6,
        label = run.label,
    )
end
hlines!(ax_d, [0.0]; color = :grey60, linestyle = :dash)
axislegend(ax_d; position = :lt, framevisible = false, labelsize = 11)
colsize!(fig.layout, 3, Relative(0.32))

Label(
    fig[0, 1:3],
    "Rock–paper–scissors: two best-response learners in the strategy simplex " *
    "(η = $(shared.learning_rate), K = $(shared.rounds_per_match))";
    fontsize = 21,
    padding = (0, 0, 6, 0),
)

# --- Record ---
record(fig, "rock_paper_scissors.mp4"; framerate = 30) do io
    for t = 1:T
        recordframe!(io)
        for (k, model) in enumerate(models)
            Agents.step!(model, 1)
            for j = 1:2
                x = model[j].strategy
                push!(histories[k][j], x)

                idx = 2(k - 1) + j
                trail = trails[idx][]
                popfirst!(trail)
                push!(trail, pt(x))
                trails[idx][] = trail
                heads[idx][] = [pt(x)]
                averages[idx][] = [pt(mean(histories[k][j]))]
            end
            push!(dist_lines[k][], Point2f(t, distance_to_nash(model[1].strategy)))
            dist_lines[k][] = dist_lines[k][]
        end
    end
    recordframe!(io)
end

println("Wrote rock_paper_scissors.mp4")

# --- Summary of what the run did, checked against the spec's predictions ---
for (k, run) in enumerate(runs)
    second_half = (T ÷ 2):(T+1)
    radius = mean(distance_to_nash.(histories[k][1][second_half]))
    winding = winding_rate(histories[k][1][second_half])
    avg = mean(histories[k][1])
    println(
        rpad(run.label, 24),
        " orbit radius = ", round(radius, digits = 3),
        " | rotation = ", round(winding, digits = 4), " rad/match",
        " (", round(Int, 2π / abs(winding)), " matches per loop)",
        " | ‖time-average − Nash‖ = ", round(distance_to_nash(avg), digits = 4),
        " | score = ", models[k][1].score,
    )
end
