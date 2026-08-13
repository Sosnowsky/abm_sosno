# Ensemble sweeps for the rock–paper–scissors learning model, testing the two
# quantitative claims in `models/rock_paper_scissors.md`:
#
#   1. The learning rate η sets the *size and speed* of the orbit: bigger steps
#      toward the best reply throw the strategies further from Nash and carry
#      them around it faster.
#   2. The throws per match K is an *observation-noise* knob, not a strategic
#      one — it only controls how finely one match estimates the opponent's mix.
#      So K and the belief memory λ should be substitutes: more throws per match
#      and more matches per belief are two ways of buying the same averaging.
#
# Everything is measured over the second half of each run (after transients) and
# averaged over both players and many seeds. Two summaries per run:
#   orbit radius   — mean ‖x − x*‖, how far from equilibrium play sits.
#   rotation rate  — mean angular velocity about Nash, rad/match. Negative means
#                    clockwise in the ternary plane, i.e. each player's strategy
#                    cycling R → S → P.

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "RockPaperScissors.jl"))
using .RockPaperScissors
using Agents, CairoMakie, Statistics

# --- Sweep configuration ---
T = 4000                       # matches per run
n_seeds = 24
etas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.6, 1.0]
rounds = [1, 3, 9, 27, 81]
memories = [(λ = 1.0, label = "memoryless  (λ = 1)"), (λ = 0.2, label = "remembering  (λ = 0.2)")]

const DEFAULT_ETA = 0.1
const DEFAULT_K = 3

"""
    measure(; kwargs...) -> (radius, rotation)

Run one realization and summarise its second half, averaged over both players.
"""
function measure(; T = T, kwargs...)
    model = initialize(; kwargs...)
    xs = trajectory(model, T)
    tail = (T ÷ 2 + 1):(T + 1)
    radius = mean(distance_to_nash(xs[t, j]) for t in tail, j = 1:2)
    rotation = mean(winding_rate(view(xs, tail, j)) for j = 1:2)
    return (radius = radius, rotation = rotation)
end

"""Ensemble over seeds, returning the mean and standard deviation of each
summary. The standard deviation across seeds is what says whether the rotation
is a real drift or just noise that happened to sum in one direction."""
function ensemble(; kwargs...)
    results = [measure(; seed = 1000 + s, kwargs...) for s = 1:n_seeds]
    r = [x.radius for x in results]
    w = [x.rotation for x in results]
    return (
        radius = mean(r),
        radius_sd = std(r),
        rotation = mean(w),
        rotation_sd = std(w),
    )
end

println("Sweeping η over $(length(etas)) values × $(length(memories)) memories × $n_seeds seeds ...")
eta_sweep = [[ensemble(; learning_rate = η, memory = m.λ, rounds_per_match = DEFAULT_K)
              for η in etas] for m in memories]

println("Sweeping K over $(length(rounds)) values ...")
k_sweep = [[ensemble(; learning_rate = DEFAULT_ETA, memory = m.λ, rounds_per_match = K)
            for K in rounds] for m in memories]

# --- Plot ---
set_theme!(fontsize = 13)
fig = Figure(size = (1150, 780))
colors = (RGBf(0.55, 0.55, 0.58), RGBf(0.15, 0.35, 0.62))
markers = (:rect, :circle)

"""Mean ± sd of one field, plotted against `xs` for both memory settings."""
function plot_series!(ax, xs, sweeps, field, sdfield)
    for (k, m) in enumerate(memories)
        ys = [getfield(s, field) for s in sweeps[k]]
        es = [getfield(s, sdfield) for s in sweeps[k]]
        errorbars!(ax, xs, ys, es; color = colors[k], whiskerwidth = 8)
        scatterlines!(
            ax, xs, ys;
            color = colors[k], markercolor = colors[k], marker = markers[k],
            markersize = 10, linewidth = 2, label = m.label,
        )
    end
end

# (1,1) orbit radius vs η
ax11 = Axis(
    fig[1, 1];
    xscale = log10, xlabel = "learning rate η", ylabel = "orbit radius  ⟨‖x − x*‖⟩",
    title = "Bigger steps ⇒ play further from equilibrium",
)
plot_series!(ax11, etas, eta_sweep, :radius, :radius_sd)
# A pure strategy sits this far from Nash: the ceiling the orbit can reach.
hlines!(ax11, [distance_to_nash([1, 0, 0])]; color = :firebrick, linestyle = :dash)
text!(
    ax11, 0.9, distance_to_nash([1, 0, 0]) - 0.015;
    text = "pure strategy", align = (:right, :top), color = :firebrick, fontsize = 11,
)
axislegend(ax11; position = :lt, framevisible = false, labelsize = 11)

# (1,2) rotation vs η. η = 1 is dropped here: the strategy then jumps straight to
# a vertex every match, so a +240° step is indistinguishable from a −120° one and
# the angle measurement aliases (its across-seed spread explodes accordingly).
ax12 = Axis(
    fig[1, 2];
    xscale = log10, yscale = log10,
    xlabel = "learning rate η", ylabel = "|rotation rate|  (rad / match)",
    title = "…and carry it around Nash faster",
)
unaliased = findall(<(1.0), etas)
for (k, m) in enumerate(memories)
    ys = abs.([s.rotation for s in eta_sweep[k][unaliased]])
    scatterlines!(
        ax12, etas[unaliased], ys;
        color = colors[k], markercolor = colors[k], marker = markers[k],
        markersize = 10, linewidth = 2, label = m.label,
    )
end
lines!(ax12, etas[unaliased], 0.7 .* etas[unaliased]; color = :firebrick, linestyle = :dash)
text!(
    ax12, 0.1, 0.7 * 0.1 / 1.6;
    text = "slope 1", color = :firebrick, fontsize = 11, align = (:center, :top),
)
axislegend(ax12; position = :lt, framevisible = false, labelsize = 11)

# (2,1) orbit radius vs K
ax21 = Axis(
    fig[2, 1];
    xscale = log10, xlabel = "throws per match K", ylabel = "orbit radius  ⟨‖x − x*‖⟩",
    title = "K barely moves the orbit's size…",
    xticks = (rounds, string.(rounds)),
    limits = (nothing, (0.0, 0.5)),
)
plot_series!(ax21, rounds, k_sweep, :radius, :radius_sd)
axislegend(ax21; position = :rc, framevisible = false, labelsize = 11)

# (2,2) rotation vs K
ax22 = Axis(
    fig[2, 2];
    xscale = log10,
    xlabel = "throws per match K", ylabel = "rotation rate  (rad / match)",
    title = "…but with no memory it sets whether the chase is coherent at all",
    xticks = (rounds, string.(rounds)),
)
plot_series!(ax22, rounds, k_sweep, :rotation, :rotation_sd)
hlines!(ax22, [0.0]; color = :grey60, linestyle = :dash)
axislegend(ax22; position = :lb, framevisible = false, labelsize = 11)

Label(
    fig[0, 1:2],
    "Rock–paper–scissors learners: what the learning rate and the match length do " *
    "($n_seeds seeds, $T matches, second half measured)";
    fontsize = 17,
)

save("rps_learning_rate_sweep.png", fig)
set_theme!()
println("Wrote rps_learning_rate_sweep.png")

# --- Numbers behind the figure ---
println("\nη sweep (K = $DEFAULT_K):")
for (k, m) in enumerate(memories)
    println("  ", m.label)
    for (i, η) in enumerate(etas)
        s = eta_sweep[k][i]
        println(
            "    η = ", rpad(η, 5),
            " radius = ", rpad(round(s.radius, digits = 3), 6),
            " rotation = ", rpad(round(s.rotation, digits = 4), 8),
            " ± ", round(s.rotation_sd, digits = 4),
        )
    end
end
println("\nK sweep (η = $DEFAULT_ETA):")
for (k, m) in enumerate(memories)
    println("  ", m.label)
    for (i, K) in enumerate(rounds)
        s = k_sweep[k][i]
        println(
            "    K = ", rpad(K, 3),
            " radius = ", rpad(round(s.radius, digits = 3), 6),
            " rotation = ", rpad(round(s.rotation, digits = 4), 8),
            " ± ", round(s.rotation_sd, digits = 4),
        )
    end
end
