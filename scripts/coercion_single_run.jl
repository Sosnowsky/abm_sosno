# One country of the coercion-resentment model, followed for 1500 years: the
# coercion its successive authorities set, the ns share q that follows it with a
# generational lag, and the same history in the (π, q) plane against the
# steady state q*(π).
#
# Run with:  julia --project=. scripts/coercion_single_run.jl [seed]

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "CoercionResentment.jl"))
using .CoercionResentment
using Agents, CairoMakie, FigureProvenance

const SEED = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 15
const YEARS = 1500
const OUT = joinpath(@__DIR__, "..", "plots", "coercion_single_run.png")

model_params = (; seed = SEED)

"""
One country's history: coercion over time with crackdowns (ticks) and collapses of the
authority (vertical lines); the ns share q against the steady state of the current coercion,
which it chases with a generational lag; and the trajectory in the (π, q) plane over the q*(π)
curve, coloured by time, so the sawtooths sit on the falling branches of q* and the rising
part between them, (π_q̲, π_q̄), is crossed only in a few fast escalations.
"""
@provenance function fig_single_run(h, p)
    set_theme!(fontsize = 13)
    fig = Figure(size = (1350, 620))
    blue, orange = RGBf(0.15, 0.35, 0.62), RGBf(0.85, 0.5, 0.15)
    t = 1:length(h.coercion)

    ax1 = Axis(fig[1, 1]; ylabel = "coercion π", title = "Seed $(p.seed): coercion, crackdowns and collapses")
    any(h.collapse) && vlines!(ax1, findall(h.collapse); color = (:firebrick, 0.5))
    lines!(ax1, t, h.coercion; color = blue)
    any(h.crackdown) && scatter!(ax1, findall(h.crackdown), fill(10.4, count(h.crackdown)); marker = :vline, color = orange, markersize = 8)
    ylims!(ax1, -0.3, 10.8)
    hidexdecorations!(ax1; grid = false)

    ax2 = Axis(fig[2, 1]; xlabel = "year", ylabel = "ns share q")
    lines!(ax2, t, [steady_state(π, p) for π in h.coercion]; color = (:black, 0.35), label = "q*(π(t))")
    lines!(ax2, t, h.q; color = blue, label = "q(t)")
    axislegend(ax2; position = :rt, framevisible = false, labelsize = 11)
    linkxaxes!(ax1, ax2)

    ax3 = Axis(fig[1:2, 2]; xlabel = "coercion π", ylabel = "ns share q", title = "Phase plane")
    πs = range(0, 10, length = 401)
    lines!(ax3, πs, [steady_state(π, p) for π in πs]; color = :black, label = "q*(π)")
    l = lines!(ax3, h.coercion, h.q; color = t, colormap = :viridis, linewidth = 1)
    Colorbar(fig[1:2, 3], l; label = "year")
    axislegend(ax3; position = :rt, framevisible = false, labelsize = 11)
    colsize!(fig.layout, 2, Relative(0.38))

    set_theme!()
    return fig
end

model = initialize(; model_params...)
h = history(model, YEARS)
println("Authorities: ", h.authority[end], ", crackdowns: ", count(h.crackdown))
p = (; model_params..., (k => getproperty(model, k) for k in (:Δu, :resentment_scale, :π̂, :resentment_steepness))...)
FigureProvenance.save(fig_single_run(h, p), OUT)
println("Wrote $OUT")
