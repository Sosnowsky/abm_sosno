# The coercion-resentment ABM under the three rules of Schøyen (2021) for the authority,
# against the paper's analytic answers:
#
#   (a) rule = :fixed         π frozen at each value; the families settle at q*(π).
#   (b) rule = :exogenous     the theorem of restraint: with the exogenous constraint ρ
#                             anywhere in the inefficient interval, the authority stays
#                             at π_q̲.
#   (c) rule = :insurrection  path dependence: the same parameters from two starting
#                             ns shares end at high or at low coercion.
#   (d) the same, many runs:  with finite N the high-coercion state is only
#                             metastable. Count the runs from the small-ns-share start
#                             that end in the other state, against N.
#
# Run with:  julia --project=. -t auto scripts/coercion_2021.jl

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "CoercionResentment.jl"))
using .CoercionResentment
using Agents, CairoMakie, Statistics, FigureProvenance

const OUT = joinpath(@__DIR__, "..", "plots", "coercion_2021.png")
# The paper's π_q̲ and π^e_q̲. Only DejaVu Sans draws the combining underline of q̲.
const q̲ = subscript("q\u0332"; font = "DejaVu Sans")
const π_q̲, π_q̲e = rich("π", q̲), rich("π", superscript("e"), q̲)
const START_A, START_B = 0.16, 0.30      # starting ns shares for (c) and (d)
const N_SWITCH = [250, 500, 1000, 2000]  # families per country in (d)
const RUNS, YEARS = 20, 3000             # runs per N and years per run in (d)

"Long-run ns share with π frozen at each value (2,500-year means after 500)."
function steady_states(πs)
    q = zeros(length(πs))
    Threads.@threads for i in eachindex(πs)
        h = history(initialize(; rule = :fixed, π0 = πs[i], q0 = 0.5, n_families = 2000, seed = i), 3000)
        q[i] = mean(h.q[501:end])
    end
    return q
end

"The authority's choice π^EC and the long-run q for each exogenous constraint ρ."
function restraint(ρs)
    π, q = zeros(length(ρs)), zeros(length(ρs))
    Threads.@threads for i in eachindex(ρs)
        h = history(initialize(; rule = :exogenous, ρ = ρs[i], q0 = 0.5, seed = 100 + i), 1500)
        π[i], q[i] = h.coercion[end], mean(h.q[501:end])
    end
    return π, q
end

"Fraction of runs from `q0` that end below π = 5 within `YEARS`, for each N."
function switched(q0)
    out = zeros(length(N_SWITCH))
    for (k, n) in enumerate(N_SWITCH)
        low = zeros(Bool, RUNS)
        Threads.@threads for r = 1:RUNS
            h = history(initialize(; rule = :insurrection, q0, n_families = n, seed = 1000 * k + r), YEARS)
            low[r] = h.coercion[end] < 5
        end
        out[k] = mean(low)
    end
    return out
end

"""
The ABM under the three rules of Schøyen (2021) for the authority, each against the paper's analytic
answer, in the paper's notation: (a) with π frozen the families settle on q*(π); (b) the
theorem of restraint, an exogenous constraint ρ anywhere in the inefficient interval
(π_q̲, π^e_q̲) leaves the authority at π_q̲; (c) path dependence under the insurrection constraint
ρ(q), two starting ns shares ending at high and low coercion; (d) what the paper cannot show:
with finite N the high state of start A is metastable, and switches more often the fewer the
families. Okabe–Ito colours, with line style and marker shape as a second cue. The underlined
subscript q̲ is set in DejaVu Sans, since Makie's default font and LaTeX mode cannot draw it.
"""
@provenance function fig_2021(val, rest, paths, sw, p)
    set_theme!(fontsize = 13)
    fig = Figure(size = (1150, 820))
    blue, verm, grey = RGBf(0 / 255, 114 / 255, 178 / 255), RGBf(213 / 255, 94 / 255, 0), RGBf(0.45, 0.45, 0.47)
    interval = RGBAf(0.5, 0.5, 0.5, 0.16)
    πs = range(0, p.π_max, length = 501)
    qstar = [steady_state(π, p) for π in πs]
    πq = best_coercion(5.0, p)                                   # π_q̲
    πqe = πs[findfirst(i -> πs[i] > 6 && qstar[i] <= steady_state(πq, p), eachindex(πs))]   # π^e_q̲
    shade!(ax) = vspan!(ax, πq, πqe; color = interval)

    ax = Axis(fig[1, 1]; xlabel = "coercion π", ylabel = "long-run ns share q",
        title = "(a) rule = :fixed — Lemma 1")
    shade!(ax)
    lines!(ax, πs, qstar; color = :black, linewidth = 2, label = "q*(π), the paper")
    scatter!(ax, val.π, val.q; color = blue, markersize = 9, label = "ABM, 2,000 families")
    axislegend(ax; position = :rt, framevisible = false, labelsize = 11)

    ax = Axis(fig[1, 2]; xlabel = "exogenous constraint ρ", ylabel = rich("coercion chosen, π", superscript("EC")),
        title = "(b) rule = :exogenous — the theorem of restraint")
    shade!(ax)
    cs = range(0, 12, length = 481)
    lines!(ax, [0, 10.5], [0, 10.5]; color = grey, linestyle = :dot, label = "π = ρ")
    lines!(ax, cs, [best_coercion(c, p) for c in cs]; color = :black, linewidth = 2, label = "the paper")
    scatter!(ax, rest.ρ, rest.π; color = blue, markersize = 9, label = "ABM")
    text!(ax, 6.2, πq + 0.35; text = rich(π_q̲, " = $(round(πq, digits = 2))"), fontsize = 11, color = grey)
    axislegend(ax; position = :lt, framevisible = false, labelsize = 11)

    ax = Axis(fig[2, 1]; xlabel = "coercion π", ylabel = "ns share q",
        title = "(c) rule = :insurrection — path dependence")
    qq = range(0, 0.55, length = 400)
    lines!(ax, [insurrection_constraint(q, p) for q in qq], qq; color = grey, linewidth = 2, linestyle = :dash, label = "insurrection constraint ρ(q)")
    lines!(ax, πs, qstar; color = :black, linewidth = 2, label = "q*(π)")
    for (h, c, m, l) in ((paths.a, blue, :circle, "start A, q = $START_A"), (paths.b, verm, :utriangle, "start B, q = $START_B"))
        lines!(ax, h.coercion, h.q; color = (c, 0.6), linewidth = 1)
        scatter!(ax, [h.coercion[end]], [mean(h.q[end-199:end])]; color = c, marker = m, markersize = 14, label = l)
    end
    xlims!(ax, 0, 10.6); ylims!(ax, 0, 0.55)
    axislegend(ax; position = :rt, framevisible = false, labelsize = 11)

    ax = Axis(fig[2, 2]; xscale = log10, xlabel = "families per country N", ylabel = "runs ending in the other state",
        title = "(d) Not in the paper: the high state is metastable",
        xticks = (N_SWITCH, string.(N_SWITCH)), limits = (nothing, (-0.05, 1.05)))
    scatterlines!(ax, N_SWITCH, sw.a; color = blue, marker = :circle, markersize = 11, label = "from start A (ends high in the paper)")
    scatterlines!(ax, N_SWITCH, sw.b; color = verm, marker = :utriangle, markersize = 11, linestyle = :dash, label = "from start B (ends low in the paper)")
    axislegend(ax; position = :rt, framevisible = false, labelsize = 11)

    Label(fig[0, :], rich("The coercion-resentment ABM reproduces Schøyen (2021)  (shaded: the inefficient interval, ",
        π_q̲, " to ", π_q̲e, ")"); fontsize = 16)
    set_theme!()
    return fig
end

# --- Run ---
println("Steady states, restraint and path dependence on $(Threads.nthreads()) threads ...")
p = initialize(; n_families = 1)   # only its parameters are used
val_π = 0:0.5:10
val = (π = val_π, q = steady_states(val_π))
ρs = 0:0.25:12
rest_π, rest_q = restraint(ρs)
rest = (ρ = ρs, π = rest_π, q = rest_q)
paths = (a = history(initialize(; rule = :insurrection, q0 = START_A, n_families = 2000, seed = 1), 1500),
         b = history(initialize(; rule = :insurrection, q0 = START_B, n_families = 2000, seed = 2), 1500))
println("Metastability: $RUNS runs × $YEARS years for each N in $N_SWITCH ...")
sw = (a = switched(START_A), b = 1 .- switched(START_B))

FigureProvenance.save(fig_2021(val, rest, paths, sw, p), OUT)
println("Wrote $OUT")

# --- Numbers behind the figure ---
πq = best_coercion(5.0, p)
println("\n(a) max |q_ABM − q*(π)| = ", round(maximum(abs.(val.q .- [steady_state(π, p) for π in val_π])), digits = 3))
inwin = findall(c -> πq < c < 8.3, ρs)
println("(b) π_q̲ = ", round(πq, digits = 2), "; ρ in the inefficient interval → π^EC in ",
    extrema(rest.π[inwin]), "; max |q − q*(chosen)| = ",
    round(maximum(abs.(rest.q .- [steady_state(π, p) for π in rest.π])), digits = 3))
println("(c) start A ends at π = ", paths.a.coercion[end], ", start B at π = ", paths.b.coercion[end])
println("(d) N = ", N_SWITCH, ": switched from A ", sw.a, ", from B ", sw.b)
