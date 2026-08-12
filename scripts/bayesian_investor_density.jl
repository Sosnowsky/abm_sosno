# Diagnostic: why raising investor density does NOT restore risk selection in the
# Bayesian model.
#
# The claim under test is that vicarious learning cannot help, for a structural
# reason rather than a tuning one: **occupancy is simultaneously what generates
# evidence and what removes opportunity.** An investor learns about a patch only
# while someone is standing on it, but can only move to a patch that is empty. So
# raising density buys more evidence and spends exactly the resource — vacancy —
# needed to act on it.
#
# The script sweeps investor density at fixed sensing radius and measures the two
# halves of that trade separately, plus their product:
#
#   1. evidence supply    — patches an investor holds evidence about        (rises)
#   2. opportunity        — empty candidate patches within reach            (falls)
#   3. actionable knowledge — candidates that are BOTH empty and known      (the product)
#   4. outcome            — mean risk of the patches actually chosen
#
# Panel 3 is the load-bearing one. If the mechanism is real, actionable knowledge
# stays tiny at every density and peaks somewhere in the middle, because the two
# factors are maximised at opposite ends of the sweep.
#
# The omniscient model is run at every density as a control. It has perfect risk
# information, so any degradation *it* shows with density is pure crowding — that
# separates "the Bayesian model cannot select" from "nobody can select when the
# grid is full", which is the confound this script exists to resolve.
#
# Run with:  julia --project=. scripts/bayesian_investor_density.jl

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "BusinessInvestor.jl"))
include(joinpath(@__DIR__, "..", "src", "BayesianInvestor.jl"))
using Agents, CairoMakie, Statistics
using .BusinessInvestor: BusinessInvestor
using .BayesianInvestor: BayesianInvestor

const BI = BusinessInvestor
const BY = BayesianInvestor

# --- Sweep configuration ---
const DIMS = (19, 19)
const NPATCHES = prod(DIMS)
const RADIUS = 1
# Run length is the sweep's other axis. 25 is the model default; longer runs let a
# settled investor accumulate evidence, and let neighbouring patches cycle through
# the occupied-then-vacated state that makes them both known and reachable.
const NSTEPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 25
const SEEDS = 1:40
const COUNTS = [5, 10, 20, 40, 70, 100, 140, 200, 260, 320]

"""
    candidate_stats(model) -> (empty_per_investor, informed_per_investor)

The two halves of the trade, measured on the current configuration.

`empty` counts the patches an investor could actually move to. `informed` counts
the subset of those it also holds evidence about — a patch qualifies only if it
was watched *while occupied* and has since been vacated, which is precisely the
narrow intersection the mechanism predicts to be small.
"""
function candidate_stats(model)
    r = model.sensing_radius
    n_empty, n_informed, n_ag = 0, 0, 0
    for a in allagents(model)
        n_ag += 1
        for p in nearby_positions(a, model, r)
            if isempty(p, model)
                n_empty += 1
                first(BY.belief(a, p)) > 0 && (n_informed += 1)
            end
        end
    end
    n_ag == 0 && return (0.0, 0.0)
    return n_empty / n_ag, n_informed / n_ag
end

"""One Bayesian realization; returns run-averaged diagnostics and final outcomes."""
function run_bayes(n_investors, seed)
    m = BY.initialize(;
        dims = DIMS,
        n_investors,
        T = 5,
        sensing_radius = RADIUS,
        seed = seed,
    )
    observed = emptyc = informed = 0.0
    for _ = 1:NSTEPS
        step!(m, 1)
        e, i = candidate_stats(m)
        observed += BY.mean_observed_patches(m)
        emptyc += e
        informed += i
    end
    return (
        observed = observed / NSTEPS,
        empties = emptyc / NSTEPS,
        informed = informed / NSTEPS,
        chosen_risk = BY.mean_chosen_risk(m),
        landscape_risk = mean(m.risk),
        shift = BY.mean_posterior_shift(m),
    )
end

"""One omniscient realization — the crowding control."""
function run_omniscient(n_investors, seed)
    m = BI.initialize(;
        dims = DIMS,
        n_investors,
        T = 5,
        sensing_radius = RADIUS,
        seed = seed,
    )
    step!(m, NSTEPS)
    return (
        chosen_risk = mean(m.risk[a.pos...] for a in allagents(m)),
        landscape_risk = mean(m.risk),
    )
end

# --- Run the sweep ---
println("Sweeping density at r = $RADIUS over $(length(SEEDS)) seeds, $NSTEPS steps...")
avg(rows, f) = mean(f(r) for r in rows)
sem(rows, f) = std(f(r) for r in rows) / sqrt(length(rows))

density = Float64[]
observed = Float64[]
empties = Float64[]
informed = Float64[]
bay_risk = Float64[]
bay_sem = Float64[]
omn_risk = Float64[]
omn_sem = Float64[]
land_risk = Float64[]
shift = Float64[]

for n in COUNTS
    b = [run_bayes(n, s) for s in SEEDS]
    o = [run_omniscient(n, s) for s in SEEDS]
    push!(density, n / NPATCHES)
    push!(observed, avg(b, r -> r.observed))
    push!(empties, avg(b, r -> r.empties))
    push!(informed, avg(b, r -> r.informed))
    push!(bay_risk, avg(b, r -> r.chosen_risk))
    push!(bay_sem, sem(b, r -> r.chosen_risk))
    push!(omn_risk, avg(o, r -> r.chosen_risk))
    push!(omn_sem, sem(o, r -> r.chosen_risk))
    push!(land_risk, avg(b, r -> r.landscape_risk))
    push!(shift, avg(b, r -> r.shift))
    println("  n = $(lpad(n, 3))  ρ = $(rpad(round(n / NPATCHES, digits = 3), 5))  " *
            "known = $(rpad(round(observed[end], digits = 2), 5))  " *
            "empty = $(rpad(round(empties[end], digits = 2), 5))  " *
            "actionable = $(rpad(round(informed[end], digits = 3), 5))  " *
            "F_bayes = $(round(bay_risk[end], digits = 4))  " *
            "F_omni = $(round(omn_risk[end], digits = 4))")
end

# Risk reduction actually achieved, relative to sitting on a random landscape.
bay_gain = land_risk .- bay_risk
omn_gain = land_risk .- omn_risk

println("\nRecovery of the omniscient advantage (bayes gain / omniscient gain):")
for (i, n) in enumerate(COUNTS)
    rec = omn_gain[i] > 1e-6 ? bay_gain[i] / omn_gain[i] : NaN
    println("  n = $(lpad(n, 3))  $(round(100 * rec, digits = 1)) %")
end

# --- Plot -------------------------------------------------------------------
# Two categorical hues only (Bayesian, omniscient); reference lines are muted ink,
# not a third series. Validated: worst-pair CVD ΔE 24.7, normal-vision 33.6.
const SURFACE = colorant"#fcfcfb"
const INK = colorant"#0b0b0b"
const INK2 = colorant"#52514e"
const MUTED = colorant"#898781"
const GRID = colorant"#e1e0d9"
const AXIS = colorant"#c3c2b7"
const BAYES = colorant"#2a78d6"
const OMNI = colorant"#eb6834"

axkw = (;
    backgroundcolor = SURFACE,
    xgridcolor = GRID,
    ygridcolor = GRID,
    xgridwidth = 1,
    ygridwidth = 1,
    bottomspinecolor = AXIS,
    leftspinecolor = AXIS,
    topspinevisible = false,
    rightspinevisible = false,
    xtickcolor = AXIS,
    ytickcolor = AXIS,
    xticklabelcolor = INK2,
    yticklabelcolor = INK2,
    xlabelcolor = INK2,
    ylabelcolor = INK2,
    titlecolor = INK,
    titlealign = :left,
    titlesize = 15,
)

fig = Figure(size = (1020, 1000), backgroundcolor = SURFACE)

# Headline figures are computed, not asserted — the conclusion changes with run
# length, so the caption has to follow the data rather than restate a prior belief.
ev_ratio = observed[end] / observed[1]
op_ratio = empties[1] / empties[end]
act_ratio = informed[end] / informed[1]
recovery = [o > 1e-6 ? b / o : NaN for (b, o) in zip(bay_gain, omn_gain)]

Label(
    fig[0, 1:2],
    "Investor density: evidence supply against the opportunity to act on it — $NSTEPS steps",
    fontsize = 20,
    color = INK,
    halign = :left,
    padding = (4, 0, 0, 8),
)
Label(
    fig[1, 1:2],
    "Across the sweep, evidence per investor rises $(round(ev_ratio, digits = 1))× " *
    "while reachable patches fall $(round(op_ratio, digits = 1))×; " *
    "their product changes $(round(act_ratio, digits = 2))×.\n" *
    "19×19 grid, sensing radius 1, $NSTEPS steps, $(length(SEEDS)) seeds per point. " *
    "Bands are ±1 SEM.",
    fontsize = 13,
    color = INK2,
    halign = :left,
    padding = (4, 0, 0, 10),
)

# -- Panel A: evidence supply rises --
axA = Axis(
    fig[2, 1];
    title = "A · Evidence supply rises with density",
    xlabel = "investor density ρ",
    ylabel = "patches known per investor",
    axkw...,
)
lines!(axA, density, observed; color = BAYES, linewidth = 2)
scatter!(axA, density, observed; color = BAYES, markersize = 9)

# -- Panel B: opportunity falls --
axB = Axis(
    fig[2, 2];
    title = "B · Opportunity falls with density",
    xlabel = "investor density ρ",
    ylabel = "empty patches in reach per investor",
    axkw...,
)
lines!(axB, density, empties; color = BAYES, linewidth = 2)
scatter!(axB, density, empties; color = BAYES, markersize = 9)

# -- Panel C: the product — the load-bearing panel --
axC = Axis(
    fig[3, 1];
    title = "C · Actionable knowledge (empty AND known)",
    xlabel = "investor density ρ",
    ylabel = "empty AND known patches per investor",
    axkw...,
)
lines!(axC, density, informed; color = BAYES, linewidth = 2)
scatter!(axC, density, informed; color = BAYES, markersize = 9)
peak = argmax(informed)
text!(
    axC,
    density[peak],
    informed[peak];
    text = "peak $(round(informed[peak], digits = 2)) of $(round(empties[peak], digits = 1)) candidates",
    color = INK2,
    fontsize = 12,
    align = (:left, :bottom),
    offset = (8, 6),
)
ylims!(axC, 0, max(0.6, 1.5 * maximum(informed)))

# -- Panel D: outcome --
axD = Axis(
    fig[3, 2];
    title = "D · Chosen risk against the landscape mean",
    xlabel = "investor density ρ",
    ylabel = "mean risk of occupied patches",
    axkw...,
)
lines!(
    axD,
    density,
    land_risk;
    color = MUTED,
    linewidth = 2,
    linestyle = :dash,
    label = "landscape mean (no selection)",
)
band!(axD, density, bay_risk .- bay_sem, bay_risk .+ bay_sem; color = (BAYES, 0.18))
band!(axD, density, omn_risk .- omn_sem, omn_risk .+ omn_sem; color = (OMNI, 0.18))
lines!(axD, density, bay_risk; color = BAYES, linewidth = 2, label = "Bayesian (risk hidden)")
scatter!(axD, density, bay_risk; color = BAYES, markersize = 9)
lines!(axD, density, omn_risk; color = OMNI, linewidth = 2, label = "omniscient (risk known)")
scatter!(axD, density, omn_risk; color = OMNI, markersize = 9)
axislegend(axD; position = :lb, framevisible = false, labelcolor = INK2, labelsize = 12)

# -- Panel E: the punchline, zoomed onto the gap --
axE = Axis(
    fig[4, 1:2];
    title = "E · Risk reduction achieved — the gap is what hiding risk costs",
    xlabel = "investor density ρ",
    ylabel = "landscape mean − chosen risk",
    axkw...,
)
hlines!(axE, [0.0]; color = AXIS, linewidth = 1)
band!(axE, density, bay_gain .- bay_sem, bay_gain .+ bay_sem; color = (BAYES, 0.18))
band!(axE, density, omn_gain .- omn_sem, omn_gain .+ omn_sem; color = (OMNI, 0.18))
lines!(axE, density, omn_gain; color = OMNI, linewidth = 2, label = "omniscient (risk known)")
scatter!(axE, density, omn_gain; color = OMNI, markersize = 9)
lines!(axE, density, bay_gain; color = BAYES, linewidth = 2, label = "Bayesian (risk hidden)")
scatter!(axE, density, bay_gain; color = BAYES, markersize = 9)
axislegend(axE; position = :rt, framevisible = false, labelcolor = INK2, labelsize = 12)

# Label recovery at the two configurations the model documents: the ensemble
# density (10 investors) and the single-run density (100).
for n in (10, 100)
    i = findfirst(==(n), COUNTS)
    (isnothing(i) || isnan(recovery[i])) && continue
    text!(
        axE,
        density[i],
        bay_gain[i];
        text = "n=$n: $(round(100 * recovery[i], digits = 0))% recovered",
        color = INK2,
        fontsize = 12,
        align = (:left, :top),
        offset = (6, -8),
    )
end

rowsize!(fig.layout, 4, Relative(0.26))
outfile = "bayesian_investor_density_t$(NSTEPS).png"
save(outfile, fig)
println("\nWrote $outfile")
