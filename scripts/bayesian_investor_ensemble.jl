# Ensemble time series for the Bayesian business-investor model at r = 1.
#
# Three model-level quantities, each as a function of time, averaged over an
# ensemble of random landscapes and placements:
#
#   1. mean investor wealth      — do investors get richer, and how fast?
#   2. mean *chosen* profit      — do they settle on higher-profit businesses?
#   3. mean *chosen* risk        — do they settle on lower-risk businesses?
#
# "Chosen" profit/risk is the profit/risk of the patch each investor currently
# occupies, averaged over investors: what the population selects, not what the
# landscape offers. Investors cannot see risk in this model — the observer can,
# which is what makes panel 3 the interesting one.
#
# The omniscient model runs alongside as the control. It is the same model with
# the true failure rate handed to the decision rule, so the gap between the two
# curves is exactly what hiding risk costs. Without it the Bayesian risk curve
# has no scale: 0.041 means nothing until you know perfect information buys 0.026.
#
# Run length matters here far more than in the omniscient model — a settled
# investor needs many steps before its evidence outweighs a prior worth 76
# pseudo-observations — so this runs to 250 steps rather than the model's default
# 25. Dashed reference lines mark the landscape means, i.e. no selection at all.
#
# Run with:  julia --project=. scripts/bayesian_investor_ensemble.jl [nsteps]

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "BusinessInvestor.jl"))
include(joinpath(@__DIR__, "..", "src", "BayesianInvestor.jl"))
using Agents, CairoMakie, Statistics

const BI = BusinessInvestor
const BY = BayesianInvestor

# --- Ensemble configuration ---
const DIMS = (19, 19)
const RADIUS = 1
const N_INVESTORS = 10          # ρ = 10/361 ≈ 0.028, matching the existing ensemble study
const NSTEPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 250
const SEEDS = 1:100

"""
    series(initialize, metrics, seed) -> NTuple of length-(NSTEPS+1) vectors

Run one realization and sample `metrics` after every step, including step 0 so
each curve starts from the landscape it was dropped onto.
"""
function series(init, metrics, seed)
    m = init(seed)
    out = [zeros(NSTEPS + 1) for _ in metrics]
    for (j, f) in enumerate(metrics)
        out[j][1] = f(m)
    end
    for t = 1:NSTEPS
        step!(m, 1)
        for (j, f) in enumerate(metrics)
            out[j][t+1] = f(m)
        end
    end
    return out, mean(m.profit), mean(m.risk)
end

# The omniscient model has no belief machinery, so its metrics are defined here
# rather than pulled from the module.
omni_wealth(m) = mean(a.wealth for a in allagents(m))
omni_profit(m) = mean(m.profit[a.pos...] for a in allagents(m))
omni_risk(m) = mean(m.risk[a.pos...] for a in allagents(m))

bay_init(seed) = BY.initialize(;
    dims = DIMS,
    n_investors = N_INVESTORS,
    T = 5,
    sensing_radius = RADIUS,
    seed = seed,
)
omn_init(seed) = BI.initialize(;
    dims = DIMS,
    n_investors = N_INVESTORS,
    T = 5,
    sensing_radius = RADIUS,
    seed = seed,
)

"""Run the whole seed ensemble; return per-time mean and SEM for each metric."""
function run_ensemble(init, metrics)
    runs = [series(init, metrics, s) for s in SEEDS]
    nmetrics = length(metrics)
    means = Vector{Vector{Float64}}(undef, nmetrics)
    sems = Vector{Vector{Float64}}(undef, nmetrics)
    for j = 1:nmetrics
        # (ntimes × nseeds), so statistics run across realizations at fixed time.
        mat = reduce(hcat, (r[1][j] for r in runs))
        means[j] = vec(mean(mat; dims = 2))
        sems[j] = vec(std(mat; dims = 2)) ./ sqrt(length(SEEDS))
    end
    return means, sems, mean(r[2] for r in runs), mean(r[3] for r in runs)
end

println("Running $(length(SEEDS))-seed ensemble, r = $RADIUS, $NSTEPS steps, " *
        "$N_INVESTORS investors (ρ = $(round(N_INVESTORS / prod(DIMS), digits = 3)))...")

bay_mean, bay_sem, land_profit, land_risk =
    run_ensemble(bay_init, (BY.mean_wealth, BY.mean_chosen_profit, BY.mean_chosen_risk))
omn_mean, omn_sem, _, _ =
    run_ensemble(omn_init, (omni_wealth, omni_profit, omni_risk))

times = 0:NSTEPS

# --- Summary at checkpoints -------------------------------------------------
println("\n step |   wealth (bayes / omni) |  chosen P (bayes / omni) |  chosen F (bayes / omni) | recovery")
for t in filter(<=(NSTEPS), (25, 50, 100, 250, NSTEPS)) |> unique
    i = t + 1
    rec = (land_risk - bay_mean[3][i]) / (land_risk - omn_mean[3][i])
    println("  $(lpad(t, 3)) | $(lpad(round(bay_mean[1][i], digits = 1), 9)) / " *
            "$(rpad(round(omn_mean[1][i], digits = 1), 6)) | " *
            "$(lpad(round(bay_mean[2][i], digits = 2), 10)) / " *
            "$(rpad(round(omn_mean[2][i], digits = 2), 7)) | " *
            "$(lpad(round(bay_mean[3][i], digits = 4), 10)) / " *
            "$(rpad(round(omn_mean[3][i], digits = 4), 7)) | " *
            "$(round(100 * rec, digits = 1)) %")
end

# --- Plot -------------------------------------------------------------------
# Same two validated categorical hues as the density diagnostic, so the two
# figures read as one study. Reference lines are muted ink, not a third series.
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

"""Draw one metric: SEM bands under 2px lines, omniscient first so blue reads on top."""
function panel!(ax, idx; reference = nothing)
    if reference !== nothing
        hlines!(ax, [reference]; color = MUTED, linewidth = 2, linestyle = :dash)
    end
    band!(
        ax,
        times,
        omn_mean[idx] .- omn_sem[idx],
        omn_mean[idx] .+ omn_sem[idx];
        color = (OMNI, 0.2),
    )
    band!(
        ax,
        times,
        bay_mean[idx] .- bay_sem[idx],
        bay_mean[idx] .+ bay_sem[idx];
        color = (BAYES, 0.2),
    )
    lines!(ax, times, omn_mean[idx]; color = OMNI, linewidth = 2)
    lines!(ax, times, bay_mean[idx]; color = BAYES, linewidth = 2)
    return ax
end

fig = Figure(size = (900, 1000), backgroundcolor = SURFACE)

Label(
    fig[0, 1],
    "Learning risk by living through it — $NSTEPS steps at sensing radius 1",
    fontsize = 20,
    color = INK,
    halign = :left,
    padding = (4, 0, 0, 8),
)
Label(
    fig[1, 1],
    "19×19 grid, $N_INVESTORS investors (ρ = $(round(N_INVESTORS / prod(DIMS), digits = 3))), " *
    "$(length(SEEDS)) realizations. Bands are ±1 SEM.\n" *
    "Dashed lines mark the landscape mean — where investors sit with no selection at all.",
    fontsize = 13,
    color = INK2,
    halign = :left,
    padding = (4, 0, 0, 10),
)

# One legend for the whole figure: all three panels carry the same two series, and
# per-panel legends collide with the reference lines they sit next to.
Legend(
    fig[2, 1],
    [
        LineElement(color = BAYES, linewidth = 2),
        LineElement(color = OMNI, linewidth = 2),
        LineElement(color = MUTED, linewidth = 2, linestyle = :dash),
    ],
    ["Bayesian (risk hidden)", "omniscient (risk known)", "landscape mean (no selection)"],
    orientation = :horizontal,
    framevisible = false,
    labelcolor = INK2,
    labelsize = 12,
    padding = (0, 0, 0, 0),
)

ax1 = Axis(fig[3, 1]; title = "Mean investor wealth", ylabel = "wealth", axkw...)
panel!(ax1, 1)

ax2 = Axis(
    fig[4, 1];
    title = "Mean chosen profit — what the population selects",
    ylabel = "profit of occupied patches",
    axkw...,
)
panel!(ax2, 2; reference = land_profit)

ax3 = Axis(
    fig[5, 1];
    title = "Mean chosen risk — invisible to the investors, and the whole question",
    xlabel = "step",
    ylabel = "risk of occupied patches",
    axkw...,
)
panel!(ax3, 3; reference = land_risk)

linkxaxes!(ax1, ax2, ax3)
hidexdecorations!(ax1; grid = false)
hidexdecorations!(ax2; grid = false)

outfile = "bayesian_investor_ensemble_t$(NSTEPS).png"
save(outfile, fig)
println("\nWrote $outfile")
