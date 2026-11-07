# Conformity in the tourist model: the mixing rule with a sharpened signal
# s_d ∝ c_d^β. Near an even split a destination's excess popularity grows by a
# factor 1 + ε(β − 1) a year whatever the population, so (a) the time until the
# tourists concentrate should depend on ε(β − 1) alone, and (b) a ten times
# larger population should only delay it by about a decade, where without
# conformity (β = 1) it slows the drift tenfold. Both against
# `conformity_prediction`.
#
# Run with:  julia --project=. -t auto scripts/tourist_conformity_sweep.jl

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "TouristDestinations.jl"))
using .TouristDestinations
using Agents, CairoMakie, Statistics, FigureProvenance

const EPSILONS = (0.1, 0.3, 1.0)
const BETAS = (1.05, 1.1, 1.2, 1.3, 1.45, 1.6)
const SEEDS = 1:10
const T_MAX = 2000
const CONCENTRATED = 2          # "concentrated": fewer than 2 effective destinations
const YEARS = 100
const N_RUNS = ((1000, 1.3), (10_000, 1.3), (1000, 1.0), (10_000, 1.0))   # (N, β) at ε = 0.3
const N_SEEDS = 1:100
const OUT = joinpath(@__DIR__, "..", "plots", "tourist_conformity_sweep.png")

"Years (updates) until the popularity has fewer than `CONCENTRATED` effective destinations, or `NaN` by `T_MAX`."
function years_to_concentrate(; ε, β, seed)
    m = initialize(; ε, β, update = :mixing, seed)
    for t = 1:T_MAX
        step!(m, 1)
        effective_number(popularity(m)) < CONCENTRATED && return float(t)
    end
    return NaN
end

"`years_to_concentrate` for every (ε, β, seed): an array indexed `[ε, β, seed]`."
function sweep()
    out = zeros(length(EPSILONS), length(BETAS), length(SEEDS))
    idx = CartesianIndices(out)
    Threads.@threads for k in eachindex(idx)
        i, j, s = Tuple(idx[k])
        out[i, j, s] = years_to_concentrate(; ε = EPSILONS[i], β = BETAS[j], seed = SEEDS[s])
    end
    return out
end

"""
Runs each (N, β) of `N_RUNS` at ε = 0.3 for every seed of `N_SEEDS`, recording Σ p² of the popularity going into
each year (year t follows t − 1 updates). Returns `s2`, the mean over seeds per year for each (N, β), and `years`,
each run's updates until fewer than `CONCENTRATED` effective destinations remain (as `years_to_concentrate`
counts them; `NaN` if not within `YEARS` − 1).
"""
function population_runs()
    jobs = [(j, s) for j in eachindex(N_RUNS) for s in N_SEEDS]
    parts = Vector{Vector{Float64}}(undef, length(jobs))
    Threads.@threads for k in eachindex(jobs)
        j, s = jobs[k]
        N, β = N_RUNS[j]
        h = history(initialize(; ε = 0.3, β, update = :mixing, n_tourists = N, seed = s), YEARS)
        parts[k] = vec(sum(abs2, h.popularity, dims = 1))
    end
    runs(j) = [parts[k] for (k, (jj, _)) in enumerate(jobs) if jj == j]
    s2 = [mean(runs(j)) for j in eachindex(N_RUNS)]
    years = [[something(findfirst(>(1 / CONCENTRATED), r), NaN) - 1 for r in runs(j)] for j in eachindex(N_RUNS)]
    return (; s2, years)
end

"First number of years for which `conformity_prediction` falls below `CONCENTRATED`."
predicted_years(ε, β; N = 1000) = findfirst(T -> conformity_prediction(ε, β, T; n_tourists = N) < CONCENTRATED, 1:100_000)

"""
Whether conformity (signal s_d ∝ c_d^β, β > 1) concentrates the tourists fast, and why. (a) Years
until fewer than 2 effective destinations remain, mean ± s.d. over seeds, for three ε, against
ε(β − 1) on log–log axes: the three sets fall close to one curve, since near an even split the
excess popularity grows by 1 + ε(β − 1) a year (ε enters otherwise only through the size of the
first fluctuations, logarithmically); lines are the linear-noise prediction. (b) The effective
number of destinations over 100 years at ε = 0.3 for 1000 and 10 000 tourists, with conformity
(β = 1.3, solid) and without (β = 1, dashed), as 1 / (mean Σp² over `pop_seeds` seeds; 8 seeds
left the curves visibly noisy), against the linear-noise prediction (dotted): with conformity
ten times more people only delay concentration by about a decade, without it they slow the
drift tenfold. Each panel's seed count is its subtitle, so the two titles line up.
Categorical colours in fixed order.
"""
@provenance function fig_conformity_sweep(yrs, epsilons, betas, pop, n_runs; pop_seeds)
    set_theme!(fontsize = 13)
    ink, muted = colorant"#0b0b0b", colorant"#898781"
    slots = (colorant"#2a78d6", colorant"#eb6834", colorant"#1baf7a")

    fig = Figure(size = (1250, 540))
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "ε (β − 1)",
        ylabel = "years until fewer than 2 effective destinations",
        title = "(a) The speed depends mostly on ε(β − 1)",
        subtitle = "ABM: mean ± s.d. over $(size(yrs, 3)) seeds", subtitlecolor = muted,
        xticks = [0.005, 0.01, 0.03, 0.1, 0.3], yticks = [3, 10, 30, 100, 300, 1000])
    for (i, ε) in enumerate(epsilons)
        x = [ε * (β - 1) for β in betas]
        m = [mean(filter(!isnan, yrs[i, j, :])) for j in eachindex(betas)]
        sd = [std(filter(!isnan, yrs[i, j, :])) for j in eachindex(betas)]
        xs = 10 .^ range(log10(minimum(x)), log10(maximum(x)), length = 60)
        lines!(ax, xs, [predicted_years(ε, 1 + xx / ε) for xx in xs]; color = slots[i], linestyle = :dot, linewidth = 1.5)
        errorbars!(ax, x, m, sd; color = slots[i], whiskerwidth = 6)
        scatter!(ax, x, m; color = slots[i], markersize = 9, label = "ε = $(isinteger(ε) ? Int(ε) : ε)")
    end
    axislegend(ax; position = :rt, framevisible = false, labelsize = 11)

    T = length(pop[1])
    ax = Axis(fig[1, 2]; yscale = log10, xlabel = "year", ylabel = "effective number of destinations",
        title = "(b) Population size barely matters with conformity (ε = 0.3)",
        subtitle = "ABM: 1 / (mean Σp² over $pop_seeds seeds)", subtitlecolor = muted,
        yticks = [1, 3, 10, 30, 100])
    for (j, (N, β)) in enumerate(n_runs)
        c = N == first(n_runs[1]) ? slots[1] : slots[2]
        ls = β == 1 ? :dash : :solid
        lines!(ax, 1:T, 1 ./ pop[j]; color = c, linestyle = ls, linewidth = 2.5,
            label = "N = $N, β = $(isinteger(β) ? Int(β) : β)")
        lines!(ax, 1:T, [conformity_prediction(0.3, β, t - 1; n_tourists = N) for t = 1:T];
            color = ink, linestyle = :dot, linewidth = 1.2)
    end
    ylims!(ax, 0.9, 120)
    axislegend(ax; position = :lb, framevisible = false, labelsize = 11)

    Legend(fig[2, 1:2],
        [MarkerElement(color = muted, marker = :circle, markersize = 9), LineElement(color = muted, linestyle = :dot),
            LineElement(color = ink, linestyle = :dot)],
        ["(a) ABM", "(a) linear-noise prediction, coloured by ε",
            "(b) linear-noise prediction, valid while popularity is near even"];
        orientation = :horizontal, framevisible = false)
    Label(fig[0, :], rich("Conformity, s", subscript("d"), " ∝ c", subscript("d"), superscript("β"),
            ": popularity grows by 1 + ε(β − 1) a year, so one destination takes everyone within decades, " *
            "whatever the population"); fontsize = 15)
    set_theme!()
    return fig
end

println("Conformity: $(length(EPSILONS)) ε × $(length(BETAS)) β × $(length(SEEDS)) seeds (up to $T_MAX years), then " *
        "$(length(N_RUNS)) population runs × $(length(N_SEEDS)) seeds, on $(Threads.nthreads()) threads ...")
yrs = sweep()
pop = population_runs()
FigureProvenance.save(fig_conformity_sweep(yrs, EPSILONS, BETAS, pop.s2, N_RUNS; pop_seeds = length(N_SEEDS)), OUT)
println("Wrote $OUT")

println("\n    ε     β   ε(β−1)   years to < 2 (ABM mean ± sd)   prediction")
for (i, ε) in enumerate(EPSILONS), (j, β) in enumerate(BETAS)
    y = filter(!isnan, yrs[i, j, :])
    println(lpad(ε, 5), lpad(β, 6), lpad(round(ε * (β - 1), digits = 3), 9),
        lpad("$(round(mean(y), digits = 1)) ± $(round(std(y), digits = 1))", 22),
        length(y) < size(yrs, 3) ? " ($(size(yrs, 3) - length(y)) runs not concentrated)" : "",
        lpad(predicted_years(ε, β), 16))
end
for (j, (N, β)) in enumerate(N_RUNS)
    y = filter(!isnan, pop.years[j])
    println("ε = 0.3, N = $N, β = $β: effective number in years 10, 20, 30, 50, 100: ",
        [round(1 / pop.s2[j][t], digits = 1) for t in (10, 20, 30, 50, 100)], "; years to < 2: ",
        isempty(y) ? "none within $(YEARS - 1)" :
        "$(round(mean(y), digits = 1)) ± $(round(std(y), digits = 1)) (s.e. $(round(std(y) / sqrt(length(y)), digits = 1)), " *
        "$(length(y)) of $(length(pop.years[j])) runs), prediction $(predicted_years(0.3, β; N))")
end
