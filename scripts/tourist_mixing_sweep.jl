# Does the mixing rule, w ← (1 − ε) w + (ε/N) Σ d with w a distribution,
# concentrate the tourists where the additive rule cannot? (a) The effective
# number of destinations after 100 years across ε, against the neutral-drift
# prediction `drift_prediction` and the additive rule's floor. (b) The same over
# 5000 years, for two mixing ε and the additive rule, each against its
# mean-field prediction.
#
# Run with:  julia --project=. -t auto scripts/tourist_mixing_sweep.jl

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "TouristDestinations.jl"))
using .TouristDestinations
using Agents, CairoMakie, Statistics, FigureProvenance

const YEARS = 100
const EPSILONS = 0.05:0.05:1.0
const SEEDS = 1:20
const LONG_YEARS = 5000
const LONG_RUNS = ((:mixing, 0.3), (:mixing, 1.0), (:additive, 1.0))
const LONG_SEEDS = 1:8
const OUT = joinpath(@__DIR__, "..", "plots", "tourist_mixing_sweep.png")

"Σ p² of the popularity after `YEARS` years, for every `(ε, seed)` under the mixing rule."
function sweep()
    s2 = zeros(length(EPSILONS), length(SEEDS))
    idx = CartesianIndices(s2)
    Threads.@threads for k in eachindex(idx)
        i, s = Tuple(idx[k])
        m = initialize(; ε = EPSILONS[i], update = :mixing, seed = SEEDS[s])
        history(m, YEARS)
        s2[i, s] = sum(abs2, popularity(m))
    end
    return s2
end

"Mean over `LONG_SEEDS` of Σ p² of the popularity going into each year, for each of `LONG_RUNS`."
function long_runs()
    out = [zeros(LONG_YEARS) for _ in LONG_RUNS]
    jobs = [(j, s) for j in eachindex(LONG_RUNS) for s in LONG_SEEDS]
    parts = Vector{Vector{Float64}}(undef, length(jobs))
    Threads.@threads for k in eachindex(jobs)
        j, s = jobs[k]
        update, ε = LONG_RUNS[j]
        h = history(initialize(; ε, update, seed = s), LONG_YEARS)
        parts[k] = vec(sum(abs2, h.popularity, dims = 1))
    end
    for (k, (j, _)) in enumerate(jobs)
        out[j] .+= parts[k] ./ length(LONG_SEEDS)
    end
    return out
end

"""
Whether the mixing rule concentrates the tourists, as the effective number of destinations
1/Σp² of the popularity (1 = everyone at one place, D = an even split). (a) After 100 years
across ε: one faint dot per seed, the seed average as 1/⟨Σp²⟩, and the neutral-drift prediction
1/(1 − (1 − 1/D)(1 − ε²/N)^T); the dashed line is the most the additive rule can concentrate,
at any ε. (b) Over 5000 years on log–log axes (grey line: year 100), seed averages against their
mean-field predictions: the mixing rule drifts towards one destination on a time scale of N/ε² years, while
the additive rule freezes within a few years. Categorical colours in fixed order, black dotted
lines for the predictions.
"""
@provenance function fig_mixing_sweep(s2, epsilons, years, long, long_runs; n_tourists = 1000, n_destinations = 100)
    set_theme!(fontsize = 13)
    ink, muted = colorant"#0b0b0b", colorant"#898781"
    slots = (colorant"#2a78d6", colorant"#eb6834", colorant"#1baf7a")
    N, D = n_tourists, n_destinations
    floor = polya_prediction(1e9, years; n_tourists = N, n_destinations = D).effective_number

    fig = Figure(size = (1250, 520))
    ax = Axis(fig[1, 1]; xlabel = "influence ε (fraction of w replaced each year)",
        ylabel = "effective number of destinations, 1 / Σ p²",
        title = "(a) Mixing rule after $years years")
    hlines!(ax, [D]; color = muted, linestyle = :dot)
    hlines!(ax, [floor]; color = muted, linestyle = :dash)
    text!(ax, 0, D; text = "ε = 0: all $D equally popular", align = (:left, :bottom), color = muted,
        fontsize = 11, offset = (4, 2))
    text!(ax, 1, floor; text = "additive rule at any ε: no lower than $(round(floor, digits = 1))",
        align = (:right, :bottom), color = muted, fontsize = 11, offset = (-4, 2))
    for j in axes(s2, 2)
        scatter!(ax, epsilons, 1 ./ s2[:, j]; color = (slots[1], 0.25), markersize = 5)
    end
    εs = range(0, 1, length = 200)
    lines!(ax, εs, [drift_prediction(ε, years; n_tourists = N, n_destinations = D) for ε in εs];
        color = ink, linestyle = :dot, linewidth = 2)
    scatter!(ax, epsilons, 1 ./ vec(mean(s2, dims = 2)); color = slots[1], markersize = 9)
    limits!(ax, -0.02, 1.02, 0, 1.08D)

    T = length(long[1])
    ax = Axis(fig[1, 2]; xscale = log10, yscale = log10, xlabel = "year", ylabel = "effective number of destinations",
        title = "(b) Over $T years", xticks = [1, 10, 100, 1000], yticks = [1, 3, 10, 30, 100])
    vlines!(ax, [years]; color = muted, linewidth = 1)
    t = 1:T
    for (j, (update, ε)) in enumerate(long_runs)
        lines!(ax, t, 1 ./ long[j]; color = slots[j], linewidth = 2.5, label = "$update, ε = $(isinteger(ε) ? Int(ε) : ε)")
        pred = update === :mixing ? [drift_prediction(ε, k - 1; n_tourists = N, n_destinations = D) for k in t] :
               [polya_prediction(ε, k - 1; n_tourists = N, n_destinations = D).effective_number for k in t]
        lines!(ax, t, pred; color = ink, linestyle = :dot, linewidth = 1.5)
    end
    ylims!(ax, 0.9, 1.3D)
    axislegend(ax; position = :lb, framevisible = false, labelsize = 11)

    Legend(fig[2, 1:2],
        [MarkerElement(color = slots[1], marker = :circle, markersize = 9),
            MarkerElement(color = (slots[1], 0.25), marker = :circle, markersize = 5),
            LineElement(color = ink, linestyle = :dot, linewidth = 2)],
        ["mixing rule, 1/⟨Σp²⟩ over $(size(s2, 2)) seeds", "one seed",
            "mean-field prediction (neutral drift; Pólya urn for the additive rule)"];
        orientation = :horizontal, framevisible = false)
    Label(fig[0, :], "$N tourists, $D destinations: with forgetting, popularity drifts until one destination has everyone, " *
                     "on a time scale of N/ε² years"; fontsize = 15)
    set_theme!()
    return fig
end

println("Mixing rule: $(length(EPSILONS)) values of ε × $(length(SEEDS)) seeds × $YEARS years, then " *
        "$(length(LONG_RUNS)) runs × $(length(LONG_SEEDS)) seeds × $LONG_YEARS years, on $(Threads.nthreads()) threads ...")
s2 = sweep()
long = long_runs()
FigureProvenance.save(fig_mixing_sweep(s2, EPSILONS, YEARS, long, LONG_RUNS), OUT)
println("Wrote $OUT")

println("\n    ε   1/<Σp²> ABM   drift prediction")
for (i, ε) in enumerate(EPSILONS)
    i % 2 == 0 || continue
    println(lpad(ε, 5), lpad(round(1 / mean(s2[i, :]), digits = 1), 13), lpad(round(drift_prediction(ε, YEARS), digits = 1), 18))
end
for (j, (update, ε)) in enumerate(LONG_RUNS)
    println("$update, ε = $ε: effective number in years 100, 1000, 5000: ",
        [round(1 / long[j][k], digits = 1) for k in (100, 1000, LONG_YEARS)])
end
