# Sweep of the influence parameter ε in the tourist-destination model: after
# 100 years, how unequal has the destinations' popularity become, and how much
# of it was decided by the luck of year 1? The ABM, over many seeds, against the
# mean-field Pólya-urn prediction `polya_prediction`.
#
# Run with:  julia --project=. -t auto scripts/tourist_influence_sweep.jl

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "TouristDestinations.jl"))
using .TouristDestinations
using Agents, CairoMakie, Statistics, FigureProvenance

const SEEDS = 1:20
const YEARS = 100
const EPSILONS = 10 .^ range(-3, 2, length = 16)
const OUT = joinpath(@__DIR__, "..", "plots", "tourist_influence_sweep.png")

"""
    sweep() -> (; neff, lock_in)

Run every `(ε, seed)` pair for `YEARS` years. Returns two `length(EPSILONS) × length(SEEDS)`
matrices: the effective number of destinations of the popularity after the last year, and the
correlation across destinations between year-1 visitors and that popularity.
"""
function sweep()
    neff = zeros(length(EPSILONS), length(SEEDS))
    lock_in = similar(neff)
    idx = CartesianIndices(neff)
    Threads.@threads for k in eachindex(idx)
        i, s = Tuple(idx[k])
        m = initialize(; ε = EPSILONS[i], seed = SEEDS[s])
        h = history(m, YEARS)
        p = popularity(m)
        neff[i, s] = effective_number(p)
        lock_in[i, s] = cor(h.visits[:, 1], p)
    end
    return (; neff, lock_in)
end

"""
The ABM against the Pólya-urn prediction across the influence parameter ε, after 100 years,
mean ± s.d. over seeds. (a) The effective number of destinations of the popularity, on its full
scale from 1 (everyone goes to the same place) to D: it falls only to the strong-influence limit
D/(1 + (D−1)Σ1/t²/N), about 86 for N = 1000, D = 100, so no ε makes this model concentrate.
(b) How much of the final popularity is year 1's luck: the correlation between year-1 visitors
and the final popularity, rising from 1/√T at weak influence to 0.78 at strong.
"""
@provenance function fig_influence_sweep(res, epsilons, years; n_tourists = 1000, n_destinations = 100)
    set_theme!(fontsize = 13)
    ink, muted, abm = colorant"#0b0b0b", colorant"#898781", colorant"#2a78d6"
    N, D = n_tourists, n_destinations
    εs = 10 .^ range(log10(first(epsilons)), log10(last(epsilons)), length = 200)
    pred = [polya_prediction(ε, years; n_tourists = N, n_destinations = D) for ε in εs]
    limit = D / (1 + (D - 1) * sum(1 / t^2 for t = 1:years) / N)
    xticks = (10.0 .^ (-3:2), ["10⁻³", "10⁻²", "10⁻¹", "1", "10", "10²"])

    fig = Figure(size = (1150, 480))
    ax = Axis(fig[1, 1]; xscale = log10, xticks, xlabel = "influence ε",
        ylabel = "effective number of destinations, 1 / Σ p²",
        title = "(a) How unequal popularity is after $years years")
    hlines!(ax, [D]; color = muted, linestyle = :dot)
    hlines!(ax, [limit]; color = muted, linestyle = :dash)
    text!(ax, first(epsilons), D; text = "ε = 0: all $D equally popular", align = (:left, :bottom),
        color = muted, fontsize = 11, offset = (2, 2))
    text!(ax, first(epsilons), limit; text = "strong-influence limit, $(round(limit, digits = 1))",
        align = (:left, :top), color = muted, fontsize = 11, offset = (2, -2))
    text!(ax, first(epsilons), 1; text = "1 = everyone goes to the same place", align = (:left, :bottom),
        color = muted, fontsize = 11, offset = (2, 4))
    lines!(ax, εs, getfield.(pred, :effective_number); color = ink)
    errorbars!(ax, epsilons, vec(mean(res.neff, dims = 2)), vec(std(res.neff, dims = 2)); color = abm, whiskerwidth = 6)
    scatter!(ax, epsilons, vec(mean(res.neff, dims = 2)); color = abm, markersize = 8)
    ylims!(ax, 0, 1.08D)

    ax = Axis(fig[1, 2]; xscale = log10, xticks, xlabel = "influence ε",
        ylabel = "corr(year-1 visitors, final popularity)",
        title = "(b) How much of it is year 1's luck")
    lines!(ax, εs, getfield.(pred, :lock_in); color = ink)
    errorbars!(ax, epsilons, vec(mean(res.lock_in, dims = 2)), vec(std(res.lock_in, dims = 2)); color = abm, whiskerwidth = 6)
    scatter!(ax, epsilons, vec(mean(res.lock_in, dims = 2)); color = abm, markersize = 8)
    ylims!(ax, 0, 1)

    Legend(fig[2, 1:2], [MarkerElement(color = abm, marker = :circle, markersize = 8), LineElement(color = ink)],
        ["ABM, mean ± s.d. over $(size(res.neff, 2)) seeds", "Pólya-urn prediction"];
        orientation = :horizontal, framevisible = false)
    Label(fig[0, :], "$N tourists, $D destinations: social influence freezes popularity but does not concentrate it";
        fontsize = 16)
    set_theme!()
    return fig
end

println("Running $(length(EPSILONS)) values of ε × $(length(SEEDS)) seeds × $YEARS years on $(Threads.nthreads()) threads ...")
res = sweep()
FigureProvenance.save(fig_influence_sweep(res, EPSILONS, YEARS), OUT)
println("Wrote $OUT")

println("\n      ε    Neff ABM   Neff Pólya   lock-in ABM   lock-in Pólya")
for (i, ε) in enumerate(EPSILONS)
    pr = polya_prediction(ε, YEARS)
    println(lpad(round(ε, sigdigits = 2), 7), lpad(round(mean(res.neff[i, :]), digits = 1), 12),
        lpad(round(pr.effective_number, digits = 1), 13), lpad(round(mean(res.lock_in[i, :]), digits = 2), 14),
        lpad(round(pr.lock_in, digits = 2), 16))
end
