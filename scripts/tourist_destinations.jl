# Where 1000 tourists go on holiday over 100 years, under each update rule:
# `:additive` (w ← w + ε Σ_{n′≠n} d) with no, weak and strong influence;
# `:mixing` (w ← (1 − ε) w + (ε/N) Σ d, w a distribution) with weak to total
# influence; and the mixing rule with conformity (signal s_d ∝ c_d^β) at fixed ε
# and growing β. The runs of one figure share the seed, hence the same tastes and
# the same first year, so any difference after year 1 is the influence alone.
#
# Run with:  julia --project=. scripts/tourist_destinations.jl [seed]

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "TouristDestinations.jl"))
using .TouristDestinations
using Agents, CairoMakie, Statistics, FigureProvenance

const SEED = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1
const YEARS = 100
const N_HIGHLIGHT = 5
const PLOTS = joinpath(@__DIR__, "..", "plots")
# One figure per rule: the model's `update`, one parameter set per column, and how to draw it.
const FIGURES = (
    additive = (update = :additive, columns = [(ε = 0.0,), (ε = 0.01,), (ε = 1.0,)],
        style = (rank = :first, vmax = 30, scale = identity),
        out = joinpath(PLOTS, "tourist_destinations.png")),
    mixing = (update = :mixing, columns = [(ε = 0.1,), (ε = 0.3,), (ε = 1.0,)],
        style = (rank = :last, vmax = 300, scale = Makie.pseudolog10),
        out = joinpath(PLOTS, "tourist_destinations_mixing.png")),
    conformity = (update = :mixing, columns = [(ε = 0.3, β = 1.0), (ε = 0.3, β = 1.1), (ε = 0.3, β = 1.3)],
        style = (rank = :last, vmax = 1000, scale = Makie.pseudolog10),
        out = joinpath(PLOTS, "tourist_destinations_conformity.png")),
)

"A parameter as written in a title: `1` rather than `1.0`."
fmt(x) = isinteger(x) ? string(Int(x)) : string(x)

"""
How the chosen destinations evolve over 100 years, one column per parameter set, from the same
seed (same tastes, same first year), for one update rule (`rule`: additive, mixing, or mixing
with conformity). Top: visitors to each destination each year. Under the additive rule
(`rank = :first`) the rows of every panel are ranked by the shared year-1 visitors, so a gradient
that survives to the right is year 1's luck locked in. Under the mixing rules (`rank = :last`)
each panel is ranked by its own final popularity, since year 1 no longer predicts it, so the
panel shows the winners emerging and the losers dying out. `scale = Makie.pseudolog10`, on the
colours and on the bottom row's y axis, keeps both 10 and 100+ visitors readable when a few
destinations take most tourists. Bottom: each destination's expected visitors per year (its
popularity × N), the noise-free version of the rows above, with the 5 destinations first and
last in that ranking highlighted. Subtitles give the effective number of destinations in the
final year, how many got any visitor, and the correlation between year-1 visitors and final
popularity.
"""
@provenance function fig_time_evolution(runs, columns, seed; rule = :additive, rank = :first, vmax = 30,
    scale = identity)
    set_theme!(fontsize = 13)
    ink, muted = colorant"#0b0b0b", colorant"#898781"
    lucky, unlucky, others = colorant"#2a78d6", colorant"#eb6834", RGBAf(0.54, 0.53, 0.51, 0.3)
    # Single-hue blue ramp, light (no visitors) to dark.
    ramp = cgrad([colorant"#fcfcfb", colorant"#cde2fb", colorant"#86b6ef", colorant"#3987e5",
        colorant"#256abf", colorant"#184f95", colorant"#0d366b"])
    D, T = size(runs[1].visits)
    N = sum(runs[1].visits[:, 1])
    year1 = runs[1].visits[:, 1]
    @assert all(r -> r.visits[:, 1] == year1, runs) "runs must share their first year"
    rank in (:first, :last) || throw(ArgumentError("rank must be :first or :last"))
    # Row order: the shared year-1 visitors, or each run's own final popularity.
    ordering(r) = rank === :first ? sortperm(year1; rev = true) : sortperm(r.popularity[:, end]; rev = true)
    rowlabel = rank === :first ? "year-1 visitors" : "its year-$T popularity"
    hilabel = rank === :first ? ("most visited in year 1", "least visited in year 1") :
              ("most popular in year $T", "least popular in year $T")
    logticks = [0, 1, 3, 10, 30, 100, 300, 1000]
    xticks = [1, 25, 50, 75, 100]
    ymax = 1.05 * maximum(maximum(N .* r.popularity) for r in runs)

    fig = Figure(size = (1350, 900))
    local hm
    for (j, (r, col)) in enumerate(zip(runs, columns))
        order = ordering(r)
        top, bottom = order[1:N_HIGHLIGHT], order[end-N_HIGHLIGHT+1:end]
        title = col.ε == 0 ? "ε = 0: no influence" : join(("$k = $(fmt(v))" for (k, v) in pairs(col)), ", ")
        haskey(col, :β) && col.β == 1 && (title *= " (no conformity)")
        ax = Axis(fig[1, j]; title, xlabel = "year", xticks,
            ylabel = j == 1 ? "destination, ranked by $rowlabel" : "",
            yreversed = true, yticks = [1, 25, 50, 75, 100])
        hm = heatmap!(ax, 1:T, 1:D, permutedims(r.visits[order, :]);
            colormap = ramp, colorrange = (0, vmax), colorscale = scale, highclip = colorant"#0d366b")
        j > 1 && hideydecorations!(ax; ticks = false, ticklabels = false)

        final = r.popularity[:, end]
        lock_in = cor(year1, final)
        visited = count(>(0), r.visits[:, end])
        ax2 = Axis(fig[2, j]; xlabel = "year", xticks, yscale = scale,
            yticks = scale === identity ? Makie.automatic : logticks,
            ylabel = j == 1 ? "expected visitors per year" : "",
            subtitle = "year $T: $(round(effective_number(final), digits = 1)) effective destinations, " *
                       "$visited visited\ncorr. of year-1 visitors with year-$T popularity: $(round(lock_in, digits = 2))",
            subtitlecolor = muted)
        pop = N .* r.popularity
        for d = 1:D
            (d in top || d in bottom) && continue
            lines!(ax2, 1:T, pop[d, :]; color = others, linewidth = 0.8)
        end
        for d in bottom
            lines!(ax2, 1:T, pop[d, :]; color = unlucky, linewidth = 2)
        end
        for d in top
            lines!(ax2, 1:T, pop[d, :]; color = lucky, linewidth = 2)
        end
        hlines!(ax2, [N / D]; color = ink, linewidth = 1, linestyle = :dot)
        limits!(ax2, 0.5, T + 0.5, 0, ymax)
        j > 1 && hideydecorations!(ax2; ticks = false, ticklabels = false, grid = false)
    end
    cticks = scale === identity ? Makie.automatic : logticks
    Colorbar(fig[1, length(runs)+1], hm; label = "visitors that year", ticks = cticks)
    Legend(fig[3, 1:length(runs)],
        [LineElement(color = lucky, linewidth = 2), LineElement(color = unlucky, linewidth = 2),
            LineElement(color = others, linewidth = 1.5), LineElement(color = ink, linestyle = :dot)],
        ["the $N_HIGHLIGHT destinations $(hilabel[1])", "the $N_HIGHLIGHT $(hilabel[2])",
            "the other destinations", "even split, N / D = $(N ÷ D)"];
        orientation = :horizontal, framevisible = false)
    title = rule === :additive ? rich("additive rule, w ← w + ε Σ", subscript("n′≠n"), " d") :
            rule === :mixing ? rich("mixing rule, w ← (1 − ε) w + (ε/N) Σ", subscript("n′"), " d") :
            rich("conformity, w ← (1 − ε) w + ε s with s", subscript("d"), " ∝ c", subscript("d"), superscript("β"))
    Label(fig[-1, :], rich("Where $N tourists go each year among $D destinations, ", title); fontsize = 17)
    Label(fig[0, :], "seed $seed: same tastes and same first year in every column"; color = muted)
    rowsize!(fig.layout, 2, Relative(0.32))
    set_theme!()
    return fig
end

for (rule, f) in pairs(FIGURES)
    runs = [history(initialize(; col..., update = f.update, seed = SEED), YEARS) for col in f.columns]
    FigureProvenance.save(fig_time_evolution(runs, f.columns, SEED; rule, f.style...), f.out)
    println("Wrote $(f.out)")

    N = sum(runs[1].visits[:, 1])
    for (r, col) in zip(runs, f.columns)
        final = r.popularity[:, end]   # going into the last year, after YEARS − 1 updates
        ε, β = col.ε, get(col, :β, 1.0)
        pred = f.update === :additive ? polya_prediction(ε, YEARS - 1).effective_number :
               β == 1 ? drift_prediction(ε, YEARS - 1) : NaN   # no closed form once conformity has concentrated it
        println("  $rule, $col: effective number of destinations in year $YEARS ",
            round(effective_number(final), digits = 1), " (mean-field ", round(pred, digits = 1), "), most popular ",
            round(N * maximum(final), digits = 1), " visitors/year, least ", round(N * minimum(final), digits = 1),
            "; destinations visited in year $YEARS: ", count(>(0), r.visits[:, end]))
    end
end
