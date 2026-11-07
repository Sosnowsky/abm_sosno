# The conformity run as a figure for the garden's note "Why did we choose Taiwan?": the expected visitors per year
# of each of the 100 destinations over 200 years at ε = 0.1, β = 1.3. It is the bottom row of
# plots/tourist_destinations_conformity.png with less on it, written as an SVG in the garden's colours that follows
# the reader's light or dark theme.
#
# 200 years rather than 100: at ε(β − 1) = 0.03 the tourists concentrate after about 115 years
# (scripts/tourist_conformity_sweep.jl), so 100 years would stop mid-way.
#
# The same run is also written as an interactive Plotly figure (plots/tourist_conformity_garden.plotly.json), for a
# ```plot block in the note: zoom, hover for each destination's numbers, hide groups of lines from the legend, and
# switch the y axis between log and linear.
#
# Run with:  julia --project=. scripts/tourist_garden_figure.jl [seed]

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "TouristDestinations.jl"))
using .TouristDestinations
using Agents, CairoMakie, FigureProvenance, JSON

const SEED = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1
const ε, β = 0.1, 1.3
const YEARS = 200
const N_RUNNERS_UP = 4
const OUT = joinpath(@__DIR__, "..", "plots", "tourist_conformity_garden.svg")
const OUT_PLOTLY = joinpath(@__DIR__, "..", "plots", "tourist_conformity_garden.plotly.json")
# The garden's colour tokens (sosno-garden, assets/garden.css), so the figure reads as part of the page.
const THEMES = (
    light = (ink = colorant"#18221d", muted = colorant"#5b6a61", rule = colorant"#cdd6cf",
        winner = colorant"#b8432c", runners = colorant"#2f6a4c"),
    dark = (ink = colorant"#e2e9e3", muted = colorant"#9aaba0", rule = colorant"#2e3a33",
        winner = colorant"#e0694f", runners = colorant"#7cc3a0"),
)

"""
A figure drawn once per theme, the same drawing in two sets of colours. It is written as the light drawing plus a
style that, when the reader's `prefers-color-scheme` is dark, paints each light colour in its dark counterpart (a
CSS rule overrides an SVG colour attribute). That works for an SVG put in a page as a plain picture, which cannot see
the page's colours, and keeps the geometry, nearly all of the file, once.
"""
mutable struct ThemedSVG
    light::Figure
    dark::Figure
end

function FigureProvenance.write_figure(f::ThemedSVG, path; kwargs...)
    endswith(path, ".svg") || throw(ArgumentError("a ThemedSVG is written as .svg, not $(basename(path))"))
    function svg(fig)
        tmp = tempname() * ".svg"
        save(tmp, fig; kwargs...)
        s = read(tmp, String)
        rm(tmp)
        return s
    end
    light, dark = svg(f.light), svg(f.dark)
    # The drawings differ only in colour, so their colour attributes pair up in order: light colour → dark colour.
    paint = r"\b(fill|stroke)=\"(rgb\([^)]*\))\""
    l, d = collect(eachmatch(paint, light)), collect(eachmatch(paint, dark))
    length(l) == length(d) || error("the light and dark drawings differ in more than colour")
    swap = Dict{Tuple{String,String},String}()
    for (a, b) in zip(l, d)
        a[1] == b[1] || error("the light and dark drawings differ in more than colour")
        get!(swap, (a[1], a[2]), b[2]) == b[2] || error("light $(a[1]) $(a[2]) has two dark counterparts")
    end
    css = join(("[$attr=\"$c\"]{$attr:$dk}" for ((attr, c), dk) in sort!(collect(swap)) if c != dk))
    light = replace(light, r"(<svg[^>]*>)" => SubstitutionString("\\1\n<style>@media (prefers-color-scheme: dark){$css}</style>");
        count = 1)
    # Cairo prints coordinates to 6 decimals, most of the file; 0.1 px is plenty for the drawing. The glyph outlines
    # (in <defs>) keep theirs.
    defs, drawing = split(light, "</defs>"; limit = 2)
    tenth(x) = (v = round(parse(Float64, x), digits = 1); isinteger(v) ? string(Int(v)) : string(v))
    drawing = replace(drawing, r" d=\"[^\"]*\"" => p -> replace(p, r"-?\d+\.\d+" => tenth, r"([MLC]) " => s"\1"))
    write(path, defs * "</defs>" * drawing)
    return path
end

"One theme's drawing of `fig_garden`, in the colours `c`."
function draw_popularity(h, c; ε, β)
    D, T = size(h.popularity)
    N = sum(h.visits[:, 1])
    pop = N .* h.popularity
    winner = argmax(pop[:, end])
    peak = vec(maximum(pop, dims = 2))
    peak[winner] = -Inf
    runners = partialsortperm(peak, 1:N_RUNNERS_UP; rev = true)
    others = setdiff(1:D, winner, runners)
    faint = RGBAf(c.muted, 0.35)

    fig = Figure(size = (700, 430), fontsize = 17, backgroundcolor = :transparent, figure_padding = (4, 14, 4, 4))
    ax = Axis(fig[3, 1]; backgroundcolor = :transparent, yscale = Makie.pseudolog10,
        yticks = [0, 1, 3, 10, 30, 100, 300, 1000], xticks = 0:50:T, xlabel = "year", ylabel = "expected visitors per year",
        xgridcolor = c.rule, ygridcolor = c.rule, xgridwidth = 1, ygridwidth = 1,
        leftspinecolor = c.rule, bottomspinecolor = c.rule, topspinevisible = false, rightspinevisible = false,
        xtickcolor = c.rule, ytickcolor = c.rule, xticklabelcolor = c.muted, yticklabelcolor = c.muted,
        xlabelcolor = c.ink, ylabelcolor = c.ink)
    for d in others
        lines!(ax, 1:T, pop[d, :]; color = faint, linewidth = 0.8)
    end
    for d in runners
        lines!(ax, 1:T, pop[d, :]; color = c.runners, linewidth = 2)
    end
    lines!(ax, 1:T, pop[winner, :]; color = c.winner, linewidth = 3)
    hlines!(ax, [N / D]; color = c.ink, linestyle = :dot, linewidth = 1.3)
    limits!(ax, 0, T, 0, 1.15N)

    Legend(fig[2, 1],
        [LineElement(color = c.winner, linewidth = 3), LineElement(color = c.runners, linewidth = 2),
            LineElement(color = faint, linewidth = 1.5), LineElement(color = c.ink, linestyle = :dot, linewidth = 1.3)],
        ["where everybody ends up", "the $N_RUNNERS_UP runners-up", "the other $(length(others))",
            "even split, $(N ÷ D) per destination"];
        orientation = :horizontal, nbanks = 2, framevisible = false, labelcolor = c.ink, halign = :left,
        tellwidth = false, colgap = 18, patchsize = (26, 10))
    Label(fig[1, 1], "$N tourists, $D destinations, influence ε = $ε, conformity β = $β"; color = c.muted,
        fontsize = 15, halign = :left, tellwidth = false)
    rowgap!(fig.layout, 1, 2)
    return fig
end

"""
Popularity under conformity for the garden's note "Why did we choose Taiwan?": each of the 100 destinations'
expected visitors per year (popularity × N) over 200 years at ε = 0.1, β = 1.3, from one seed. The destination that
everybody ends up at is drawn in the accent colour, the 4 others whose popularity peaked highest (runners-up that
grow for a while and then collapse) in the second colour, the rest faint, with the even split (N/D = 10) dotted.
The y axis is pseudo-logarithmic, so both the early spread around 10 and the winner's 1000 can be read. It is the
bottom row of `fig_time_evolution` with less on it (no ranking by the final year, no statistics), 200 years long
because concentration takes about 115 years at ε(β − 1) = 0.03, and in the garden's colours, drawn for its light
and its dark theme.
"""
@provenance function fig_garden(h; ε, β, themes = THEMES)
    return ThemedSVG(draw_popularity(h, themes.light; ε, β), draw_popularity(h, themes.dark; ε, β))
end

"A Plotly figure: its traces and layout, written as the JSON that Plotly's `newPlot` and the garden's ```plot blocks read."
mutable struct PlotlyFigure  # mutable: FigureProvenance tags the object to find its docstring
    data::Vector{Dict{String,Any}}
    layout::Dict{String,Any}
end

function FigureProvenance.write_figure(f::PlotlyFigure, path; kwargs...)
    endswith(path, ".json") || throw(ArgumentError("a PlotlyFigure is written as .json, not $(basename(path))"))
    write(path, JSON.json(Dict("data" => f.data, "layout" => f.layout)))
    return path
end

"""
The garden figure (`fig_garden`) as an interactive Plotly figure for the note "Why did we choose Taiwan?": the same
run, lines and colours, now with a tooltip naming each destination and its expected visitors, zoom, legend entries
that hide or show a whole group (the winner, the 4 runners-up, the other 95, the even split), and buttons (top left, clear of Plotly's
toolbar at the top right) switching the y axis between log (as in the SVG, where 0 cannot be shown: it starts at 0.3) and linear. Colours are written as
the garden's tokens (`var(--accent)`, …), which the page resolves for its light or dark theme; the run's parameters go in `layout.meta.caption`, which the garden shows as the caption; numbers are rounded
to 0.01 visitors to keep the file small.
"""
@provenance function fig_garden_plotly(h; ε, β)
    D, T = size(h.popularity)
    N = sum(h.visits[:, 1])
    pop = N .* h.popularity
    winner = argmax(pop[:, end])
    peak = vec(maximum(pop, dims = 2))
    peak[winner] = -Inf
    runners = partialsortperm(peak, 1:N_RUNNERS_UP; rev = true)
    others = setdiff(1:D, winner, runners)
    years = collect(1:T)
    hover = "destination %{meta}<br>year %{x}: %{y:.1f} visitors<extra></extra>"
    line(d, group, name, color, width, first; opacity = 1.0) = Dict{String,Any}(
        "type" => "scatter", "mode" => "lines", "x" => years, "y" => round.(pop[d, :], digits = 2),
        "name" => name, "legendgroup" => group, "showlegend" => first, "meta" => d, "hovertemplate" => hover,
        "opacity" => opacity, "line" => Dict("color" => color, "width" => width))
    data = Dict{String,Any}[]
    for (i, d) in enumerate(others)
        push!(data, line(d, "others", "the other $(length(others))", "var(--muted)", 0.8, i == 1; opacity = 0.35))
    end
    for (i, d) in enumerate(runners)
        push!(data, line(d, "runners", "the $N_RUNNERS_UP runners-up", "var(--moss)", 2, i == 1))
    end
    push!(data, line(winner, "winner", "where everybody ends up", "var(--accent)", 3, true))
    push!(data, Dict{String,Any}("type" => "scatter", "mode" => "lines", "x" => [1, T], "y" => [N / D, N / D],
        "name" => "even split, $(N ÷ D) per destination", "hoverinfo" => "skip",
        "line" => Dict("color" => "var(--ink)", "width" => 1.3, "dash" => "dot")))
    logrange = [log10(0.3), log10(1.15N)]
    button(label, args) = Dict("label" => label, "method" => "relayout", "args" => [args])
    layout = Dict{String,Any}(
        "hovermode" => "closest",
        "margin" => Dict("t" => 40),
        "xaxis" => Dict("title" => Dict("text" => "year"), "range" => [0, T]),
        "yaxis" => Dict("title" => Dict("text" => "expected visitors per year"), "type" => "log", "range" => logrange,
            "tickvals" => [1, 3, 10, 30, 100, 300, 1000]),
        # at the bottom of the whole figure, not of the plot, so it never runs into the axis title when it wraps on a phone
        "legend" => Dict("orientation" => "h", "x" => 0, "xanchor" => "left", "yref" => "container", "y" => 0,
            "yanchor" => "bottom"),
        # shown by the garden as the figure's caption, which wraps on narrow screens (a Plotly title does not)
        "meta" => Dict("caption" => "$N tourists, $D destinations, influence ε = $ε, conformity β = $β."),
        "updatemenus" => [Dict("type" => "buttons", "direction" => "right", "showactive" => true, "active" => 0,
            "x" => 0, "xanchor" => "left", "y" => 1.03, "yanchor" => "bottom", "pad" => Dict("r" => 0, "t" => 0),
            "bgcolor" => "var(--panel)", "bordercolor" => "var(--rule)", "font" => Dict("color" => "var(--ink)"),
            "buttons" => [button("log", Dict("yaxis.type" => "log", "yaxis.range" => logrange)),
                button("linear", Dict("yaxis.type" => "linear", "yaxis.range" => [0, 1.15N]))])])
    return PlotlyFigure(data, layout)
end

h = history(initialize(; ε, β, update = :mixing, seed = SEED), YEARS)
FigureProvenance.save(fig_garden(h; ε, β), OUT)
FigureProvenance.save(fig_garden_plotly(h; ε, β), OUT_PLOTLY)
println("Wrote $OUT_PLOTLY ($(round(filesize(OUT_PLOTLY) / 1024, digits = 1)) KB)")
println("Wrote $OUT ($(round(filesize(OUT) / 1024, digits = 1)) KB)")
N = sum(h.visits[:, 1])
eff = [effective_number(h.popularity[:, t]) for t = 1:YEARS]
println("seed $SEED: fewer than 2 effective destinations from year ", something(findfirst(<(2), eff), "never"),
    "; winner's expected visitors in year $YEARS: ", round(N * maximum(h.popularity[:, end]), digits = 1),
    "; highest runner-up peak: ", round(N * sort(vec(maximum(h.popularity, dims = 2)); rev = true)[2], digits = 1))
