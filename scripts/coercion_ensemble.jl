# Ensemble for the coercion-resentment model: does the simplest agent-based
# version of Schøyen (2021) reproduce the stylized facts of Schøyen, Sirnes and
# Andersen (2026)? Each run is one country; the pooled country-years after a
# burn-in play the role of their ~100k Polity/FH/V-Dem observations.
#
#   (a) Validation: with coercion frozen, the families settle at the paper's
#       steady state q*(π), inefficient window included.
#   (b) Hypothesis 2: the distribution of coercion is bimodal, with a gap where
#       the window is.
#   (c) Hypothesis 1: legitimacy 1 − q is non-linear in coercion, lower in the
#       middle than on either side.
#   (d) Hypothesis 3: regimes collapse most often at middle coercion.
#   (e) Table 4: coercion over the 15 years before a crackdown (a rise of more
#       than 1 point in a year, from an already high level) against random years.
#   (f) Path dependence (2021, Sect. 2.2): identical regimes end at low or high
#       coercion depending on where they start.
#
# Every result panel also shows a control with the same regimes and families but
# a resentment too weak to make a window (max C′ = 0.78 < 2), so the panels say
# which facts need the window and which come from the regime rule alone.
#
# Run with:  julia --project=. -t auto scripts/coercion_ensemble.jl

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "CoercionResentment.jl"))
using .CoercionResentment
using Agents, CairoMakie, Statistics, Random, FigureProvenance

# --- Ensemble configuration ---
const N_COUNTRIES = 200
const YEARS = 1500
const BURN_IN = 200                   # years dropped from every country
const CONTROL = (; resentment_steepness = 0.3)
const HIGH = 5.0                      # crackdowns start from coercion ≥ HIGH
const LAGS = 0:15
const OUT = joinpath(@__DIR__, "..", "plots", "coercion_ensemble.png")

"Run `N_COUNTRIES` countries with seeds `1:N_COUNTRIES`, in parallel."
function run_countries(; kwargs...)
    H = Vector{Any}(undef, N_COUNTRIES)
    Threads.@threads for s = 1:N_COUNTRIES
        H[s] = history(initialize(; seed = s, kwargs...), YEARS)
    end
    return H
end

"One field of every country after the burn-in, concatenated."
pool(H, field) = reduce(vcat, [getfield(h, field)[BURN_IN+1:end] for h in H])

"Mean of `y` in each unit bin of `x` on the 0–10 scale, `NaN` where empty."
function binned(x, y)
    return [begin
        inbin = (e .<= x .< e + 1) .| ((e == 9) .& (x .== 10))
        any(inbin) ? mean(y[inbin]) : NaN
    end for e = 0:9]
end

"""
    crackdown_profile(H; seed) -> (crackdown, random, n)

Mean coercion `s` years before each crackdown, `s ∈ LAGS`, where a crackdown is
a rise of more than 1 in a year from a coercion of at least `HIGH`; `s = 0` is the
crackdown year itself, as in the paper's Table 4. The null is
the same average over random years whose previous year also had coercion of at
least `HIGH`, so the two differ only in what came next.
"""
function crackdown_profile(H; seed = 1)
    window(x, t) = [x[t-s] for s in LAGS]
    events, anchors = Vector{Float64}[], Tuple{Int,Int}[]
    for (i, h) in enumerate(H), t = (BURN_IN+last(LAGS)+1):YEARS
        x = h.coercion
        x[t-1] >= HIGH || continue
        push!(anchors, (i, t))
        x[t] - x[t-1] > 1 && push!(events, window(x, t))
    end
    rng = MersenneTwister(seed)
    random = [window(H[i].coercion, t) for (i, t) in rand(rng, anchors, 20_000)]
    return (crackdown = mean(events), random = mean(random), n = length(events))
end

"Long-run minority share with coercion frozen at each `π`."
function validation(πs; n_families = 2000, years = 1500)
    q = zeros(length(πs))
    Threads.@threads for i in eachindex(πs)
        m = initialize(; π0 = πs[i], ease = 0, crackdown = 0, collapse_rate = 0, n_families, seed = i)
        q[i] = mean(history(m, years).q[years÷2:end])
    end
    return q
end

"Final coercion of regimes with one tolerance, started at each `π0`, never collapsing."
function path_dependence(π0s; tolerance = 0.3, years = 1500)
    π = zeros(length(π0s))
    Threads.@threads for i in eachindex(π0s)
        m = initialize(; π0 = π0s[i], tolerance = (tolerance, tolerance), collapse_rate = 0, seed = i)
        π[i] = mean(history(m, years).coercion[end-199:end])
    end
    return π
end

"""
Six panels testing the simplest ABM of Schøyen (2021) against the stylized facts of
Schøyen, Sirnes and Andersen (2026): (a) the families reproduce q*(π) with coercion frozen;
(b) coercion is bimodal with a gap at the window; (c) legitimacy dips at middle coercion;
(d) collapse peaks at middle coercion; (e) coercion falls in the years before a crackdown;
(f) identical regimes end low or high depending on their start. Panels b–e overlay a
no-window control (max C' < 2), which is what tells the facts that need the window
(b, c) from those the regime rule produces anyway (d, e).
"""
@provenance function fig_stylized_facts(res, ctl, val, path, p)
    set_theme!(fontsize = 13)
    fig = Figure(size = (1350, 820))
    blue, grey = RGBf(0.15, 0.35, 0.62), RGBf(0.6, 0.6, 0.63)
    window = RGBAf(0.91, 0.64, 0.24, 0.18)
    πs = range(0, 10, length = 401)
    qstar = [steady_state(π, p) for π in πs]
    # The inefficient window: from the first minimum of q* to where it returns there.
    i_min = argmin(qstar[πs.<p.resentment_threshold])
    i_peak = i_min + argmax(qstar[i_min:end]) - 1
    i_end = i_peak + findfirst(<=(qstar[i_min]), qstar[i_peak:end]) - 1
    shade!(ax) = vspan!(ax, πs[i_min], πs[i_end]; color = window)
    centres = 0.5:1:9.5

    ax = Axis(fig[1, 1]; xlabel = "coercion π", ylabel = "minority share q",
        title = "(a) Families reproduce q*(π)")
    shade!(ax)
    lines!(ax, πs, qstar; color = :black, label = "q*(π), Schøyen (2021)")
    scatter!(ax, val.π, val.q; color = blue, label = "ABM, π frozen")
    axislegend(ax; position = :rt, framevisible = false, labelsize = 11)

    ax = Axis(fig[1, 2]; xlabel = "coercion π", ylabel = "share of country-years",
        title = "(b) Coercion is bimodal (H2)")
    shade!(ax)
    for (H, c, l) in ((ctl, grey, "no window"), (res, blue, "S-shaped resentment"))
        x = pool(H, :coercion)
        stephist!(ax, x; bins = 0:0.5:10, normalization = :probability, color = c, linewidth = 2, label = l)
    end
    axislegend(ax; position = :ct, framevisible = false, labelsize = 11)

    ax = Axis(fig[1, 3]; xlabel = "coercion π", ylabel = "mean legitimacy 1 − q",
        title = "(c) Legitimacy dips in the middle (H1)")
    shade!(ax)
    for (H, c) in ((ctl, grey), (res, blue))
        scatterlines!(ax, centres, binned(pool(H, :coercion), pool(H, :legitimacy)); color = c)
    end
    lines!(ax, πs, 1 .- qstar; color = :black, linestyle = :dash, label = "1 − q*(π)")
    axislegend(ax; position = :lt, framevisible = false, labelsize = 11)

    ax = Axis(fig[2, 1]; xlabel = "coercion π", ylabel = "collapses per regime-year",
        title = "(d) Collapse peaks in the middle (H3)")
    shade!(ax)
    for (H, c) in ((ctl, grey), (res, blue))
        scatterlines!(ax, centres, binned(pool(H, :coercion), Float64.(pool(H, :collapse))); color = c)
    end

    ax = Axis(fig[2, 2]; xlabel = "years before the crackdown", ylabel = "mean coercion",
        title = "(e) Easing precedes crackdowns (Table 4)", xreversed = true)
    for (prof, c, l) in ((crackdown_profile(ctl), grey, "no window"), (crackdown_profile(res), blue, "S-shaped"))
        scatterlines!(ax, LAGS, prof.crackdown; color = c, label = "$l: before $(prof.n) crackdowns")
        lines!(ax, LAGS, prof.random; color = c, linestyle = :dash, label = "$l: random high years")
    end
    axislegend(ax; position = :lb, framevisible = false, labelsize = 10)

    ax = Axis(fig[2, 3]; xlabel = "initial coercion π₀", ylabel = "long-run coercion",
        title = "(f) Path dependence, tolerance 0.3")
    shade!(ax)
    scatter!(ax, path.π0, path.π; color = blue)
    lines!(ax, [0, 10], [0, 10]; color = grey, linestyle = :dot)

    Label(fig[0, :],
        "Coercion-resentment ABM: $N_COUNTRIES countries × $(YEARS - BURN_IN) years " *
        "(shaded: the inefficient window of q*)"; fontsize = 17)
    set_theme!()
    return fig
end

# --- Run ---
println("Running $N_COUNTRIES countries × $YEARS years, twice, on $(Threads.nthreads()) threads ...")
res = run_countries()
ctl = run_countries(; CONTROL...)
println("Validation and path dependence ...")
val_π = 0:0.5:10
val = (π = val_π, q = validation(val_π))
path_π0 = 0:0.25:10
path = (π0 = path_π0, π = path_dependence(path_π0))
p = initialize(; n_families = 1)   # only its parameters are used, for q*

FigureProvenance.save(fig_stylized_facts(res, ctl, val, path, p), OUT)
println("Wrote $OUT")

# --- Numbers behind the figure ---
for (name, H) in (("S-shaped resentment", res), ("no window", ctl))
    x = pool(H, :coercion)
    println("\n$name")
    println("  share of country-years with 3 ≤ π < 7: ", round(mean(3 .<= x .< 7), digits = 3))
    println("  legitimacy by unit bin:  ", round.(binned(x, pool(H, :legitimacy)), digits = 3))
    println("  collapses per year:      ", round.(binned(x, Float64.(pool(H, :collapse))), digits = 4))
    prof = crackdown_profile(H)
    println("  crackdowns: ", prof.n, ";  coercion 15, 2, 1, 0 years before: ",
        round.(prof.crackdown[[16, 3, 2, 1]], digits = 2), " vs random ",
        round.(prof.random[[16, 3, 2, 1]], digits = 2))
end
println("\nValidation, max |q_ABM − q*|: ", round(maximum(abs.(val.q .- [steady_state(π, p) for π in val_π])), digits = 3))
