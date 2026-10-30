"""
    CoercionResentment

Model logic for the coercion-resentment ABM described in
`models/coercion_resentment.md`, an agent-based version of Schøyen (2021,
Cliometrica) built to reproduce the stylized facts of Schøyen, Sirnes and
Andersen (2026, Public Choice).

Notation follows Schøyen (2021). The 2026 paper's coercion `C ∈ [0, 1]` is
`π / π_max` here; its legitimacy is `L = 1 − q`.

One model is one country. Its **families** pass an identity — the non-state
identity *ns* or the state identity *s* — to their children, Bisin–Verdier style:
a parent socialises the child into its own identity with probability `τ`, and if
that fails the child copies a random member of the population. With the
quadratic cost `H(τ) = τ²/2k`, the paper's first-order conditions (4)–(5),

    (1 − q)(Δu − (π − C(π))) = H′(τ^ns)     ns parent: the penalty π hurts, resentment C helps
    q (Δu + π)               = H′(τ^s)      s parent,

give `τ^ns = k (1 − q) g_ns` and `τ^s = k q g_s` with the gains
`g_ns = Δu − π + C(π)` and `g_s = Δu + π`. The share `q` of ns parents relaxes,
over generations, to the paper's steady state

    q*(π) = (Δu − π + C(π)) / (2Δu + C(π)).

With an S-shaped resentment `C`, `q*` falls to its minimum at the
non-confrontational level `π_q̲`, rises to its maximum at `π_q̄`, and falls again,
back to `q*(π_q̲)` at `π^e_q̲`. On the inefficient interval `(π_q̲, π^e_q̲)` more
coercion does not shrink the ns group.

One **authority** sets the coercion `π` each year by one of four rules (`rule`):

- `:satisficing` (the default, the model behind the 2026 stylized facts): no
  optimisation. The authority judges its policy by whether the ns group is
  growing, which it reads off the socialisation flow: ns parents investing more
  than s parents. Coercion is costly, so it eases by `ease` a year while the ns
  group is not growing; it holds while the group grows but is within its
  tolerance `q_max`; and it cracks down by `crackdown` when the group is both
  above `q_max` and growing. It falls each year with a probability proportional
  to the resentful ns group, `q·C(π)`, and is replaced by an authority with a
  fresh `q_max` that inherits the coercion apparatus.
- `:fixed`: `π` never changes (Schøyen 2021, Lemma 1: `q` settles at `q*(π)`).
- `:exogenous`: the paper's authority, which knows `q*` and takes the `π ≤ ρ`
  with the lowest `q*(π)`, `π^EC`, under an exogenous constraint `ρ` (the theorem
  of restraint).
- `:insurrection`: the same, under the endogenous insurrection constraint `ρ(q)`
  of the paper's Sect. 2.2 (path dependence and coercion dependence).

The three 2021 rules have no `q_max`, no collapse and no turnover of authorities.

Time steps are years. Families reproduce with probability `1/generation` a year,
so `q` is a slow stock while coercion is a fast flow.

The module is plotting-agnostic. All tunable parameters are `initialize` keyword
arguments; the yearly state of the authority and of the population lives in
`Ref`s in the model's `properties`.
"""
module CoercionResentment

using Agents, Random

export Family,
    initialize,
    family_step!,
    authority_step!,
    resentment,
    gains,
    steady_state,
    ns_share,
    insurrection_constraint,
    best_coercion,
    history

# --- The agent ----------------------------------------------------------------

"""
    Family

A household, replaced by its child when it reproduces. `ns` is its identity:
`true` for the non-state identity the authority penalises, `false` for *s*. Every other
quantity a family needs (its gain, hence its effort) depends only on its identity
and on the country-wide `π` and `q`, so it is computed once a year in
[`authority_step!`](@ref) rather than stored here.
"""
@agent struct Family(NoSpaceAgent)
    ns::Bool
end

# --- The payoffs --------------------------------------------------------------

"""
    resentment(π, model_or_params) -> Float64

The S-shaped coercion resentment `C(π) = A [atan(s(π − π̂)) + atan(s π̂)]`:
zero at `π = 0`, convex below its inflection point `π̂` and concave above,
saturating at `A(π/2 + atan(s π̂))`. It has to be steeper than 2 somewhere for an
inefficient interval to exist (Schøyen 2021, Assumption 3); `A = 0` turns
resentment off.
"""
resentment(π, p) = p.resentment_scale * (atan(p.resentment_steepness * (π - p.π̂)) +
                                         atan(p.resentment_steepness * p.π̂))

"""
    gains(π, p) -> (g_ns, g_s)

What an ns and an s parent gain from a child with their own identity rather
than the other one: `g_ns = Δu − π + C(π)` and `g_s = Δu + π`, where
`Δu = ū − u̲`. The levels `ū`, `u̲` cancel; only `Δu`, the penalty `π` and the
resentment `C(π)` remain.
"""
function gains(π, p)
    C = resentment(π, p)
    return (p.Δu - π + C, p.Δu + π)
end

"""
    steady_state(π, p) -> Float64

The paper's steady-state ns share `q*(π) = g_ns / (g_ns + g_s)`, which solves
`(1 − q) g_ns = q g_s`: the two efforts balance, `τ^ns = τ^s`. Zero once the ns
gain is negative, i.e. once even ns parents prefer an s child.
"""
function steady_state(π, p)
    g_ns, g_s = gains(π, p)
    return clamp(g_ns / (g_ns + g_s), 0.0, 1.0)
end

"Socialisation efforts `(τ^ns, τ^s)` of an ns and an s parent, clipped to `[0, 1]`."
function efforts(q, π, p)
    g_ns, g_s = gains(π, p)
    return (clamp(p.effort * (1 - q) * g_ns, 0.0, 1.0), clamp(p.effort * q * g_s, 0.0, 1.0))
end

# --- The submodels ------------------------------------------------------------

"""
    family_step!(family, model)

With probability `1/generation` the family is replaced by its child. The parent
socialises the child into its own identity with probability `τ`; otherwise the
child copies a random member of the population, which is an ns member with
probability `q`. Using last year's `q` and efforts keeps the year synchronous:
which family reproduces first does not matter.
"""
function family_step!(family::Family, model::StandardABM)
    rng = abmrng(model)
    rand(rng) < 1 / model.generation || return
    τ = family.ns ? model.τ_ns[] : model.τ_s[]
    if rand(rng) >= τ
        family.ns = rand(rng) < model.q[]
    end
    return
end

"The current share `q` of ns families."
ns_share(model::StandardABM) = count(f -> f.ns, allagents(model)) / nagents(model)

"""
    insurrection_constraint(q, p) -> Float64

The insurrection constraint: the most coercion an ns group of share `q`
tolerates before it rises up,
`ρ(q) = ρ_low + (π_max + 0.5 − ρ_low) / (1 + exp((q − ρ_tip)/ρ_width))`. A small
group sets no constraint (`ρ` above the scale), a large one tolerates `ρ_low`, and
the drop is centred at `ρ_tip` with width about `4ρ_width`. Schøyen (2021) only
assumes it falls (Assumption 4), and writes it as `ρ(q*(π))`; here it reads the
current `q`. The logistic is this model's choice.
"""
insurrection_constraint(q, p) = p.ρ_low +
    (p.π_max + 0.5 - p.ρ_low) / (1 + exp((q - p.ρ_tip) / p.ρ_width))

"""
    best_coercion(ρ, p) -> Float64

The 2021 paper's choice `π^EC`: the `π ≤ ρ` with the lowest `q*(π)`, from the
running minimum of `q*` tabulated on a grid of `π_max/1000` at construction.
For `ρ` inside the inefficient interval this is `π_q̲`, whatever `ρ` is.
"""
function best_coercion(ρ, p)
    grid = p.best_π
    i = floor(Int, clamp(ρ, 0, p.π_max) / p.π_max * (length(grid) - 1)) + 1
    return grid[clamp(i, 1, length(grid))]
end

"The running minimum of `q*` over `[0, π]` on a grid: entry `i` is the best `π ≤ grid[i]`."
function best_coercion_table(p; n = 1001)
    πs = range(0, p.π_max, length = n)
    best = similar(collect(πs))
    bi, bq = 1, steady_state(πs[1], p)
    for i in eachindex(πs)
        v = steady_state(πs[i], p)
        if v < bq - 1e-12
            bi, bq = i, v
        end
        best[i] = πs[bi]
    end
    return best
end

"Draw a new authority's tolerance `q_max`, the ns share above which it cracks down."
new_q_max(rng, p) = p.q_max[1] + (p.q_max[2] - p.q_max[1]) * rand(rng)

"""
    authority_step!(model)

Runs after every family has reproduced. Under `rule = :fixed` it only records
`q`; under `:exogenous` it sets `π = best_coercion(ρ)`, and under
`:insurrection` `π = best_coercion(ρ(q))`. Under `:satisficing`, in order:

1. Observe the ns share `q`, and whether the ns group is growing under the
   current coercion: `τ^ns > τ^s`, since `E[Δq] ∝ q(1 − q)(τ^ns − τ^s)`. This is
   the flow without its sampling noise, as a census of how children are raised
   would show it, so the authority reacts to its own policy without the
   generational lag of the stock `q`.
2. If the ns group is not growing, ease (`π -= ease`). If it is growing and
   exceeds the tolerance `q_max`, crack down (`π += crackdown`). Otherwise
   hold. On the rising part of `q*`, `(π_q̲, π_q̄)`, a crackdown makes the ns
   group grow faster, so an alarmed authority there escalates year after year
   until it is through.
3. Collapse with probability `collapse_rate · q · C(π) / C(π_max)`: the size of
   the resentful ns group, relative to its largest possible value. The new
   authority keeps `π` and draws a new `q_max`.
4. Compute next year's efforts from the new `π` and `q`.
"""
function authority_step!(model::StandardABM)
    rng = abmrng(model)
    q = ns_share(model)
    model.q[] = q
    π = model.coercion[]
    model.cracked_down[] = false
    model.collapsed[] = false

    if model.rule === :satisficing
        τ_ns, τ_s = efforts(q, π, model)
        if τ_ns <= τ_s
            π -= model.ease
        elseif q > model.q_max_now[]
            π += model.crackdown
            model.cracked_down[] = true
        end
        π = clamp(π, 0.0, model.π_max)
        # Drawn every year, even with collapse_rate = 0, so that runs stay
        # reproducible from the seed whatever the rate.
        if rand(rng) < model.collapse_rate * q * resentment(π, model) / resentment(model.π_max, model)
            model.collapsed[] = true
            model.authority[] += 1
            model.q_max_now[] = new_q_max(rng, model)
        end
    elseif model.rule === :exogenous
        π = best_coercion(model.ρ, model)
    elseif model.rule === :insurrection
        π = best_coercion(insurrection_constraint(q, model), model)
    end
    model.coercion[] = π

    model.τ_ns[], model.τ_s[] = efforts(q, π, model)
    return
end

# --- Construction -------------------------------------------------------------

"""
    initialize(; kwargs...) -> StandardABM

Build one country.

Keyword arguments:
- `n_families`: population size `N`. Sets the finite-population noise in `q`.
- `rule`: how the authority sets `π`: `:satisficing` (default), `:fixed`,
  `:exogenous` or `:insurrection`. See the module docstring.
- `π0`: initial coercion; `nothing` draws it uniformly from `[0, π_max]`. Under
  `:exogenous` and `:insurrection` the authority replaces it with its own choice at once.
- `q0`: initial ns share, assigned at random to `round(q0 N)` families;
  `nothing` starts the families in equilibrium with `π0`, at `q*(π0)`.
- `π_max`: the top of the coercion scale (10, like the paper's indices).
- `Δu`: the loss from a child with the other identity.
- `resentment_scale`, `π̂`, `resentment_steepness`: `A`, the inflection point
  `π̂` and `s` of [`resentment`](@ref).
- `effort`: `k` in the cost `H(τ) = τ²/2k`.
- `generation`: years per generation. One family in `generation` reproduces a year.
- `ease`, `crackdown`, `q_max`, `collapse_rate` (`:satisficing` only): the
  yearly easing while the ns group is not growing; the jump in coercion when it
  grows past the tolerance; the `(lo, hi)` range each authority's tolerance
  `q_max` is drawn from; and the yearly collapse probability at the largest
  possible resentful ns group (`0` keeps one authority forever).
- `ρ` (`:exogenous` only): the exogenous constraint on coercion.
- `ρ_low`, `ρ_tip`, `ρ_width` (`:insurrection` only): the parameters of
  [`insurrection_constraint`](@ref).
- `seed`: RNG seed.

`rule = :fixed` leaves the families alone to relax to `q*(π0)`. The older
`ease = crackdown = collapse_rate = 0` does the same with a different random
stream, since the satisficing rule still draws its collapse test.
"""
function initialize(;
    n_families = 1000,
    q0 = nothing,
    π0 = nothing,
    π_max = 10.0,
    Δu = 5.0,
    resentment_scale = 2.6,
    π̂ = 4.5,
    resentment_steepness = 1.6,
    effort = 0.1,
    generation = 25,
    ease = 0.1,
    crackdown = 1.5,
    q_max = (0.15, 0.55),
    collapse_rate = 0.05,
    rule = :satisficing,
    ρ = 6.0,
    ρ_low = 1.5,
    ρ_tip = 0.21,
    ρ_width = 0.012,
    seed = 1234,
)
    rule in (:satisficing, :fixed, :exogenous, :insurrection) ||
        throw(ArgumentError("rule must be :satisficing, :fixed, :exogenous or :insurrection"))
    q_max[1] <= q_max[2] || throw(ArgumentError("q_max must be (lo, hi) with lo ≤ hi"))
    ρ >= 0 || throw(ArgumentError("ρ must be ≥ 0"))
    ρ_width > 0 || throw(ArgumentError("ρ_width must be > 0"))
    rng = MersenneTwister(seed)
    π = isnothing(π0) ? π_max * rand(rng) : float(π0)
    0 <= π <= π_max || throw(ArgumentError("π0 must be in [0, π_max]"))

    params = (;
        π_max = float(π_max),
        Δu = float(Δu),
        resentment_scale = float(resentment_scale),
        π̂ = float(π̂),
        resentment_steepness = float(resentment_steepness),
        effort = float(effort),
        generation = generation,
        ease = float(ease),
        crackdown = float(crackdown),
        q_max = float.(q_max),
        collapse_rate = float(collapse_rate),
        rule = rule,
        ρ = float(ρ),
        ρ_low = float(ρ_low),
        ρ_tip = float(ρ_tip),
        ρ_width = float(ρ_width),
    )
    params = (; params..., best_π = best_coercion_table(params))
    rule === :exogenous && (π = best_coercion(ρ, params))
    q0 = isnothing(q0) ? steady_state(π, params) : q0
    0 <= q0 <= 1 || throw(ArgumentError("q0 must be in [0, 1]"))
    rule === :insurrection && (π = best_coercion(insurrection_constraint(q0, params), params))
    n_ns = round(Int, q0 * n_families)
    q = n_ns / n_families
    τ_ns, τ_s = efforts(q, π, params)
    properties = (;
        params...,
        # Yearly state. `Ref`s so the properties NamedTuple stays immutable while
        # the authority and the population summary can be updated in place.
        coercion = Ref(π),
        q = Ref(q),
        τ_ns = Ref(τ_ns),
        τ_s = Ref(τ_s),
        q_max_now = Ref(new_q_max(rng, params)),
        cracked_down = Ref(false),
        collapsed = Ref(false),
        authority = Ref(1),
    )
    model = StandardABM(
        Family;
        agent_step! = family_step!,
        model_step! = authority_step!,
        properties,
        rng,
    )
    identities = shuffle!(rng, [i <= n_ns for i = 1:n_families])
    for ns in identities
        add_agent!(Family, model, ns)
    end
    return model
end

"""
    history(model, T) -> NamedTuple of length-`T` vectors

Run `T` years and record, after each, the coercion `π`, the ns share `q`, the
legitimacy `L = 1 − q`, whether the authority cracked down or collapsed that
year, and which authority was in power (counting from 1).
"""
function history(model::StandardABM, T::Int)
    out = (
        coercion = zeros(T),
        q = zeros(T),
        legitimacy = zeros(T),
        crackdown = falses(T),
        collapse = falses(T),
        authority = zeros(Int, T),
    )
    for t = 1:T
        Agents.step!(model, 1)
        out.coercion[t] = model.coercion[]
        out.q[t] = model.q[]
        out.legitimacy[t] = 1 - model.q[]
        out.crackdown[t] = model.cracked_down[]
        out.collapse[t] = model.collapsed[]
        out.authority[t] = model.authority[]
    end
    return out
end

end # module
