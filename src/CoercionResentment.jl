"""
    CoercionResentment

Model logic for the coercion-resentment ABM described in
`models/coercion_resentment.md`, an agent-based version of Schøyen (2021,
Cliometrica) built to reproduce the stylized facts of Schøyen, Sirnes and
Andersen (2026, Public Choice).

One model is one country. Its **families** pass an identity — minority (the
non-state morality) or majority — to their children, Bisin–Verdier style: a
parent invests effort `τ` in socialising the child into its own identity, and if
that fails the child copies a random member of the population. Effort is the
best reply to the parent's gain from a same-identity child,

    g_m = Δu − π + C(π)      minority parent: the penalty π hurts, resentment C helps
    g_M = Δu + π             majority parent
    τ_m = k (1 − q) g_m,     τ_M = k q g_M        (quadratic effort cost),

so the minority share `q` relaxes, over generations, to the paper's steady state

    q*(π) = (Δu − π + C(π)) / (2Δu + C(π)).

With an S-shaped resentment `C`, `q*` falls, rises (the inefficient window, where
more coercion breeds a larger minority) and falls again.

One **regime** sets the coercion `π` each year by a satisficing rule, not by
optimisation. It judges its policy by whether the minority is growing, which it
reads off the socialisation flow: minority parents investing more than majority
parents. Coercion is costly, so it eases by `ease` a year while the minority is
not growing; it holds while the minority grows but is within its tolerance; and
it cracks down by `crackdown` when the minority is both above its tolerance and
growing. It falls
each year with a probability proportional to the resentful minority, `q·C(π)`,
and is replaced by a regime with a fresh tolerance that inherits the coercion
apparatus.

Time steps are years. Families reproduce with probability `1/generation` a year,
so the minority share is a slow stock while coercion is a fast flow.

The module is plotting-agnostic. All tunable parameters are `initialize` keyword
arguments; the yearly state of the regime and of the population lives in `Ref`s
in the model's `properties`.
"""
module CoercionResentment

using Agents, Random

export Family,
    initialize,
    family_step!,
    regime_step!,
    resentment,
    gains,
    steady_state,
    minority_share,
    history

# --- The agent ----------------------------------------------------------------

"""
    Family

A household, replaced by its child when it reproduces. `minority` is its
identity: `true` for the non-state morality the regime penalises. Every other
quantity a family needs (its gain, hence its effort) depends only on its identity
and on the country-wide `π` and `q`, so it is computed once a year in
[`regime_step!`](@ref) rather than stored here.
"""
@agent struct Family(NoSpaceAgent)
    minority::Bool
end

# --- The payoffs --------------------------------------------------------------

"""
    resentment(π, model_or_params) -> Float64

The S-shaped coercion resentment `C(π) = A [atan(s(π − π_c)) + atan(s π_c)]`:
zero at `π = 0`, steepest at the threshold `π_c`, saturating at `A(π/2 +
atan(s π_c))`. It has to be steeper than 2 somewhere for an inefficient window to
exist (Schøyen 2021, Assumption 3); `A = 0` turns resentment off.
"""
resentment(π, p) = p.resentment_scale * (atan(p.resentment_steepness * (π - p.resentment_threshold)) +
                                         atan(p.resentment_steepness * p.resentment_threshold))

"""
    gains(π, p) -> (g_m, g_M)

What a minority and a majority parent gain from a child with their own identity
rather than the other one. `u` cancels; only `Δu`, the penalty `π` and the
resentment `C(π)` remain.
"""
function gains(π, p)
    C = resentment(π, p)
    return (p.Δu - π + C, p.Δu + π)
end

"""
    steady_state(π, p) -> Float64

The paper's steady-state minority share `q*(π) = g_m / (g_m + g_M)`, which solves
`(1 − q) g_m = q g_M`: the two efforts balance. Zero once the minority's gain is
negative, i.e. once even minority parents prefer a majority child.
"""
function steady_state(π, p)
    g_m, g_M = gains(π, p)
    return clamp(g_m / (g_m + g_M), 0.0, 1.0)
end

"Socialisation efforts of a minority and a majority parent, clipped to `[0, 1]`."
function efforts(q, π, p)
    g_m, g_M = gains(π, p)
    return (clamp(p.effort * (1 - q) * g_m, 0.0, 1.0), clamp(p.effort * q * g_M, 0.0, 1.0))
end

# --- The submodels ------------------------------------------------------------

"""
    family_step!(family, model)

With probability `1/generation` the family is replaced by its child. The parent
socialises the child into its own identity with probability `τ`; otherwise the
child copies a random member of the population, which is a minority member with
probability `q`. Using last year's `q` and efforts keeps the year synchronous:
which family reproduces first does not matter.
"""
function family_step!(family::Family, model::StandardABM)
    rng = abmrng(model)
    rand(rng) < 1 / model.generation || return
    τ = family.minority ? model.τ_m[] : model.τ_M[]
    if rand(rng) >= τ
        family.minority = rand(rng) < model.q[]
    end
    return
end

"The current minority share."
minority_share(model::StandardABM) = count(f -> f.minority, allagents(model)) / nagents(model)

"Draw a new regime's tolerance, the minority share above which it cracks down."
new_tolerance(rng, p) = p.tolerance[1] + (p.tolerance[2] - p.tolerance[1]) * rand(rng)

"""
    regime_step!(model)

Runs after every family has reproduced. In order:

1. Observe the minority share `q`, and whether the minority is growing under
   the current coercion: `τ_m > τ_M`, since `E[Δq] ∝ q(1 − q)(τ_m − τ_M)`. This is
   the flow without its sampling noise, as a census of how children are raised
   would show it, so the regime reacts to its own policy without the
   generational lag of the stock `q`.
2. If the minority is not growing, ease (`π -= ease`). If it is growing and
   exceeds the regime's tolerance, crack down (`π += crackdown`). Otherwise
   hold. In the inefficient window a crackdown makes the minority grow faster,
   so an alarmed regime there escalates year after year until it is through.
3. Collapse with probability `collapse_rate · q · C(π) / C(π_max)`: the size of
   the resentful minority, relative to its largest possible value. The new regime
   keeps `π` and draws a new tolerance.
4. Compute next year's efforts from the new `π` and `q`.
"""
function regime_step!(model::StandardABM)
    rng = abmrng(model)
    q = minority_share(model)
    model.q[] = q
    π = model.coercion[]
    τ_m, τ_M = efforts(q, π, model)

    model.cracked_down[] = false
    if τ_m <= τ_M
        π -= model.ease
    elseif q > model.tolerance_now[]
        π += model.crackdown
        model.cracked_down[] = true
    end
    π = clamp(π, 0.0, model.π_max)
    model.coercion[] = π

    model.collapsed[] = false
    if rand(rng) < model.collapse_rate * q * resentment(π, model) / resentment(model.π_max, model)
        model.collapsed[] = true
        model.regime[] += 1
        model.tolerance_now[] = new_tolerance(rng, model)
    end

    model.τ_m[], model.τ_M[] = efforts(q, π, model)
    return
end

# --- Construction -------------------------------------------------------------

"""
    initialize(; kwargs...) -> StandardABM

Build one country.

Keyword arguments:
- `n_families`: population size `N`. Sets the finite-population noise in `q`.
- `π0`: initial coercion; `nothing` draws it uniformly from `[0, π_max]`.
- `q0`: initial minority share, assigned at random to `round(q0 N)` families;
  `nothing` starts the families in equilibrium with `π0`, at `q*(π0)`.
- `π_max`: the top of the coercion scale (10, like the paper's indices).
- `Δu`: the loss from a child with the other identity.
- `resentment_scale`, `resentment_threshold`, `resentment_steepness`: `A`, `π_c`
  and `s` of [`resentment`](@ref).
- `effort`: `k`, the inverse cost of socialisation effort.
- `generation`: years per generation. One family in `generation` reproduces a year.
- `ease`: yearly easing of coercion while the minority is tolerated.
- `crackdown`: the jump in coercion when it is not.
- `tolerance`: `(lo, hi)`, the uniform range each regime's tolerance is drawn from.
- `collapse_rate`: the yearly collapse probability at the largest possible
  resentful minority. `0` keeps one regime forever.
- `seed`: RNG seed.

`ease = crackdown = collapse_rate = 0` freezes coercion at `π0`, which leaves the
families alone to relax to `q*(π0)`.
"""
function initialize(;
    n_families = 1000,
    q0 = nothing,
    π0 = nothing,
    π_max = 10.0,
    Δu = 5.0,
    resentment_scale = 2.6,
    resentment_threshold = 4.5,
    resentment_steepness = 1.6,
    effort = 0.1,
    generation = 25,
    ease = 0.1,
    crackdown = 1.5,
    tolerance = (0.15, 0.55),
    collapse_rate = 0.05,
    seed = 1234,
)
    tolerance[1] <= tolerance[2] || throw(ArgumentError("tolerance must be (lo, hi) with lo ≤ hi"))
    rng = MersenneTwister(seed)
    π = isnothing(π0) ? π_max * rand(rng) : float(π0)
    0 <= π <= π_max || throw(ArgumentError("π0 must be in [0, π_max]"))

    params = (;
        π_max = float(π_max),
        Δu = float(Δu),
        resentment_scale = float(resentment_scale),
        resentment_threshold = float(resentment_threshold),
        resentment_steepness = float(resentment_steepness),
        effort = float(effort),
        generation = generation,
        ease = float(ease),
        crackdown = float(crackdown),
        tolerance = float.(tolerance),
        collapse_rate = float(collapse_rate),
    )
    q0 = isnothing(q0) ? steady_state(π, params) : q0
    0 <= q0 <= 1 || throw(ArgumentError("q0 must be in [0, 1]"))
    n_minority = round(Int, q0 * n_families)
    q = n_minority / n_families
    τ_m, τ_M = efforts(q, π, params)
    properties = (;
        params...,
        # Yearly state. `Ref`s so the properties NamedTuple stays immutable while
        # the regime and the population summary can be updated in place.
        coercion = Ref(π),
        q = Ref(q),
        τ_m = Ref(τ_m),
        τ_M = Ref(τ_M),
        tolerance_now = Ref(new_tolerance(rng, params)),
        cracked_down = Ref(false),
        collapsed = Ref(false),
        regime = Ref(1),
    )
    model = StandardABM(
        Family;
        agent_step! = family_step!,
        model_step! = regime_step!,
        properties,
        rng,
    )
    minority = shuffle!(rng, [i <= n_minority for i = 1:n_families])
    for m in minority
        add_agent!(Family, model, m)
    end
    return model
end

"""
    history(model, T) -> NamedTuple of length-`T` vectors

Run `T` years and record, after each, the coercion, the minority share, the
legitimacy `1 − q`, whether the regime cracked down or collapsed that year, and
which regime was in power.
"""
function history(model::StandardABM, T::Int)
    out = (
        coercion = zeros(T),
        q = zeros(T),
        legitimacy = zeros(T),
        crackdown = falses(T),
        collapse = falses(T),
        regime = zeros(Int, T),
    )
    for t = 1:T
        Agents.step!(model, 1)
        out.coercion[t] = model.coercion[]
        out.q[t] = model.q[]
        out.legitimacy[t] = 1 - model.q[]
        out.crackdown[t] = model.cracked_down[]
        out.collapse[t] = model.collapsed[]
        out.regime[t] = model.regime[]
    end
    return out
end

end # module
