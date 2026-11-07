"""
    TouristDestinations

Model logic for the tourist-destination ABM described in
`models/tourist_destinations.md`: how social influence shapes where people go
on holiday.

`N` tourists each travel once a year to one of `D` destinations. Tourist `n`
holds a weight `w_nd` for every destination `d`, drawn at random at the start
(independent tastes), and each year throws a `D`-faced die whose faces are
weighted by `w_nd`. After the holidays everyone sees where everyone else went,
and the weights move towards what they saw, by one of two rules (`update`):

- `:additive`: every weight grows by `ε` times the number of *other* tourists
  who went there that year,

      w_nd ← w_nd + ε Σ_{n′≠n} [d_{n′} = d].

  The weights are not normalised (the die only uses their ratios). The
  reinforcement is linear and every past year keeps the same weight, so the
  population behaves like a Pólya urn: a destination's expected share is its
  share of the visits accumulated so far, a martingale whose steps shrink like
  `1/t`. The shares freeze at whatever the first years happened to give.
  [`polya_prediction`](@ref) gives the mean-field version.
- `:mixing`: the weights are a probability distribution over destinations, and
  each year a fraction `ε` of it is replaced by the distribution of where
  everyone went,

      w_nd ← (1 − ε) w_nd + (ε/N) Σ_{n′} [d_{n′} = d].

  Older years fade as `(1 − ε)^age`, so the steps of the martingale do not
  shrink. The shares drift for ever, like gene frequencies under neutral
  genetic drift (Wright–Fisher), until one destination has everyone. Normalising
  the `:additive` weights after every year gives the same rule, with
  `ε_mixing = ε(N − 1) / (1 + ε(N − 1))`, the tourist's own trip left out.
  [`drift_prediction`](@ref) gives its mean-field version.

  With a conformity exponent `β ≠ 1` the signal is no longer the plain
  distribution of trips but a sharpened one,

      s_d = c_d^β / Σ_d′ c_d′^β,    c_d = Σ_{n′} [d_{n′} = d],
      w_nd ← (1 − ε) w_nd + ε s_d,

  so that for `β > 1` popular destinations count more than their share of
  visitors (social proof, trending lists, feeds ranked by popularity). `β = 1`
  is the mixing rule. Near the even split a destination's excess popularity `δ`
  then grows as `δ ← [1 + ε(β − 1)] δ` whatever `N` is, so drift becomes
  selection and every run ends with one destination for everyone, within
  decades rather than millennia. [`conformity_prediction`](@ref) gives the early,
  near-even part of this.

`ε = 0` keeps the initial tastes for ever under either rule.

The module is plotting-agnostic. All tunable parameters are `initialize` keyword
arguments.
"""
module TouristDestinations

using Agents, Random

export Tourist,
    initialize,
    tourist_step!,
    influence_step!,
    popularity,
    history,
    effective_number,
    polya_prediction,
    drift_prediction,
    conformity_prediction

# --- The agent ----------------------------------------------------------------

"""
    Tourist

A person who travels once a year. `weights[d]` is their weight `w_nd` on
destination `d`; `destination` is where they went this year.
"""
@agent struct Tourist(NoSpaceAgent)
    weights::Vector{Float64}
    destination::Int
end

# --- The submodels ------------------------------------------------------------

"Draw an index of `w` with probability proportional to its entry (a weighted die)."
function weighted_choice(rng, w)
    r = rand(rng) * sum(w)
    acc = 0.0
    for d in eachindex(w)
        acc += w[d]
        r < acc && return d
    end
    return lastindex(w)   # only reached through round-off, when r ≈ sum(w)
end

"""
    tourist_step!(tourist, model)

The tourist throws their weighted die and goes where it lands. Weights change
only after everyone has travelled, in [`influence_step!`](@ref), so the order
in which tourists choose does not matter.
"""
function tourist_step!(tourist::Tourist, model::StandardABM)
    tourist.destination = weighted_choice(abmrng(model), tourist.weights)
    return
end

"""
    influence_step!(model)

Runs after every tourist has travelled. Counts this year's visitors to each
destination into `model.visits`, then updates every tourist's weights:

- `:additive`: adds `ε` times each destination's count, minus the tourist's own
  trip (the sum is over `n′ ≠ n`). Since the tourist's own destination has at
  least one visitor (them), no weight ever falls.
- `:mixing`: `w ← (1 − ε) w + ε s`, a mixture of two distributions, so `w`
  stays one. The signal is `s = visits / N`, or with conformity exponent
  `β ≠ 1`, `s = visits.^β / sum(visits.^β)`.
"""
function influence_step!(model::StandardABM)
    v = model.visits
    fill!(v, 0)
    for t in allagents(model)
        v[t.destination] += 1
    end
    ε = model.ε
    ε == 0 && return
    if model.update === :additive
        for t in allagents(model)
            t.weights .+= ε .* v
            t.weights[t.destination] -= ε
        end
    else
        β = model.β
        share = β == 1 ? v ./ nagents(model) : v .^ β ./ sum(v .^ β)
        for t in allagents(model)
            t.weights .= (1 - ε) .* t.weights .+ ε .* share
        end
    end
    return
end

# --- Construction -------------------------------------------------------------

"""
    initialize(; kwargs...) -> StandardABM

Build the population.

Keyword arguments:
- `n_tourists`: `N`, the number of tourists (1000).
- `n_destinations`: `D`, the number of destinations (100).
- `ε`: the influence parameter: ≥ 0 under `:additive`, in `[0, 1]` under
  `:mixing`, where it is the fraction of a tourist's distribution replaced each
  year.
- `update`: `:additive` (default) or `:mixing`; see the module docstring.
- `β`: the conformity exponent of the `:mixing` signal (> 0). `1` (default) is
  the plain distribution of trips; `β > 1` makes popular destinations count
  more than their share. Only `:mixing` takes `β ≠ 1`.
- `initial_weights`: a function `rng -> w` returning one tourist's initial weight
  vector of length `D`; the default draws each `w_nd` uniformly from `[0, 1]`.
  Under `:mixing` it is divided by its sum, so each tourist starts with a
  probability distribution.
- `seed`: RNG seed.

`destination` is 0 until the first year has been run.
"""
function initialize(;
    n_tourists = 1000,
    n_destinations = 100,
    ε = 0.01,
    update = :additive,
    β = 1.0,
    initial_weights = rng -> rand(rng, n_destinations),
    seed = 1234,
)
    update in (:additive, :mixing) || throw(ArgumentError("update must be :additive or :mixing"))
    ε >= 0 || throw(ArgumentError("ε must be ≥ 0"))
    update === :mixing && ε > 1 && throw(ArgumentError("ε must be ≤ 1 under :mixing"))
    β > 0 || throw(ArgumentError("β must be > 0"))
    β == 1 || update === :mixing || throw(ArgumentError("β ≠ 1 needs update = :mixing"))
    n_tourists >= 1 && n_destinations >= 1 ||
        throw(ArgumentError("need at least one tourist and one destination"))
    rng = MersenneTwister(seed)
    properties = (; ε = float(ε), β = float(β), update, n_destinations, visits = zeros(Int, n_destinations))
    model = StandardABM(
        Tourist;
        agent_step! = tourist_step!,
        model_step! = influence_step!,
        properties,
        rng,
    )
    for _ = 1:n_tourists
        w = float.(initial_weights(rng))
        length(w) == n_destinations && all(>=(0), w) && sum(w) > 0 ||
            throw(ArgumentError("initial weights must be $n_destinations non-negative numbers, not all zero"))
        update === :mixing && (w ./= sum(w))
        add_agent!(Tourist, model, w, 0)
    end
    return model
end

# --- Running and summaries ----------------------------------------------------

"""
    popularity(model) -> Vector{Float64}

Each destination's expected share of next year's tourists: the mean over
tourists of `w_nd / Σ_d′ w_nd′`, their chance of going there. It is the
noise-free version of next year's `visits / N`.
"""
function popularity(model::StandardABM)
    p = zeros(model.n_destinations)
    for t in allagents(model)
        p .+= t.weights ./ sum(t.weights)
    end
    return p ./ nagents(model)
end

"""
    history(model, T) -> (; visits, popularity)

Run `T` years. Returns two `D × T` matrices: `visits[d, t]` is how many tourists
went to destination `d` in year `t`, and `popularity[d, t]` is its
[`popularity`](@ref) going into year `t`, the expected share that `visits[:, t]`
was drawn from (column 1 holds the initial tastes alone).
"""
function history(model::StandardABM, T::Int)
    D = model.n_destinations
    visits = zeros(Int, D, T)
    pop = zeros(D, T)
    for t = 1:T
        pop[:, t] .= popularity(model)
        Agents.step!(model, 1)
        visits[:, t] .= model.visits
    end
    return (; visits, popularity = pop)
end

"""
    effective_number(counts) -> Float64

The effective number of destinations `1 / Σ_d p_d²` (inverse Simpson index) of
a vector of visitor counts, with `p_d` the share of destination `d`: `D` when
every destination gets the same number of visitors, 1 when everyone goes to
the same place.
"""
function effective_number(counts)
    p = counts ./ sum(counts)
    return 1 / sum(abs2, p)
end

"""
    polya_prediction(ε, T; n_tourists = 1000, n_destinations = 100,
                     initial_total = n_destinations / 2) -> (; effective_number, lock_in)

Mean-field prediction, after `T` years, of how unequal the destinations'
popularity has become and how much of it was decided in year 1.

Treat every tourist as holding the population's average weights: initial total
`W₀ = initial_total` (`D/2` for weights uniform on `[0, 1]`) spread evenly, so
each destination starts at share `X = 1/D`, plus `ε` times all the visits so
far. The year-`t` visits are then a multinomial sample of `N` from the current
shares, and the share of destination `d` moves by

    ΔX_d(t) = a_t (v_d(t)/N − X_d(t−1)),    a_t = εN / (W₀ + εN t),

a martingale whose steps shrink like `1/t`. Summing their variances,
`Var X_d(T) = X(1 − X) S` with `S = Σ_t a_t² / N`, so

- `effective_number = D / (1 + (D − 1) S)`, the expected
  [`effective_number`](@ref) of the popularity after `T` years. As `ε → ∞`,
  `a_t → 1/t` and it falls no lower than `D / (1 + (D − 1) Σ 1/t² / N)`
  (86 for `N = 1000`, `D = 100`, `T = 100`): no influence, however strong, makes
  everyone go to the same few places.
- `lock_in = a_1 / sqrt(Σ_t a_t²)`, the correlation across destinations between
  year-1 visitors and the popularity after `T` years: the part of the final
  popularity that is year 1's luck. It rises from `1/√T` at small `ε` (every year
  counts the same) to `1/√(Σ 1/t²) ≈ 0.78` at large `ε` (year 1 counts most).

It ignores the tourists' own visits being left out (an `O(1/N)` effect) and the
small differences between destinations in the population's average initial
tastes, which matter only for `ε` well below `10⁻³` with the default sizes.
At `ε = 0` it returns `(D, 0)`; at `T = 0`, `(D, NaN)`, since there is no year 1 yet.
"""
function polya_prediction(ε, T; n_tourists = 1000, n_destinations = 100,
    initial_total = n_destinations / 2)
    N, D, W₀ = n_tourists, n_destinations, initial_total
    ε == 0 && return (effective_number = float(D), lock_in = 0.0)
    T == 0 && return (effective_number = float(D), lock_in = NaN)
    a = [ε * N / (W₀ + ε * N * t) for t = 1:T]
    S = sum(abs2, a) / N
    return (effective_number = D / (1 + (D - 1) * S), lock_in = a[1] / sqrt(sum(abs2, a)))
end

"""
    drift_prediction(ε, T; n_tourists = 1000, n_destinations = 100) -> Float64

Mean-field prediction of the effective number of destinations of the popularity
after `T` years under `update = :mixing`, as `1 / E[Σ_d p_d²]`.

The popularity `X` (the tourists' average distribution) moves each year by
`ΔX = ε (v/N − X)`, where the visits `v` are a multinomial sample of `N` from
`X` once the tourists' own tastes have faded. Its expectation is zero, so
popularity has no drift towards the popular, but its variance,
`ε² X(1 − X) / N`, does not shrink with time as it does under `:additive`.
Then `E[Σ X²]` grows by `ε²(1 − Σ X²) / N` a year, and the heterogeneity
`1 − Σ X²` decays geometrically from `1 − 1/D`:

    E[Σ X²(T)] = 1 − (1 − 1/D) (1 − ε²/N)^T.

This is neutral genetic drift in a Wright–Fisher population of effective size
`N/ε²`. The effective number falls towards 1, one destination for everyone, on
a time scale of `N/ε²` years. It ignores the differences between the tourists'
initial tastes, which make the yearly sample slightly less noisy while they last
(a relative correction of about `1/(3D)`, fading as `(1 − ε)^{2t}`).
"""
function drift_prediction(ε, T; n_tourists = 1000, n_destinations = 100)
    D = n_destinations
    return 1 / (1 - (1 - 1 / D) * (1 - ε^2 / n_tourists)^T)
end

"""
    conformity_prediction(ε, β, T; n_tourists = 1000, n_destinations = 100) -> Float64

Linear-noise prediction of the effective number of destinations `1 / E[Σ p_d²]`
of the popularity after `T` years under `update = :mixing` with conformity
exponent `β`, valid while the popularity is still close to an even split.

Write a destination's popularity as `X_d = 1/D + δ_d`. Its share of this year's
visitors is `X_d + η_d`, with sampling noise of variance
`σ² = X(1 − X)/N = (D − 1)/(D² N)`. To first order in `δ` and `η` the sharpened
signal is `s_d ≈ 1/D + β (δ_d + η_d)`, so

    δ ← g δ + ε β η,    g = 1 + ε(β − 1),

and from an even start `Var δ(T) = ε² β² σ² (g^{2T} − 1) / (g² − 1)`, giving
`1 / (1/D + D Var δ)`. For `β > 1` the excess grows like `g^T`: the rate depends
on `ε(β − 1)` alone, and `N` only sets the size of the first fluctuations, so a
10 times larger population starts concentrating `ln(10) / (2 ln g)` years later.
At `β = 1` it is the first-order form of [`drift_prediction`](@ref). Once
popularity is far from even the linearisation fails (the prediction falls below
1 for large `T`, which the effective number cannot); it is clamped at 1.
"""
function conformity_prediction(ε, β, T; n_tourists = 1000, n_destinations = 100)
    N, D = n_tourists, n_destinations
    σ² = (D - 1) / (D^2 * N)
    g = 1 + ε * (β - 1)
    growth = g == 1 ? float(T) : (g^(2T) - 1) / (g^2 - 1)
    V = ε^2 * β^2 * σ² * growth
    return max(1.0, 1 / (1 / D + D * V))
end

end # module
