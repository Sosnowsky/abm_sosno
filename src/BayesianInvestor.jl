"""
    BayesianInvestor

Model logic for the *Bayesian-belief* business-investor ABM described in
`models/The business investor model.md`. The omniscient-risk predecessor lives in
`src/BusinessInvestor.jl` and is documented in
`models/The business investor model 0.md`; it is the control this model is
compared against.

Investors live on a grid whose patches are *businesses*, each with a fixed
`profit` and `risk`. The single change from the predecessor: **`risk` is hidden**.
An investor never reads `F`, and instead maintains a Beta belief about each
business's failure rate, updated from wipeouts it *witnesses* within its
`sensing_radius` — its own and those of the investors occupying nearby patches.
A business only fails while occupied, so an empty patch emits no evidence.

Each step has two phases: investors reposition using *believed* utility
`U = (W + T*profit) * (1 - F̂)^T`, then the model runs accounting (profit,
bankruptcy, ruin) followed by observation (belief updates). Accounting and
observation are **separate loops over all agents** — an investor must not witness
a neighbour that has not yet done its accounting, or beliefs would depend on
scheduler order.

The belief machinery draws **no random numbers**, so at `sensing_radius = 0` this
model consumes exactly the same RNG stream as the predecessor and reproduces its
results bit for bit. That is the regression test in `test_bayesian_investor.jl`.

This module is realization- and visualization-agnostic (no Makie dependency).
Tunable parameters are `initialize` keyword arguments; the ones the step
functions read at run time live in the model's `properties`.
"""
module BayesianInvestor

using Agents, Random

export Investor,
    initialize,
    investor_step!,
    model_step!,
    accounting_step!,
    observation_step!,
    utility,
    posterior_mean,
    belief,
    resolve_prior,
    mean_wealth,
    mean_chosen_profit,
    mean_chosen_risk,
    mean_posterior_error,
    mean_posterior_shift,
    mean_observed_patches,
    mean_data_weight,
    informed_move_fraction,
    density

# Investors carry wealth, a belief memory, and a transient wipeout flag.
#
# `beliefs` maps a grid position to `(n, k)`: the number of steps this investor
# observed that patch *occupied*, and how many of those ended in the occupant
# being wiped out. A `Dict` rather than a matrix because an investor only ever
# observes a handful of the 361 patches; the rest stay at the prior and need no
# storage.
#
# `wiped` is the only channel by which one investor's outcome reaches another's
# beliefs. It is set during accounting and read during observation, and must be
# cleared each step so a stale flag is never witnessed twice.
@agent struct Investor(GridAgent{2})
    wealth::Float64
    beliefs::Dict{Tuple{Int,Int},Tuple{Int,Int}}
    wiped::Bool
end

"""
    belief(investor, pos) -> (n, k)

This investor's evidence about the patch at `pos`: `n` steps observed occupied,
`k` of which ended in a wipeout. A patch never seen occupied returns `(0, 0)`,
which puts it exactly at the prior.
"""
belief(investor::Investor, pos) = get(investor.beliefs, pos, (0, 0))

"""
    posterior_mean(n, k, α₀, β₀) -> Float64

Mean of the Beta posterior `Beta(α₀ + k, β₀ + n - k)` on a business's failure
rate: `F̂ = (α₀ + k) / (α₀ + β₀ + n)`.

With the default moment-matched prior (`α₀ = 4.18`, `β₀ = 71.82`) an unobserved
patch gives `F̂ = 0.055`, the true landscape mean — investors start calibrated
about risk *in general* and lack only the ability to tell one business from
another. The prior is worth `α₀ + β₀ = 76` pseudo-observations, so 25 steps of
evidence shifts `F̂` by only a few thousandths.
"""
posterior_mean(n, k, α₀, β₀) = (α₀ + k) / (α₀ + β₀ + n)

"""
    utility(wealth, profit, n, k, T, α₀, β₀) -> Float64

Value an investor assigns to a business: `(wealth + T*profit) * (1 - F̂)^T`, with
`F̂` the posterior mean implied by the evidence `(n, k)`.

Textually identical to the predecessor's utility; only the risk term changed,
from the true `F` to a point estimate. It remains the expected wealth after `T`
steps under the belief, because both ruin and bankruptcy reset wealth to zero.
With negative profits `wealth + T*profit` can be negative, and then the formula
prefers *higher* risk — an investor facing certain losses would rather be wiped
out. That is a genuine consequence of the reset-to-zero rule, not a special case,
so the formula is left unchanged.
"""
utility(wealth, profit, n, k, T, α₀, β₀) =
    (wealth + T * profit) * (1 - posterior_mean(n, k, α₀, β₀))^T

"""
    resolve_prior(prior_mean, prior_strength, risk_range) -> (α₀, β₀)

Turn the public *(mean, strength)* prior parameterization into the `(α₀, β₀)` the
step functions want. Either argument may be `nothing`, in which case it is
derived from `risk_range` so that the prior is **moment-matched** to the true risk
distribution: for `F ~ U(lo, hi)` with mean `m` and variance `v = (hi-lo)^2/12`,

    s₀ = m(1-m)/v - 1,    α₀ = m₀·s₀,    β₀ = (1-m₀)·s₀

The two coordinates separate two questions that `(α₀, β₀)` confounds:
`prior_mean` is a *calibration* claim (is the investor right on average?),
`prior_strength` a *confidence* claim (how hard does it hold that belief before
evidence arrives?). Moving `α₀` alone shifts both at once.

Note `prior_strength` is derived from the *landscape's* moments even when
`prior_mean` is overridden — overriding the mean is meant to misspecify
calibration only, leaving the confidence unchanged.
"""
function resolve_prior(prior_mean, prior_strength, risk_range)
    lo, hi = risk_range
    landscape_mean = (lo + hi) / 2
    landscape_var = (hi - lo)^2 / 12

    m₀ = isnothing(prior_mean) ? landscape_mean : prior_mean
    s₀ = if isnothing(prior_strength)
        @assert landscape_var > 0 "cannot moment-match a prior to a degenerate risk_range $risk_range"
        landscape_mean * (1 - landscape_mean) / landscape_var - 1
    else
        prior_strength
    end

    @assert 0 < m₀ < 1 "prior_mean must lie strictly in (0, 1), got $m₀"
    @assert s₀ > 0 "prior_strength must be positive, got $s₀"

    α₀ = m₀ * s₀
    β₀ = (1 - m₀) * s₀
    @assert α₀ > 0 && β₀ > 0 "prior must have α₀ > 0 and β₀ > 0, got ($α₀, $β₀)"
    # α₀ < 1 is a *shape* change, not merely a weaker belief: the Beta density
    # diverges at zero, so the prior becomes a spike near F ≈ 0 with a long tail
    # rather than a vague bump. Legitimate, but qualitatively different — flag it
    # rather than let it be discovered.
    if α₀ < 1
        @warn "α₀ < 1: prior is a spike near F ≈ 0 with a long tail, not a vague bump" α₀ β₀ m₀ s₀
    end
    return α₀, β₀
end

"""
    investor_step!(investor, model)

Phase 1 (repositioning). Among the *empty* patches within `model.sensing_radius`
(Chebyshev distance), move to the one with the highest **believed** utility if it
strictly beats staying put; otherwise stay. Ties are broken at random. A
`sensing_radius` of 0 leaves no candidates, so the investor never relocates.

Note the asymmetry this creates. The investor's own patch is observed every
single step, so it is always the best-measured thing in view, while a candidate
carries evidence only if it was watched *while occupied* and has since emptied.
Everything else in range sits at the prior. Evidence and opportunity are
anti-correlated, which is why risk selection here is rare rather than impossible.
"""
function investor_step!(investor::Investor, model::StandardABM)
    rng = abmrng(model)
    P, T, α₀, β₀ = model.profit, model.T, model.α₀, model.β₀

    here = investor.pos
    n_here, k_here = belief(investor, here)
    # Baseline: the believed utility of staying. A patch must strictly beat this
    # to justify a move, which is what "stay if current is best" means.
    best_u = utility(investor.wealth, P[here...], n_here, k_here, T, α₀, β₀)
    candidates = Tuple{Int,Int}[]

    for p in nearby_positions(investor, model, model.sensing_radius)
        isempty(p, model) || continue  # one investor per patch
        n_p, k_p = belief(investor, p)
        u = utility(investor.wealth, P[p...], n_p, k_p, T, α₀, β₀)
        if u > best_u
            best_u = u
            empty!(candidates)
            push!(candidates, p)
        elseif u == best_u && !isempty(candidates)
            push!(candidates, p)  # tie among neighbours already better than staying
        end
    end

    isempty(candidates) && return  # current patch is best: stay

    target = rand(rng, candidates)
    model.n_moves[] += 1
    # An "informed" move is one onto a patch the investor actually has evidence
    # about. Predicted to be rare: the patch had to be occupied while watched,
    # then vacated. This counter is the cleanest test of that prediction.
    if first(belief(investor, target)) > 0
        model.n_informed_moves[] += 1
    end
    move_agent!(investor, target, model)
    return
end

"""
    accounting_step!(model)

Phase 2a. Each investor's wealth changes by the profit of the business it
occupies — which may be a loss. An investor pushed below zero **goes bankrupt**:
wealth resets to zero and it keeps playing. Then, with probability equal to that
business's risk, a **ruin event** wipes the wealth to zero. A business is only
tested against its risk while occupied, so empty patches never fail.

Either mechanism sets the investor's `wiped` flag, which is what observers see.
The model tallies the two causes separately even though neither the occupant nor
any observer can distinguish them — that gap is the point, and the tallies
quantify how badly `k` overstates the true failure rate.

Investors are independent here, so order is irrelevant.
"""
function accounting_step!(model::StandardABM)
    rng = abmrng(model)
    for investor in allagents(model)
        # Clear last step's flag before anything can witness it.
        investor.wiped = false
        pos = investor.pos
        investor.wealth += model.profit[pos...]
        # Bankruptcy: debts are not carried, the investor simply restarts at zero.
        if investor.wealth < 0
            investor.wealth = 0.0
            investor.wiped = true
            model.n_bankruptcies[] += 1
        end
        if rand(rng) < model.risk[pos...]
            investor.wealth = 0.0
            investor.wiped = true
            model.n_ruins[] += 1
        end
    end
    return
end

"""
    observation_step!(model)

Phase 2b. Every investor scans the patches within `model.sensing_radius`
**including its own**, and for each one that is occupied records one observation:
`n += 1`, and `k += 1` if that patch's occupant was wiped this step. Empty
patches are skipped and contribute nothing — an unoccupied business cannot fail,
so it emits no evidence.

Must run only after *all* accounting is done: a wipeout has to have happened
before it can be witnessed. Observation only reads state, so the order among
observers is irrelevant. Draws no random numbers.
"""
function observation_step!(model::StandardABM)
    r = model.sensing_radius
    for investor in allagents(model)
        # `nearby_positions` excludes the centre, but the own patch is the single
        # most important observation — it is the only one guaranteed occupied.
        observe!(investor, investor.pos, model)
        for p in nearby_positions(investor, model, r)
            observe!(investor, p, model)
        end
    end
    return
end

"""
    observe!(investor, pos, model)

Record one observation of `pos` into `investor`'s belief memory, if occupied.
The observer sees only *that* the occupant was wiped, never why, and never the
occupant's wealth — so a poor investor bankrupting itself on a loss-making patch
looks exactly like a business collapsing.
"""
function observe!(investor::Investor, pos, model::StandardABM)
    ids = ids_in_position(pos, model)
    isempty(ids) && return  # empty business: no evidence at all
    occupant = model[first(ids)]  # single occupancy is enforced throughout
    n, k = belief(investor, pos)
    investor.beliefs[pos] = (n + 1, occupant.wiped ? k + 1 : k)
    return
end

"""
    model_step!(model)

Phase 2: accounting for every investor, *then* observation for every investor.

Two separate loops, deliberately. A fused loop would let an investor witness
neighbours that have not yet done their accounting, making the belief update
depend on scheduler order. This ordering is a correctness requirement, not a
style choice.
"""
function model_step!(model::StandardABM)
    accounting_step!(model)
    observation_step!(model)
    return
end

"""
    initialize(; kwargs...) -> StandardABM

Build a fresh Bayesian business-investor model.

Keyword arguments:
- `dims`: grid size in patches (default `(19, 19)`).
- `n_investors`: number of investors (default `100`). Also sets the **density**
  `ρ = n_investors / prod(dims)`, which controls how much vicarious evidence
  exists at all — it is a cognitive parameter here, not just a sample size.
- `T`: planning horizon in the utility function (default `5`).
- `sensing_radius`: Chebyshev radius an investor can see, learn from, and move to
  (default `1`; `0` freezes movement and leaves own-patch learning only).
- `profit_range`: `(lo, hi)` uniform range for each business's profit (default
  `(-5.0, 10.0)`; negative values are loss-making businesses).
- `risk_range`: `(lo, hi)` uniform range for each business's risk (default
  `(0.01, 0.1)`).
- `prior_mean`, `prior_strength`: the Beta prior on a business's failure rate.
  Both `nothing` by default, then moment-matched to `risk_range` (giving
  `m₀ = 0.055`, `s₀ = 76`). See [`resolve_prior`](@ref).
- `profit`, `risk`: optional precomputed matrices (override the ranges), e.g. to
  reuse one fixed business landscape across runs.
- `seed`: RNG seed.

The RNG is drawn from in exactly the order the omniscient predecessor uses
(profit matrix, risk matrix, then placements), and no belief machinery consumes
randomness, so `sensing_radius = 0` reproduces that model exactly.
"""
function initialize(;
    dims = (19, 19),
    n_investors = 100,
    T = 5,
    sensing_radius = 1,
    profit_range = (-5.0, 10.0),
    risk_range = (0.01, 0.1),
    prior_mean = nothing,
    prior_strength = nothing,
    profit = nothing,
    risk = nothing,
    seed = 1234,
)
    rng = MersenneTwister(seed)

    plo, phi = profit_range
    rlo, rhi = risk_range
    P = isnothing(profit) ? plo .+ (phi - plo) .* rand(rng, dims...) : profit
    F = isnothing(risk) ? rlo .+ (rhi - rlo) .* rand(rng, dims...) : risk
    @assert size(P) == dims "profit matrix must have size $dims"
    @assert size(F) == dims "risk matrix must have size $dims"
    @assert n_investors <= prod(dims) "cannot place $n_investors investors on $(prod(dims)) patches"

    α₀, β₀ = resolve_prior(prior_mean, prior_strength, risk_range)

    properties = (;
        profit = P,
        risk = F,
        T = T,
        sensing_radius = sensing_radius,
        dims = dims,
        α₀ = α₀,
        β₀ = β₀,
        # Cumulative tallies. `Ref`s so the properties NamedTuple stays immutable
        # and type-stable while the counts can still be updated in place.
        n_bankruptcies = Ref(0),
        n_ruins = Ref(0),
        n_moves = Ref(0),
        n_informed_moves = Ref(0),
    )
    # Non-periodic: the business grid has real edges, not a wrap-around torus.
    space = GridSpace(dims, periodic = false)
    model = StandardABM(
        Investor,
        space;
        agent_step! = investor_step!,
        model_step! = model_step!,
        properties,
        rng,
        scheduler = Schedulers.Randomly(),
    )

    # One investor per patch: add_agent_single! places each on a distinct empty
    # patch. Each investor gets its *own* belief Dict — sharing one would make
    # every investor omniscient by accident.
    for _ = 1:n_investors
        add_agent_single!(Investor, model, 0.0, Dict{Tuple{Int,Int},Tuple{Int,Int}}(), false)
    end

    return model
end

# --- Model-level metrics -----------------------------------------------------
#
# The first three match the predecessor's so runs can be compared directly. The
# rest measure the belief machinery: how much evidence exists, how far beliefs
# travel from the prior, and whether any of it reaches the decision rule.

"Mean investor wealth."
mean_wealth(model) = mean_over(a -> a.wealth, model)

"Mean profit of the patches investors currently occupy — what the population selects."
mean_chosen_profit(model) = mean_over(a -> model.profit[a.pos...], model)

"Mean *true* risk of the patches investors currently occupy. Investors cannot see this; the observer can."
mean_chosen_risk(model) = mean_over(a -> model.risk[a.pos...], model)

"""
    mean_posterior_error(model)

Mean `|F̂(p) - F(p)|` over every (investor, observed patch) pair. Predicted to be
roughly flat over a run rather than declining: 25 steps cannot outweigh a prior
worth 76 pseudo-observations.
"""
function mean_posterior_error(model)
    total, count = 0.0, 0
    for a in allagents(model), (pos, (n, k)) in a.beliefs
        total += abs(posterior_mean(n, k, model.α₀, model.β₀) - model.risk[pos...])
        count += 1
    end
    return count == 0 ? 0.0 : total / count
end

"""
    mean_posterior_shift(model)

Mean `|F̂ - F̂_prior|` over every (investor, observed patch) pair — how far beliefs
travel from the prior at all. Predicted to stay below about `0.015`.
"""
function mean_posterior_shift(model)
    prior = posterior_mean(0, 0, model.α₀, model.β₀)
    total, count = 0.0, 0
    for a in allagents(model), (_, (n, k)) in a.beliefs
        total += abs(posterior_mean(n, k, model.α₀, model.β₀) - prior)
        count += 1
    end
    return count == 0 ? 0.0 : total / count
end

"Mean number of distinct patches an investor holds evidence about."
mean_observed_patches(model) = mean_over(a -> length(a.beliefs), model)

"""
    mean_data_weight(model)

Mean `w = n / (s₀ + n)` over every (investor, observed patch) pair: the share of
a belief that came from evidence rather than from the prior. Under a
moment-matched prior `w` is also the *reliability* of that evidence, so it doubles
as the expected `corr(F̂, F)²` — which is why it is the natural x-axis for the
prior-strength and run-length sweeps.
"""
function mean_data_weight(model)
    s₀ = model.α₀ + model.β₀
    total, count = 0.0, 0
    for a in allagents(model), (_, (n, _)) in a.beliefs
        total += n / (s₀ + n)
        count += 1
    end
    return count == 0 ? 0.0 : total / count
end

"""
    informed_move_fraction(model)

Cumulative share of relocations whose destination the investor already had
evidence about (`n > 0`). Predicted to be low: a candidate must have been watched
while occupied and then vacated, so evidence and opportunity are anti-correlated.
"""
informed_move_fraction(model) =
    model.n_moves[] == 0 ? 0.0 : model.n_informed_moves[] / model.n_moves[]

"Investor density `ρ = n_investors / prod(dims)` — the supply of vicarious evidence."
density(model) = nagents(model) / prod(model.dims)

"Mean of `f` over investors; 0.0 for an empty model."
function mean_over(f, model)
    total, count = 0.0, 0
    for a in allagents(model)
        total += f(a)
        count += 1
    end
    return count == 0 ? 0.0 : total / count
end

end # module
