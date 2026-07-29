"""
    BusinessInvestor

Model logic for the business-investor ABM described in
`models/The business investor model.md`.

Investors live on a grid whose patches are *businesses*, each with a fixed
`profit` and `risk`. Profit may be negative — a loss-making business. Every step
has two phases: investors first reposition to a neighbouring business offering a
better profit/risk trade-off (utility `U = (W + T*profit) * (1 - risk)^T`), then
do their accounting — wealth changes by the profit of the business they sit on,
they go bankrupt (wealth reset to zero) if that pushes wealth below zero, and
with probability `risk` the wealth is wiped to zero anyway. At most one investor
occupies a patch at a time.

This module is realization- and visualization-agnostic (no Makie dependency), so
single-run and ensemble scripts can reuse it. Tunable parameters are `initialize`
keyword arguments; the ones the step functions read at run time (`profit`, `risk`,
`T`) live in the model's `properties`.
"""
module BusinessInvestor

using Agents, Random

export Investor, initialize, investor_step!, accounting_step!, utility

# Investors carry only their wealth; position comes from GridAgent.
@agent struct Investor(GridAgent{2})
    wealth::Float64
end

"""
    utility(wealth, profit, risk, T) -> Float64

Value an investor assigns to a business: `(wealth + T*profit) * (1 - risk)^T`.
`T` is the (fixed) planning horizon: higher profit raises the value, higher risk
lowers it through the survival factor `(1 - risk)^T`.

This is exactly the expected wealth after `T` steps, because both ruin and
bankruptcy reset wealth to zero: survive with probability `(1 - risk)^T` and hold
`wealth + T*profit`, otherwise hold `0`. With negative profits `wealth + T*profit`
can be negative, and then the formula prefers *higher* risk — an investor facing
certain losses would rather be wiped out. That is a genuine consequence of the
reset-to-zero rule, not a special case, so the formula is left unchanged.
"""
utility(wealth, profit, risk, T) = (wealth + T * profit) * (1 - risk)^T

"""
    investor_step!(investor, model)

Phase 1 (repositioning). Among the *empty* patches within `model.sensing_radius`
(Chebyshev distance), move to the one with the highest utility if it strictly
beats staying put; otherwise stay. Ties between qualifying patches are broken at
random. A `sensing_radius` of 0 leaves no candidates, so the investor never
relocates; 1 is the original Moore-neighbour rule; larger values let an investor
jump directly to the best business it can see.
"""
function investor_step!(investor::Investor, model::StandardABM)
    rng = abmrng(model)
    P, F, T = model.profit, model.risk, model.T

    here = investor.pos
    # Baseline: the utility of staying. A patch must strictly beat this to justify
    # a move, which is what "stay if current is best" means.
    best_u = utility(investor.wealth, P[here...], F[here...], T)
    candidates = Tuple{Int,Int}[]

    for p in nearby_positions(investor, model, model.sensing_radius)
        isempty(p, model) || continue  # one investor per patch
        u = utility(investor.wealth, P[p...], F[p...], T)
        if u > best_u
            best_u = u
            empty!(candidates)
            push!(candidates, p)
        elseif u == best_u && !isempty(candidates)
            push!(candidates, p)  # tie among neighbours already better than staying
        end
    end

    isempty(candidates) && return  # current patch is best: stay
    move_agent!(investor, rand(rng, candidates), model)
    return
end

"""
    accounting_step!(model)

Phase 2 (accounting). Each investor's wealth changes by the profit of the
business it occupies — which may be a loss. An investor whose wealth is pushed
below zero **goes bankrupt**: its wealth is reset to zero and it keeps playing.
Then, with probability equal to that business's risk, the wealth is wiped to zero
by a ruin event. Investors are independent, so the order is irrelevant.
"""
function accounting_step!(model::StandardABM)
    rng = abmrng(model)
    for investor in allagents(model)
        pos = investor.pos
        investor.wealth += model.profit[pos...]
        # Bankruptcy: debts are not carried, the investor simply restarts at zero.
        if investor.wealth < 0
            investor.wealth = 0.0
        end
        if rand(rng) < model.risk[pos...]
            investor.wealth = 0.0
        end
    end
    return
end

"""
    initialize(; kwargs...) -> StandardABM

Build a fresh business-investor model.

Keyword arguments:
- `dims`: grid size in patches (default `(19, 19)`).
- `n_investors`: number of investors (default `100`).
- `T`: planning horizon in the utility function (default `5`).
- `sensing_radius`: Chebyshev radius of patches an investor can see and move to
  each step (default `1`; `0` disables repositioning).
- `profit_range`: `(lo, hi)` uniform range for each business's profit (default
  `(-5.0, 10.0)`; negative values are loss-making businesses).
- `risk_range`: `(lo, hi)` uniform range for each business's risk (default `(0.01, 0.1)`).
- `profit`, `risk`: optional precomputed matrices (override the ranges), e.g. to
  reuse one fixed business landscape across runs.
- `seed`: RNG seed.
"""
function initialize(;
    dims = (19, 19),
    n_investors = 100,
    T = 5,
    sensing_radius = 1,
    profit_range = (-5.0, 10.0),
    risk_range = (0.01, 0.1),
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

    properties = (; profit = P, risk = F, T = T, sensing_radius = sensing_radius)
    # Non-periodic: the business grid has real edges, not a wrap-around torus.
    space = GridSpace(dims, periodic = false)
    model = StandardABM(
        Investor,
        space;
        agent_step! = investor_step!,
        model_step! = accounting_step!,
        properties,
        rng,
        scheduler = Schedulers.Randomly(),
    )

    # One investor per patch: add_agent_single! places each on a distinct empty patch.
    for _ = 1:n_investors
        add_agent_single!(Investor, model, 0.0)
    end

    return model
end

end # module
