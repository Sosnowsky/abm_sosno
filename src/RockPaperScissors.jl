"""
    RockPaperScissors

Model logic for the rock–paper–scissors learning ABM described in
`models/rock_paper_scissors.md`.

Two (or `N`) players repeatedly play best-of-`K` matches. A player's state is a
**mixed strategy** `x = (r, p, s)` on the 2-simplex. After each match every
player folds the opponent's observed throws into a belief about the opponent's
mix, and steps its own strategy toward the best reply to that belief:

    belief   ← (1 - λ)·belief + λ·f            f = observed throw frequencies
    strategy ← (1 - η)·strategy + η·e_BR(belief)
    strategy ← (1 - ε)·strategy + ε·(⅓,⅓,⅓)    optional exploration floor

Both updates are convex combinations of points on the simplex, so the strategy
stays on the simplex exactly — nothing is ever renormalised or projected. The
Nash equilibrium `(⅓,⅓,⅓)` is a fixed point of the *expected* dynamics but is
not attracting: the players orbit it instead of settling on it.

The module is plotting-agnostic (no Makie dependency) so it can be reused by
single-run and parameter-sweep scripts alike. All tunable parameters are
`initialize` keyword arguments; the two the step function needs at run time
(`rounds_per_match`, `exploration`) live in the model's `properties`, while the
per-player ones (`learning_rate`, `memory`) live on the agents so that players
can be given *different* learning rates.
"""
module RockPaperScissors

using Agents, Random, StaticArrays, LinearAlgebra

export Player,
    initialize,
    match_step!,
    play_match!,
    best_response,
    expected_payoffs,
    sample_action,
    trajectory,
    ternary,
    distance_to_nash,
    angle_around_nash,
    unwrap,
    winding_rate,
    exploitability,
    running_mean,
    PAYOFF,
    NASH,
    VERTEX,
    ACTION_NAMES

# --- The game -------------------------------------------------------------

"Actions are indexed 1 = rock, 2 = paper, 3 = scissors."
const ACTION_NAMES = ("R", "P", "S")

"`PAYOFF[i, j]` is the row player's payoff when it plays `i` and the column
player plays `j`. Zero-sum and antisymmetric: `PAYOFF' == -PAYOFF`."
const PAYOFF = @SMatrix [
    0 -1 1
    1 0 -1
    -1 1 0
]

"The three pure strategies, i.e. the vertices of the simplex."
const VERTEX = (
    SVector(1.0, 0.0, 0.0),
    SVector(0.0, 1.0, 0.0),
    SVector(0.0, 0.0, 1.0),
)

"The unique Nash equilibrium of rock–paper–scissors: the uniform mix."
const NASH = SVector(1 / 3, 1 / 3, 1 / 3)

"""
    expected_payoffs(q) -> SVector{3,Float64}

Expected round payoff of each pure action against an opponent playing the mixed
strategy `q`, i.e. `PAYOFF * q`. Explicitly:
`u_R = q_S - q_P`, `u_P = q_R - q_S`, `u_S = q_P - q_R`.
"""
expected_payoffs(q) = PAYOFF * SVector{3,Float64}(q)

"""
    best_response(q, rng) -> Int

The action maximising the expected payoff against belief `q`, with ties broken
uniformly at random.

Because `sum(q) == 1`, the comparisons collapse to `u_R - u_P = 3q_S - 1` and
`u_R - u_S = 1 - 3q_P`, so the best response is `R` exactly when `q_S ≥ ⅓` and
`q_P ≤ ⅓` (and cyclically for `P` and `S`). The three lines `q_i = ⅓` all pass
through the Nash point, cutting the simplex into three 120° wedges — one per
best response — meeting at the centre. That is the geometric reason the
dynamics rotate rather than converge, and it also fixes which way round they
go: the opponent moves to whatever beats you, so you move to whatever beats
*that* — which is the action you already beat. From rock the target is
scissors, from scissors paper, from paper rock, so a player's own strategy
cycles `R → S → P`, against the beats-order rather than with it.
"""
function best_response(q, rng::AbstractRNG)
    u = expected_payoffs(q)
    best = maximum(u)
    nties = count(==(best), u)
    nties == 1 && return argmax(u)
    # Exact ties happen on the wedge boundaries (and at Nash, where all three
    # payoffs are 0); pick one of the tied actions uniformly.
    k = rand(rng, 1:nties)
    for i = 1:3
        if u[i] == best
            k -= 1
            k == 0 && return i
        end
    end
    return 3  # unreachable
end

"""
    sample_action(x, rng) -> Int

Draw a single throw from the mixed strategy `x` by inverse-CDF sampling. Hot
path, so it is written out rather than delegated to `Distributions`.
"""
function sample_action(x, rng::AbstractRNG)
    u = rand(rng)
    u < x[1] && return 1
    u < x[1] + x[2] && return 2
    return 3
end

# --- The agent ------------------------------------------------------------

"""
    Player

A rock–paper–scissors learner. Has no position: the only "space" in this model
is the strategy simplex, which is a state variable, not a habitat.

- `strategy`: the mixed strategy `(r, p, s)` actually played.
- `belief`: the running estimate of the *opponent's* mix, formed only from
  throws this player has seen. It is never set from the opponent's true
  strategy — that has to be learned.
- `learning_rate` (η), `memory` (λ): per-player, so two learners can differ.
- `wins`/`losses`/`draws`: match tallies. `score`: cumulative round payoff.
"""
@agent struct Player(NoSpaceAgent)
    strategy::SVector{3,Float64}
    belief::SVector{3,Float64}
    learning_rate::Float64
    memory::Float64
    wins::Int
    losses::Int
    draws::Int
    score::Int
end

# --- The submodels --------------------------------------------------------

"""
    play_match!(a, b, model) -> (throws_a, throws_b, net_a)

Play one match between `a` and `b`: exactly `model.rounds_per_match` throws,
always played out in full — no early stopping and no replaying of drawn rounds.
Stopping at "first to 2 wins" would condition the sample on the outcome and
bias the observed frequencies, which are the whole input to learning here.

Returns each player's throw counts and `net_a`, the sum of `a`'s round payoffs
(so `net_a > 0` means `a` won the match, `0` means the match was drawn, and
`b`'s net is `-net_a`).

Despite the name nothing on the agents is mutated; the `!` marks that the
model's RNG is advanced.
"""
function play_match!(a::Player, b::Player, model::StandardABM)
    rng = abmrng(model)
    throws_a = zeros(MVector{3,Int})
    throws_b = zeros(MVector{3,Int})
    net_a = 0
    for _ = 1:model.rounds_per_match
        i = sample_action(a.strategy, rng)
        j = sample_action(b.strategy, rng)
        throws_a[i] += 1
        throws_b[j] += 1
        net_a += PAYOFF[i, j]
    end
    return SVector(throws_a), SVector(throws_b), net_a
end

"""
    learn!(player, f, model)

Fold the opponent's observed throw frequencies `f` into `player`'s belief, then
step its strategy toward the best reply to that belief. Both steps are convex
combinations of simplex points, so `sum(strategy) == 1` is preserved to
floating-point.
"""
function learn!(player::Player, f::SVector{3,Float64}, model::StandardABM)
    λ = player.memory
    η = player.learning_rate
    ε = model.exploration

    belief = (1 - λ) * player.belief + λ * f
    strategy = (1 - η) * player.strategy + η * VERTEX[best_response(belief, abmrng(model))]
    if ε > 0
        strategy = (1 - ε) * strategy + ε * NASH
    end

    player.belief = belief
    player.strategy = strategy
    return player
end

"Record a match result on a player, given its net round payoff."
function record_result!(player::Player, net::Int)
    player.score += net
    if net > 0
        player.wins += 1
    elseif net < 0
        player.losses += 1
    else
        player.draws += 1
    end
    return player
end

"""
    match_step!(model)

One model step = one round of matches. Players are paired at random (for
`N = 2` there is only one pair; with an odd `N` one player sits out), each pair
plays a match, and then *both* players update.

Updates are **synchronous** by construction: the observations are taken from
the match that has already been played, so neither player can see the other's
new strategy before the next match. A persistent nonzero average payoff in a
symmetric run is the signature of that invariant being broken.
"""
function match_step!(model::StandardABM)
    K = model.rounds_per_match
    # Sort before shuffling so the pairing is reproducible from the seed alone
    # (agent iteration order is not guaranteed).
    ids = shuffle(abmrng(model), sort!(collect(allids(model))))

    for k = 1:2:(length(ids)-1)
        a, b = model[ids[k]], model[ids[k+1]]
        throws_a, throws_b, net_a = play_match!(a, b, model)

        record_result!(a, net_a)
        record_result!(b, -net_a)

        # Each player learns from what the *other* one threw.
        learn!(a, throws_b ./ K, model)
        learn!(b, throws_a ./ K, model)
    end
    return model
end

# --- Construction ---------------------------------------------------------

"Broadcast a scalar parameter over players, or check a per-player vector."
function per_player(value, n::Int, name::Symbol)
    if value isa Number
        return fill(float(value), n)
    end
    length(value) == n ||
        throw(ArgumentError("$name must be a scalar or have length n_players = $n"))
    return float.(collect(value))
end

"Sample a strategy uniformly from the simplex, i.e. from `Dirichlet(1, 1, 1)`."
function random_strategy(rng::AbstractRNG)
    x = SVector(randexp(rng), randexp(rng), randexp(rng))
    return x / sum(x)
end

"Validate and normalise a user-supplied strategy."
function as_strategy(x)
    v = SVector{3,Float64}(x)
    all(≥(0), v) || throw(ArgumentError("strategy $x has a negative probability"))
    isapprox(sum(v), 1; atol = 1e-8) ||
        throw(ArgumentError("strategy $x does not sum to 1"))
    return v / sum(v)
end

function initial_strategies(init, n::Int, rng::AbstractRNG)
    init === :random && return [random_strategy(rng) for _ = 1:n]
    init === :uniform && return fill(NASH, n)
    # Otherwise: one explicit strategy for everyone, or one per player.
    if length(init) == 3 && first(init) isa Number
        return fill(as_strategy(init), n)
    end
    length(init) == n ||
        throw(ArgumentError("init must be :random, :uniform, a strategy, or $n strategies"))
    return [as_strategy(x) for x in init]
end

"""
    initialize(; kwargs...) -> StandardABM

Build a fresh rock–paper–scissors learning model.

Keyword arguments:
- `n_players`: number of agents (2 in the base model; an odd number leaves one
  player idle each step).
- `rounds_per_match`: throws per match, `K`. Under best-response learning this
  is an *observation-noise* knob, not a strategic one: it sets how finely the
  opponent's mix can be estimated from one match (with `K = 3`, only in
  multiples of ⅓).
- `learning_rate`: `η ∈ (0, 1]`, how far to step toward the best reply. Scalar,
  or one value per player.
- `memory`: `λ ∈ (0, 1]`, the weight the belief puts on the newest match.
  `λ = 1` is memoryless (react only to the last match). Scalar or per player.
- `exploration`: `ε ∈ [0, 1]`, a per-step pull back toward the uniform mix.
  `0` lets strategies approach the vertices; `ε = 1` pins them at Nash.
- `init`: `:random` (uniform on the simplex), `:uniform` (everyone at Nash), a
  single strategy, or one strategy per player.
- `seed`: RNG seed.
"""
function initialize(;
    n_players = 2,
    rounds_per_match = 3,
    learning_rate = 0.1,
    memory = 1.0,
    exploration = 0.0,
    init = :random,
    seed = 1234,
)
    n_players >= 2 || throw(ArgumentError("need at least 2 players"))
    rounds_per_match >= 1 || throw(ArgumentError("rounds_per_match must be ≥ 1"))
    0 <= exploration <= 1 || throw(ArgumentError("exploration must be in [0, 1]"))
    isodd(n_players) &&
        @warn "n_players is odd: one player sits out each step under random pairing"

    ηs = per_player(learning_rate, n_players, :learning_rate)
    λs = per_player(memory, n_players, :memory)
    all(η -> 0 <= η <= 1, ηs) || throw(ArgumentError("learning_rate must be in [0, 1]"))
    all(λ -> 0 < λ <= 1, λs) || throw(ArgumentError("memory must be in (0, 1]"))

    rng = MersenneTwister(seed)
    strategies = initial_strategies(init, n_players, rng)

    properties = (;
        rounds_per_match = Int(rounds_per_match),
        exploration = float(exploration),
    )
    model = StandardABM(Player; model_step! = match_step!, properties, rng)

    for i = 1:n_players
        # Beliefs start uninformative — a player must learn its opponent.
        add_agent!(Player, model, strategies[i], NASH, ηs[i], λs[i], 0, 0, 0, 0)
    end
    return model
end

"""
    trajectory(model, T) -> Matrix{SVector{3,Float64}}

Run `T` matches and return the strategy of every player after every one of
them, as a `(T + 1) × n_players` matrix whose first row is the initial state.
Columns are ordered by agent id, so column `j` is always the same player.
"""
function trajectory(model::StandardABM, T::Int)
    ids = sort!(collect(allids(model)))
    xs = Matrix{SVector{3,Float64}}(undef, T + 1, length(ids))
    for (j, id) in enumerate(ids)
        xs[1, j] = model[id].strategy
    end
    for t = 1:T
        Agents.step!(model, 1)
        for (j, id) in enumerate(ids)
            xs[t+1, j] = model[id].strategy
        end
    end
    return xs
end

# --- Observation helpers --------------------------------------------------

"""
    ternary(x) -> SVector{2,Float64}

Barycentric coordinates of a strategy in an equilateral triangle with
`R = (0, 0)`, `P = (1, 0)`, `S = (½, √3/2)`. Unlike plotting `(r, p)` directly,
this treats the three actions symmetrically, so a rotationally symmetric orbit
looks rotationally symmetric.
"""
ternary(x) = SVector(x[2] + x[3] / 2, x[3] * sqrt(3) / 2)

const TERNARY_NASH = ternary(NASH)

"Euclidean distance from a strategy to the Nash equilibrium: the orbit radius."
distance_to_nash(x) = norm(SVector{3,Float64}(x) - NASH)

"""
    angle_around_nash(x) -> Float64

Angle of a strategy about the Nash point in the ternary plane, in `(-π, π]`.
The vertices are laid out counter-clockwise in the order `R` (−150°),
`P` (−30°), `S` (90°), but learners travel the *other* way round — `R → S → P`,
see [`best_response`](@ref) — so a coherent chase shows up as a **negative**
mean angular velocity.
"""
function angle_around_nash(x)
    v = ternary(x) - TERNARY_NASH
    return atan(v[2], v[1])
end

"""
    unwrap(θ) -> Vector{Float64}

Remove the ±2π jumps from a sequence of angles so the cumulative rotation can
be read off directly.
"""
function unwrap(θ::AbstractVector{<:Real})
    out = Vector{Float64}(undef, length(θ))
    isempty(θ) && return out
    out[1] = θ[1]
    offset = 0.0
    for i = 2:length(θ)
        d = θ[i] - θ[i-1]
        d > π && (offset -= 2π)
        d < -π && (offset += 2π)
        out[i] = θ[i] + offset
    end
    return out
end

"""
    winding_rate(xs) -> Float64

Mean angular velocity around Nash, in radians per match, for a sequence of
strategies. The cleanest scalar summary of "it cycles"; `2π` per unit means one
full loop of the simplex per match.
"""
function winding_rate(xs::AbstractVector)
    length(xs) < 2 && return 0.0
    θ = unwrap([angle_around_nash(x) for x in xs])
    return (θ[end] - θ[1]) / (length(θ) - 1)
end

"""
    exploitability(x) -> Float64

The expected round payoff a best-responding opponent could extract from a
player using `x`. Zero exactly at Nash and nowhere else, so it doubles as a
distance-to-equilibrium measure in payoff units.
"""
exploitability(x) = maximum(expected_payoffs(x))

"""
    running_mean(xs) -> Vector

The cumulative time average of a sequence. Under fictitious play in a zero-sum
game this converges to Nash even though the instantaneous strategy never does —
plotting both on the same simplex is the point of the exercise.
"""
function running_mean(xs::AbstractVector)
    out = similar(xs)
    acc = zero(first(xs))
    for (i, x) in enumerate(xs)
        acc = acc + x
        out[i] = acc / i
    end
    return out
end

end # module
