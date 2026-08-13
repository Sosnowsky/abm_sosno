#abm

# Rock–Paper–Scissors Learning Model — Specification

Two (or `N`) agents repeatedly play matches of rock–paper–scissors. Each agent's
**strategy** is a mixed strategy — a probability vector `x = (r, p, s)` with
`r + p + s = 1` — and after every match each agent **adjusts its strategy toward
a best response to what it saw the opponent throw**, at a rate set by a learning
rate `η`. The object of interest is the *trajectory of the strategy points in the
2-simplex*: RPS has a unique Nash equilibrium at the centre `(⅓, ⅓, ⅓)`, but
naive learners do not sit there — they chase each other around it.

> **Status.** Specification only, nothing implemented yet. The four design
> decisions that were open have been settled: **best-response (fictitious play)
> learning**, **fixed `K = 3` throws per match**, a **ternary plot** of the
> simplex, and **`N = 2` first** with the population version designed for but
> deferred. Alternatives are recorded as *Variants* where they might be revisited.

## Purpose

- **Does learning find the equilibrium?** The unique Nash equilibrium of RPS is
  the uniform mix. Ask whether two adaptive agents converge to it, orbit it, or
  spiral away from it.
- **What does the learning rate do?** `η` sets how hard an agent lunges at the
  counter to what it just saw. Small `η` should give slow, tight orbits; large
  `η` should throw the pair out toward the boundary of the simplex, where play is
  nearly pure and exploitable.
- **How much does the match structure matter?** A best-of-3 match gives you three
  observations of the opponent per update. Does the granularity of observation
  change the dynamics, or only the noise?
- **Visualise the strategy space.** The deliverable is an animation of the two
  strategy points moving in the simplex, not a number.

The model is a two-player, discrete-time, *stochastic* version of classical
**fictitious play** on RPS — the standard textbook example of a game whose
learning dynamics *cycle* rather than converge. The twist over the textbook
version is that beliefs are formed from a handful of noisy samples per step
rather than from the opponent's exact mix.

## The Game

Actions are `R`, `P`, `S` (indices 1, 2, 3). Row player's payoff:

|        | vs R | vs P | vs S |
|--------|------|------|------|
| **R**  |  0   | −1   | +1   |
| **P**  | +1   |  0   | −1   |
| **S**  | −1   | +1   |  0   |

Zero-sum and symmetric. Unique Nash equilibrium: both players uniform,
`x* = (⅓, ⅓, ⅓)`. Expected payoff at equilibrium is 0, and — importantly — at
`x*` *every* opponent strategy earns exactly 0, so there is no gradient pointing
back toward the centre. That is the source of the cycling.

## Entities, State Variables, and Scales

One kind of entity: the **Player**.

| Entity | State variables |
|--------|-----------------|
| **Player** (agent) | `strategy::SVector{3,Float64}` — mixed strategy `(r,p,s)`, on the simplex |
| | `belief::SVector{3,Float64}` — running estimate of the opponent's mix |
| | `wins::Int`, `losses::Int`, `draws::Int` — match bookkeeping |
| | `score::Float64` — cumulative round payoff (optional, for a payoff-based variant) |

- **No physical space.** Position, if any, is purely a visualisation device (see
  *Implementation Notes*).
- **Time step** = one **match** (a best-of-`K`). Within a step there are up to
  `K` **rounds** (single throws), which are not model steps.
- Run length `T` = 500–5000 matches; the interesting structure (orbits) shows up
  over hundreds of matches.
- `N = 2` for the base model; the spec keeps `N` general (see *Populations*).

## Process Overview and Scheduling

One model step, executed in `model_step!` (agents have no independent behaviour
between matches):

1. **Pair up.** `N = 2`: the only pair. `N > 2`: draw a random perfect matching
   (see *Populations*).
2. **Play the match.** Repeat **exactly `K = 3` times** — no early stopping,
   no replayed draws: each player draws an action independently from its own
   `strategy` (`rand(abmrng(model))` against the cumulative distribution). Score
   each round with the payoff table. The **match winner** is whoever won more
   rounds; equal round-wins is a **drawn match** (e.g. 1–1 with one drawn round).
   Fixing the throw count keeps the number of observations per step constant and
   the observed frequency `f` an unbiased estimate of the opponent's mix — a
   stop-at-2-wins rule would condition the sample on the outcome.
3. **Observe.** Each player records the multiset of the opponent's `K` throws as
   an empirical frequency `f ∈ Δ²`, e.g. two rocks and one paper → `f = (⅔,⅓,0)`.
   Own throws are also visible to the opponent — observation is symmetric and
   noiseless.
4. **Update beliefs** (all players, from pre-update state):

   ```
   belief ← (1 − λ)·belief + λ·f
   ```

5. **Update strategies** (all players, simultaneously):

   ```
   strategy ← (1 − η)·strategy + η·e_{BR(belief)}
   strategy ← (1 − ε)·strategy + ε·(⅓,⅓,⅓)          # optional exploration floor
   ```

   where `e_i` is the vertex of the simplex for action `i` and `BR` is defined
   below. Both operations are convex combinations of simplex points, so the
   strategy **stays on the simplex automatically** — no renormalisation or
   projection is ever needed.
6. **Collect** strategies, beliefs, match outcome.

Updates are **synchronous**: both players update from the same match's
observations, and neither sees the other's new strategy before the next match.

### The best-response map

Given a belief `q = (q_R, q_P, q_S)` about the opponent, expected round payoffs
are `u = A·q`:

```
u_R = q_S − q_P      u_P = q_R − q_S      u_S = q_P − q_R
```

`BR(q) = argmax_i u_i`, ties broken uniformly at random. Using `Σq = 1`, the
comparisons simplify to `u_R − u_P = 3q_S − 1` and `u_R − u_S = 1 − 3q_P`, so:

- `BR = R ⟺ q_S ≥ ⅓ and q_P ≤ ⅓`
- `BR = P ⟺ q_R ≥ ⅓ and q_S ≤ ⅓`
- `BR = S ⟺ q_P ≥ ⅓ and q_R ≤ ⅓`

The three lines `q_i = ⅓` are **concurrent at the Nash point**, so the simplex is
cut into **three 120° wedges** meeting at the centre, one per best response.
Nothing distinguishes the centre itself — the best response is discontinuous
exactly there. This is the geometric reason the dynamics rotate: as the
opponent's belief-point crosses from wedge to wedge, your target jumps to the
next vertex, and you are perpetually chasing a target a third of a turn away.

It also fixes **which way round** they go, and the answer is the opposite of the
obvious guess. Your opponent moves to whatever beats you, so you move to
whatever beats *that* — which is the action you already beat. From rock the
target is scissors, from scissors paper, from paper rock: a player's own
strategy cycles **`R → S → P`**, *against* the beats-order, not with it. With the
ternary layout below (`R`, `P`, `S` counter-clockwise) that is **clockwise**
motion, i.e. a negative angular velocity about Nash.

**Variants (not implemented, recorded for later).** Two other rules also "learn
from the opponent's observed behaviour", and both would slot in behind a
`rule = :best_response | …` parameter if the base model gets boring:

- **Replicator / payoff-proportional.** `x_i ← x_i·(1 + η(u_i − ū))`
  renormalised, with `u = A·f` from the last match and `ū = x·u`. Actions that
  would have scored well against what the opponent actually threw gain weight.
  Gives the classical replicator orbits — closed curves around Nash in
  continuous time (conserved quantity `r·p·s`), spiralling slowly outward under
  Euler discretisation. Smoother, and never reaches the boundary.
- **Match-outcome reinforcement.** Update on whether the match was won or lost,
  ignoring which throws produced it. Much less informative; mainly interesting as
  a "how little information is enough?" question.

## Design Concepts

- **Basic principle.** Learning dynamics in a zero-sum game with no pure
  equilibrium. The interest is in whether adaptation is *self-defeating*: any
  exploitable regularity in your play is punished, so no fixed strategy other
  than uniform is safe, and adaptive agents end up cycling.
- **Adaptive behaviour.** Each agent moves toward what would have been the best
  reply to the opponent's recent behaviour. Objectives (maximise expected round
  payoff) are explicit and identical for both agents.
- **Sensing.** Each agent observes the opponent's **actions**, not its strategy.
  It never sees `(r,p,s)` directly — only `K` samples from it per match. The gap
  between the two is where all the interesting noise comes from.
- **Learning.** `belief` is a memory of the opponent, exponentially discounted at
  rate `λ`. `λ = 1` means "react only to the last match"; small `λ` means a long
  memory, approaching classical fictitious play with a flat average.
- **Interaction.** Direct and symmetric: the only interaction is playing a match,
  and the only channel between agents is the throws each observes.
- **Stochasticity.** Two sources: (1) actions are *sampled* from mixed
  strategies, so an agent with strategy `(⅓,⅓,⅓)` still hands its opponent a
  lopsided empirical frequency most of the time (with `K = 3`, `f` can only take
  values in multiples of ⅓); (2) tie-breaking in `BR` and in the initial
  strategies. There is no environmental noise.
- **Observation.** The primary observable is the **path of each strategy point in
  the simplex** over time; secondary observables in *Outputs*.
- **Emergence.** Cycling around Nash, and the time-average of play converging to
  Nash while the instantaneous play never does.

## Parameters

| Parameter | Symbol | Default | Meaning |
|-----------|--------|---------|---------|
| `n_players` | `N` | `2` | Number of agents |
| `rounds_per_match` | `K` | `3` | Throws per match, always played out in full |
| `learning_rate` | `η` | `0.1` | Step size toward the best response, `∈ (0,1]` |
| `memory` | `λ` | `1.0` | Belief update weight on the newest match, `∈ (0,1]` |
| `exploration` | `ε` | `0.0` | Per-step pull back toward uniform; keeps strategies off the boundary |
| `n_matches` | `T` | `1000` | Model steps |
| `init` | — | `:random` | `:random` (uniform on the simplex, via `Dirichlet(1,1,1)`), `:uniform` (both at Nash), or explicit vectors |
| `seed` | — | — | RNG seed |

Notes on the defaults:

- `η = 0.1` is a compromise: `η = 1` means "always play the pure counter to the
  last match", which is maximally exploitable and produces a period-3 limit cycle
  around the boundary; `η → 0` gives near-continuous-time dynamics with tiny slow
  orbits, and is the regime to compare against the textbook ODE.
- `λ = 1` (memoryless) makes the model's only memory the strategy itself, which
  is already a discounted average of past best responses. Setting `λ < 1` adds a
  *second*, independent timescale — worth having, but it complicates the picture,
  so start memoryless.
- `ε = 0` by default. With `η` large and `ε = 0`, strategies can get arbitrarily
  close to the vertices; `ε > 0` bounds the orbit radius. Turn it on only if the
  boundary behaviour becomes a nuisance.
- `K = 3` is a **coarse** observation: `f` can only take values in multiples of
  ⅓, so a single match tells you very little about the opponent's mix. Under
  best-response learning `K` is therefore an **observation-noise knob**, not a
  strategic one — larger `K` gives a finer `f`, a steadier belief, and tighter
  orbits. It would only become strategically loaded under a rule that reads the
  *match* outcome rather than the throws.

## Initialization

1. Create `N` players with `wins = losses = draws = 0`.
2. Set `strategy` per `init`. For `:random`, sample uniformly on the simplex
   (`rand(Dirichlet(ones(3)))`, or normalise three `Exponential(1)` draws — *not*
   two uniform sorts unless done correctly).
3. Set `belief` to `(⅓,⅓,⅓)` (uninformative), *not* to the opponent's true
   strategy — beliefs must be learned.

For reproducible pictures, a good deliberate start is both agents at `(⅓,⅓,⅓)`
with `seed` fixed: the pair sits at Nash until sampling noise alone kicks them
off it, which demonstrates that Nash is **not attracting** here.

## Input Data

None. The model is closed and self-contained.

## Outputs / Observations

Per model step, per agent: `strategy`, `belief`, match result, cumulative round
payoff.

**Primary visualisation — the simplex.** Plot each agent's strategy as a point in
the 2-simplex, animated over matches, with a fading trail.

Use a **ternary (barycentric) plot** on an equilateral triangle: vertices
`R = (0,0)`, `P = (1,0)`, `S = (½, √3/2)`, and a strategy drawn at

```
(x, y) = r·R + p·P + s·S = (p + s/2, s·√3/2)
```

This treats all three actions alike, so a rotationally symmetric orbit *looks*
rotationally symmetric. It is a handful of lines in Makie — three edges, three
labels, one `scatter!` — and needs no ternary-plot package. Overlay the Nash
point at the centroid and, optionally, the three best-response wedge boundaries
radiating from it.

(For the record: plotting `(r, p)` on a right triangle would lose nothing
mathematically — `(r,p,s) ↦ (r,p)` is an affine bijection onto
`{r,p ≥ 0, r+p ≤ 1}` — but it breaks the game's three-fold symmetry visually,
making `s` look like a leftover and circular orbits look sheared. The
`(r,p) → (x,y)` conversion above is the only extra code the ternary version
costs.)

Derived time series worth recording:

- **Distance to Nash** `‖x − x*‖` per agent — the orbit radius. Growing ⇒
  spiralling out; constant ⇒ closed orbit; shrinking ⇒ convergence.
- **Angle around Nash** and its unwrapped cumulative value ⇒ **winding number**
  and mean angular velocity. The cleanest scalar summary of "it cycles", and the
  natural thing to plot against `η`.
- **Time-averaged strategy** `(1/t)Σ x_τ` — expected to converge to Nash even
  when `x_t` does not. Showing these two curves on the same simplex plot is the
  single most instructive figure in the model.
- **Win rate** over a sliding window, and cumulative payoff. In a symmetric
  zero-sum game between symmetric learners this should hover at 0, but the
  *variance* is informative.
- **Exploitability** of each agent: `max_i (A·x)_i` — how much a best-responding
  opponent could earn against its current mix. 0 exactly at Nash.

## Results

The predictions below were written before the model was implemented, so they can
be scored honestly. Measurements are over the second half of 4000-match runs,
averaged over both players and 24 seeds
(`scripts/rps_learning_rate_sweep.jl`); ✅ = held, ❌ = wrong.

1. ✅ **No convergence to Nash in strategies.** With `ε = 0` the pair orbits `x*`
   indefinitely. Nash is a fixed point of the *expected* dynamics, but sampling
   noise alone is enough to push a pair started exactly at `(⅓,⅓,⅓)` off it.
2. ❌ **…but not in the predicted direction.** The spec first said `R → P → S`.
   The measured rotation is consistently *negative* (clockwise), i.e. each
   player's strategy cycles **`R → S → P`**, for the reason now given under *The
   best-response map*: you move to what beats what beats you, which is the action
   you already beat. The two players stay a third of a cycle apart, leapfrogging
   — whoever moved last is momentarily winning.
3. ✅ **Time-average does converge.** `‖mean(x) − x*‖` falls as
   `0.018 → 0.008 → 0.004` at `T = 1000 → 4000 → 16000` matches (`λ = 1`; the
   `λ = 0.2` run is slightly closer at every `T`), roughly the `1/√T` of an
   unbiased average, while instantaneous play stays 0.17–0.42 away and does not
   settle at all. Plotting the two together is the clearest single picture in
   the model.
4. ✅ **Orbit radius grows with `η`**, monotonically over the whole range, and
   `η = 1` pins strategies exactly on the vertices (radius `√(2/3) ≈ 0.816`).
5. ✅ **Angular velocity grows with `η`**, roughly linearly for the memoryless
   learner in the mid range; the remembering learner is closer to `∝ η^0.8`. At
   `η = 1` the measure aliases (a +240° step is indistinguishable from −120°) and
   is excluded from the plot.
6. ✅ **Neither agent wins.** Mean score of player 1 over 300 seeds × 2000
   matches: `−3.6 ± 3.3`, i.e. `−0.0006` points per round played.

Two results the spec did not anticipate:

7. **The memory `λ` is what makes the chase coherent, not `η`.** With `λ = 1`
   (memoryless, the spec's default) the belief is one match's three throws, the
   best-response target flickers, and the strategies random-walk near Nash: the
   rotation is there but slow and buried in noise (`−0.026 ± 0.014` rad/match at
   `η = 0.1`). With `λ = 0.2` the same run locks into a clean triangular limit
   cycle (`−0.139 ± 0.003` rad/match, ~45 matches per loop) at 2.4× the radius.
   Memory is what turns a noisy drift into a visible orbit.
8. **`K` and `λ` are substitutes, and the spec's `K` prediction was wrong in
   detail.** It predicted larger `K` ⇒ smaller orbit. Measured: `K` barely moves
   the radius at all (0.17–0.19 across `K ∈ {1…81}` at `λ = 1`). What `K` moves
   is *coherence*: at `λ = 1` the rotation rate grows from `−0.026` at `K = 3` to
   `−0.31` at `K = 81`. At `λ = 0.2` changing `K` does essentially nothing
   (`−0.14` throughout) — the belief is already averaging over matches, so extra
   throws per match are redundant. More throws per match and more matches per
   belief buy the same thing. (`K = 1` is a special case: a single throw is a
   *pure* belief, so the target is the exact counter with no ties, which is why
   it rotates faster than `K = 3`.)

## Experiments

1. ✅ **Single run, animated** — `scripts/rps_single_run.jl`. Two panels at the
   same seed and `η = 0.12`, differing only in `λ`, plus a distance-to-Nash
   trace. Renders `rock_paper_scissors.mp4`.
2. ✅ **`η` sweep** and 3. ✅ **`K` sweep** — `scripts/rps_learning_rate_sweep.jl`,
   producing `rps_learning_rate_sweep.png` and the numbers quoted above.
4. **Asymmetric learners** (not yet run). Give the two agents different `η` or
   `λ` and ask who wins — `initialize` already accepts per-player vectors for
   both. Prediction, and a counter-intuitive one: the *slower* learner should do
   better, because being predictable is what gets you punished.

Later, if the two-player model is exhausted: the population version below, and
the replicator variant for comparison on identical seeds.

## Populations (`N > 2`) — deferred

Not in the first implementation, but the code should not make it painful to add:
`initialize` takes `n_players`, agent state carries a single belief vector, and
`match_step!` loops over pairs rather than hard-coding two agents. Decisions to
make when the time comes:

- **Pairing.** Random perfect matching each step (requires `N` even), or each
  agent plays every other (round robin, `N(N−1)/2` matches per step), or each
  agent plays one random opponent (so an agent may play more than once per step).
  **Default suggestion:** random perfect matching, `N` even.
- **Beliefs.** One belief *per opponent* (needs `N−1` belief vectors per agent,
  and pairing must repeat for it to be useful) or **one belief about "the
  population"**, updated from whoever you just played. **Default suggestion:** a
  single population-level belief — it is the interesting object, it makes the
  model a proper mean-field learning dynamic, and it keeps agent state
  fixed-size, which is why the base model already stores `belief` as one vector.
- **Visualisation.** `N` points in the simplex; expect the cloud to rotate
  coherently at first and then either lock into a rotating cluster or spread out,
  depending on `η`.

## Implementation Notes (Julia / Agents.jl)

- **No space.** Use `@agent struct Player(NoSpaceAgent)` and
  `StandardABM(Player; model_step! = match_step!, properties, rng)`. All the
  logic lives in `model_step!`, since a match involves two agents at once and
  `agent_step!` would double-update.
  - *Rejected alternative:* a `ContinuousSpace((1,1))` with the agent's
    **position** set to its strategy `(r, p)` and `move_agent!` as the update,
    which would make `abmplot`/`abmvideo` animate the strategy space for free.
    Cute, but the space knows nothing of the constraint `r + p ≤ 1` and the plot
    comes out as a right triangle, not the ternary one we want. Keep the model
    space-free and drive Makie directly.
- **Strategy storage.** `SVector{3,Float64}` from `StaticArrays` (already a
  dependency of Agents.jl) — immutable, fast, and the convex-combination updates
  are one-liners: `x = (1-η) * x + η * e[br]`.
- **Sampling an action.** With `N = 2` and a 3-outcome distribution, a manual
  inverse-CDF (`u = rand(rng); u < x[1] ? 1 : u < x[1]+x[2] ? 2 : 3`) avoids a
  `Distributions.jl` dependency in the hot loop.
- **Numerical drift.** Repeated convex combinations keep `Σx = 1` to floating
  point; add an `@assert isapprox(sum(x), 1)` in tests, don't renormalise in the
  loop.
- **RNG.** Always draw from `abmrng(model)` so runs are reproducible from `seed`.
- **Watch the `!`/`=` gotcha** in keyword arguments: `model_step! = match_step!`
  with spaces (see `CLAUDE.md`).
- **Files**, following repo convention:
  - `src/RockPaperScissors.jl` — agent type, `match_step!`, `initialize`,
    plotting-agnostic, everything a keyword argument. Also a small
    `ternary(x) -> Point2f` helper, since both scripts need it.
  - `scripts/rps_single_run.jl` — the animated ternary plot (headline figure).
  - `scripts/rps_learning_rate_sweep.jl` — the `η` ensemble.
  - `test/test_rock_paper_scissors.jl` — payoff table, `BR` map on known beliefs
    (including the wedge boundaries `q_i = ⅓`), simplex invariance after many
    updates, zero-sum accounting, and `ternary` mapping the three vertices to the
    right corners.

## References

- G. W. Brown, *Iterative solution of games by fictitious play* (1951) — the
  original best-response learning dynamic.
- J. Hofbauer & K. Sigmund, *Evolutionary Games and Population Dynamics* (1998)
  — RPS replicator orbits and the conserved quantity `r·p·s`.
- D. Fudenberg & D. Levine, *The Theory of Learning in Games* (1998) — fictitious
  play in zero-sum games, time-average convergence.
- B. Sinervo & C. M. Lively, *The rock–paper–scissors game and the evolution of
  alternative male strategies*, Nature **380**, 240 (1996) — the biological
  instance.
