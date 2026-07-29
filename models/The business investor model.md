#abm

# Business Investor Model — Summary

A toy economy in which **investors** move over a grid of **businesses**, each
business offering a fixed profit — which may be a **loss** — and a fixed risk of
ruin. Investors **cannot observe risk**. They hold a Bayesian belief about each
business's failure rate, updated from the only risk signal they actually receive:
the number of times they have been wiped out while sitting on it.

> **Status.** This document specifies the *Bayesian-belief* version of the model.
> The code in [`src/BusinessInvestor.jl`](../src/BusinessInvestor.jl) still
> implements the previous **omniscient-risk** version, in which investors read
> each business's true `risk` directly. Everything in "Predictions for the
> Bayesian version" is derived, not measured. The omniscient model is retained
> throughout as the control to compare against.

## Purpose

The previous version handed each investor the true failure rate *F* of every
business it could see. The results showed that mattered: chosen risk fell
steadily over a run, and **rich investors became risk-averse** without any rule
saying so. This version asks the follow-up:

- **How much of that survives when risk must be learned?** Investors see profit
  but not risk, and must infer *F* from their own history of going broke.
- **Is 25 steps of Bernoulli data enough to learn anything at all?** This turns
  out to be the decisive question, and the answer is no — see below.
- **What does the risk/return trade-off collapse into when the risk half is
  unknowable?**

The prior is deliberately **well-specified** — moment-matched to the true risk
distribution — so that the *only* thing investors lack is patch-specific
knowledge. Nothing here is driven by starting out wrong about the landscape.

## Entities, State Variables, and Scales

| Entity | State variables |
|--------|-----------------|
| **Business** (patch) | `profit` *P* (fixed, may be negative), `risk` *F* (fixed, **hidden from agents**) |
| **Investor** (agent) | `pos`, `wealth` *W* ≥ 0, and a **belief memory**: for every patch it has occupied, `n` = steps spent there and `k` = resets suffered there |

- Landscape: a **19 × 19 grid** of businesses, **non-periodic** (real edges, not
  a torus).
- **At most one investor per business** — patches are exclusive.
- Businesses never change: *P* and *F* are set at initialization and are
  constant for the whole run.
- Belief memory is **per investor**, keyed by absolute grid position, and
  **persists for the whole run** — an investor that abandons a patch keeps what
  it learned about it. Investors never share information.
- Default run length: **25 time steps**.

## Process Overview and Scheduling

Each time step runs two phases:

1. **Investor repositioning** (`agent_step!`, random order over investors).
   The investor looks at the *empty* patches within `sensing_radius`
   (Chebyshev distance — `r = 1` is the Moore neighbourhood) and moves to the
   one with the highest **believed** utility, **but only if it strictly beats
   staying put**. Ties among qualifying patches are broken at random. Order
   matters here only through the exclusion rule: an investor that moves first
   frees its old patch and blocks its new one.
2. **Accounting and belief update** (`model_step!`, after all moves). For every
   investor, on the patch *p* it now occupies:
   - `W ← W + P(p)` — a negative *P* is a loss.
   - If that leaves `W < 0` the investor **goes bankrupt**: `W` is reset to 0 and
     it keeps playing. Debts are never carried forward.
   - Independently, **with probability `F(p)` its wealth is wiped to zero** by a
     ruin event.
   - `n(p) ← n(p) + 1`, and `k(p) ← k(p) + 1` **if wealth ended this step at zero
     by either mechanism**.

   Investors are independent, so order is irrelevant.

## Design Concepts

### Basic principle

A risk/return trade-off in which **only the return half is observable**. Profit
accrues every step, risk destroys everything accumulated so far, and roughly a
third of the landscape is loss-making. The investor sees profit exactly and must
infer risk from experience.

### Sensing — what is and is not observable

| Quantity | Observable? |
|----------|-------------|
| `profit` *P* of any patch in radius | **Yes**, exactly |
| Occupancy of any patch in radius | **Yes** |
| `risk` *F* of any patch | **No** — never, not even for the patch it occupies |
| Own wealth, own reset history per patch | **Yes** |
| Anything outside `sensing_radius`, or any other investor's experience | **No** |

Removing *F* is the entire change. Everything else the investor could see before,
it still sees.

### Adaptive behavior — the Bayesian belief

Each ruin event is a Bernoulli trial with the patch's unknown rate *F*, so the
conjugate prior is a Beta. For a patch on which the investor has spent `n` steps
and suffered `k` resets:

$$F \mid \text{data} \;\sim\; \mathrm{Beta}(\alpha_0 + k,\; \beta_0 + n - k)$$

The prior is **moment-matched to the true risk distribution** `U(0.01, 0.1)`,
which has mean `0.055` and variance `0.000675`:

$$\alpha_0 = 4.18, \qquad \beta_0 = 71.82$$

The investor collapses the posterior to its **mean**

$$\hat F = \frac{\alpha_0 + k}{\alpha_0 + \beta_0 + n} = \frac{4.18 + k}{76 + n}$$

and plugs it into the *unchanged* objective function:

$$U = (W + T P)\,(1 - \hat F)^{T}$$

with planning horizon `T = 5`. The formula is textually identical to the previous
model; only the risk term has changed, from a known constant to a point estimate.

An unvisited patch has `n = k = 0`, so `F̂ = 0.055` exactly, giving survival
factor `(1 − 0.055)^5 = 0.7536`. The true-`F` factor for a typical patch is
`0.7536` as well — **investors start essentially perfectly calibrated about risk
in general.** What they lack is any ability to tell one business from another.

### Why the belief barely moves — the central finding

This is the result the model is really about, and it is a statement about the
*data*, not about the prior.

An investor at a typical patch faces `F ≈ 0.055`. Over an entire 25-step run it
therefore observes, on average, **1.38 ruin events** — and those are spread over
however many patches it occupies. You cannot estimate a rate from one and a bit
events. Concretely, the prior is worth `s₀ = α₀ + β₀ = 76` pseudo-observations,
so the data would need `n = 76` steps *on a single patch* merely to draw level
with it. A 25-step run cannot get a third of the way there:

| `n` steps on one patch | `F̂` if `k = 0` | `F̂` if `k = 1` |
|---|---|---|
| 5 | 0.0516 | 0.0640 |
| 10 | 0.0486 | 0.0602 |
| **25** (a whole run) | **0.0414** | **0.0513** |
| 50 | 0.0332 | 0.0411 |
| 100 | 0.0237 | 0.0294 |
| 200 | 0.0151 | 0.0188 |

Across a full run `F̂` moves from `0.055` to somewhere in `[0.041, 0.051]` at
best. In utility terms that is a survival factor shifting from `0.7536` to at
most `0.8095` — a few percent, against profit differences that span an order of
magnitude. **The Bayesian layer is very nearly inert.**

This is not a defect to be tuned away. It is the honest answer to "can an
investor learn risk by living through it?", and the answer is: not on this
timescale, for *any* reasonable prior. A weaker prior would move faster but only
by overreacting to noise — with 1.38 expected events, whatever it concluded would
be mostly wrong. The information simply is not there.

### The incumbency effect all but vanishes

Because staying is evaluated with data and moving is evaluated at the prior, the
comparison is still slightly rigged in favour of standing still. An investor on a
patch with history `(n, k)` relocates to an unvisited neighbour only if

$$\frac{W + T P_{\text{new}}}{W + T P_{\text{here}}} \;>\; \left[\frac{1 - \hat F(n,k)}{1 - 0.055}\right]^{T} \;\equiv\; \Lambda(n,k)$$

With the matched prior this **incumbency threshold** stays within a few percent
of 1:

| `n` (clean steps) | 1 | 2 | 5 | 10 | 25 |
|---|---|---|---|---|---|
| `Λ(n, 0)` | 1.004 | 1.007 | 1.018 | 1.034 | 1.074 |

and one reset gives only `Λ(1,1) = 0.937` — a 6 % nudge to leave, not a stampede.
For comparison, a naive uniform `Beta(1,1)` prior would give `Λ(1,0) = 4.21` and
`Λ(1,1) = 0.132`: investors locked in after a single step and fled outright when
burned. **That dramatic behaviour was an artefact of prior misspecification, not
a consequence of hiding `F`.** With a correct prior it disappears.

### What is left: profit-ranking

Since every unvisited patch carries the identical factor `0.7536`, the risk term
is a **common constant** across all candidate moves and cancels out of the
comparison. The decision reduces to

$$\arg\max_p \;(W + T\,P(p))\;=\;\arg\max_p \;P(p)$$

So the model degenerates, to a very good approximation, into a **greedy local
profit climber**: the investor walks uphill on profit within its sensing radius
and stops at a local maximum, with `Λ ≈ 1` providing a slight hysteresis and the
weak flight reflex after a reset providing the only risk-driven motion in the
entire model.

This is the sharp version of the question the model was built to ask. Risk
selection in the omniscient model required *comparing* the risks of different
businesses. An investor who can only learn about a business by occupying it can
never make that comparison, because it has data on exactly one candidate — the
one it is already standing on. **Risk selection is not merely slowed by hiding
`F`; it is structurally impossible.**

### Confounded evidence — now a minor effect

The investor cannot tell a ruin event from its own bankruptcy: it observes only
"my wealth went to zero," and both increment `k`. On a loss-making business this
happens every step, so `k = n` and `F̂` climbs (`0.113` at `n = 5`, `0.289` at
`n = 25`) — the business is progressively misjudged as dangerous when it is
merely unprofitable.

Under the matched prior this rarely gets a chance to accumulate. A poor investor
on a loss-making patch has `W + T·P < 0`, so `U` is already **negative**, and any
neighbour with positive profit beats it immediately. Flight from loss-making
businesses is driven by the **profit term**, not by the mistaken risk inference,
and the investor is usually gone within a step. The confounding only bites when
an investor is pinned — all neighbours occupied or equally loss-making. (Under
the old `Beta(1,1)` prior this effect was substantial; here it is vestigial.)

### Objectives, prediction, learning

The horizon `T` remains the only forecast; investors do not simulate the future.
There is now nominally learning, but it is private, confined to patches actually
occupied, and — per the section above — too slow to matter over a default run.
There is no exploration drive: an investor never visits a patch to find out about
it, only to profit from it.

### Interaction and stochasticity

- **Interaction:** indirect only, via competition for patches. Investors never
  trade with, observe, or learn from each other. Two investors on the same patch
  at different times form independent beliefs about it.
- **Stochasticity:** the landscape (*P*, *F*), initial placement, scheduling
  order, tie-breaking, and the per-step ruin event. Bankruptcy and the belief
  update are both deterministic given the state. **The belief machinery consumes
  no random numbers**, which matters for the regression test below.

### Observation

Alongside the previous metrics (mean wealth, mean *chosen* profit, mean *chosen*
risk), the belief machinery makes several new quantities worth recording:

- **Mean posterior error** `|F̂(p) − F(p)|` over occupied patches, and its trend.
- **Mean posterior shift** `|F̂ − 0.055|` — how far beliefs travel from the prior
  at all. Predicted to be small.
- **Relocations per investor.**
- **Bankruptcies vs ruin events**, tallied separately by the *model* even though
  the investor cannot distinguish them, to quantify how badly `k` overstates *F*.

## Parameters

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `dims` | Grid size in businesses | `(19, 19)` |
| `n_investors` | Number of investors | `100` (single run), `10` (ensemble) |
| `T` | Planning horizon in *U* | `5` |
| `sensing_radius` | Chebyshev radius an investor can see and move to | `1` (`0` = frozen) |
| `profit_range` | Uniform range for each business's *P* | `(-5.0, 10.0)` |
| `risk_range` | Uniform range for each business's *F* | `(0.01, 0.1)` |
| `prior_mean` *m₀* | Mean of the Beta prior on a business's failure rate | `nothing` → `mean(risk_range)` = `0.055` |
| `prior_strength` *s₀* | Weight of that prior, in pseudo-observations | `nothing` → moment-matched = `76` |
| `nsteps` | Run length | `25` |
| `seed` | RNG seed | `1234` |

### Parameterizing the prior

The prior is specified as **(mean, strength)**, not as `(α₀, β₀)`. Nothing is lost
— the map is a bijection, `s₀ = α₀ + β₀` and `m₀ = α₀/s₀`, so every Beta remains
reachable — but the two coordinates separate two questions the `(α₀, β₀)`
parameterization confounds:

- **`prior_mean`** is a *calibration* claim: is the investor right about risk in
  the landscape on average?
- **`prior_strength`** is a *confidence* claim: how hard does it hold that belief
  before evidence arrives?

Moving `α₀` alone shifts both at once, which makes a sweep uninterpretable.
Separating them lets each experiment vary one axis.

Both default to `nothing` and are then **derived from `risk_range`**, so a naive
`initialize()` is unchanged and a new landscape stays matched automatically. For
`F ~ U(lo, hi)` with mean `m` and variance `v = (hi−lo)²/12`:

$$s_0 = \frac{m(1-m)}{v} - 1, \qquad \alpha_0 = m_0 s_0, \qquad \beta_0 = (1 - m_0) s_0$$

With the defaults this gives `s₀ = 76`, `α₀ = 4.18`, `β₀ = 71.82`. Every prior in
"Experiments worth running" is then a pair of coordinates rather than a magic
number:

| Prior | `(prior_mean, prior_strength)` |
|-------|-------------------------------|
| Matched (default) | `(0.055, 76)` |
| Right mean, weak — isolates strength | `(0.055, 2)` |
| Uniform `Beta(1,1)` — deliberate misspecification | `(0.5, 2)` |
| Optimistic `Beta(1,99)` | `(0.01, 100)` |

Two guards worth building in:

- Assert `α₀ > 0` and `β₀ > 0`. Additionally, **`α₀ < 1` is a shape change, not
  just a weaker belief**: the Beta density then diverges at zero, so the prior
  becomes a spike near `F ≈ 0` with a long tail rather than a vague bump. At the
  default mean this happens for `s₀ < 18.2`, which is most of the interesting
  sweep range — so flag it rather than let it be discovered.
- `s₀` is meaningless in isolation; it only means anything against `nsteps`.
  `s₀ = 76` is crushing over 25 steps and mild over 500.

### The knob that actually predicts behaviour

Report the **data weight**

$$w = \frac{n}{s_0 + n}$$

as a derived diagnostic alongside the metrics — the share of the final belief
that came from evidence rather than from the prior. It is what determines whether
learning shows up at all, and it is the natural x-axis for the prior-strength and
run-length sweeps. At the defaults (`s₀ = 76`, a full run of `n = 25` on one
patch) it is `0.25`: three quarters of the investor's belief is still the prior it
was born with.

Because `w` depends only on `n/s₀`, **`prior_strength` and `nsteps` are the same
lever as far as the belief is concerned** — but they are opposites
scientifically. Lowering `s₀` makes the agent learn faster and *worse*, swinging
on noise (with 1.38 expected ruin events, a weak prior concludes `F ≈ 0` about a
quarter of the time). Raising `nsteps` gives it more life to learn from. Given
the finding above, **`nsteps` is now as interesting a knob as `sensing_radius`**;
`prior_strength` is best understood as a *cognitive* parameter like `T` — how
much the investor presumes to know about a world it was dropped into — rather
than as a fact about the landscape.

## Initialization

- Each business draws `P ~ U(-5, 10)` and `F ~ U(0.01, 0.1)`, independently. The
  landscape mean profit is 2.5, the mean risk 0.055, and 1/3 of businesses are
  loss-making (`P < 0`).
- Investors are placed on distinct random empty patches with `W = 0` and **empty
  belief memory** — every patch, including their own, sits at the matched prior.

## Input Data

None — the business landscape is constant over the run.

## Outputs / Observations

### Baseline: the omniscient-risk model

**Measured** results from the previous version, in which investors read the true
*F* directly. Ensemble of 100 landscapes, 10 investors each, 25 steps.

![Business investor ensemble](../business_investor_ensemble.png)

| `r` | final mean wealth | chosen *P* (step 1 → 25) | final chosen *F* |
|-----|------------------|--------------------------|------------------|
| 0 | 44.9 | 2.4 → 2.4 | 0.0543 |
| 1 | 136.0 | 8.08 → 7.75 | 0.0300 |
| 2 | 157.7 | 8.85 → 8.30 | 0.0218 |
| 3 | 165.7 | 9.09 → 8.53 | 0.0193 |
| 5 | 169.4 | 9.24 → 8.66 | 0.0180 |

Chosen profit jumps immediately then drifts slightly *down*, while chosen risk
falls steadily throughout — the signature of investors trading profit away for
safety as `W` grows.

### Predictions for the Bayesian version

Derived, not measured. Stated so the implementation can be checked against them.

- **`r = 0` must reproduce the baseline exactly — 44.9.** Frozen investors never
  consult a belief, and the belief update draws no random numbers, so with the
  same seeds the RNG stream is untouched. This is a hard regression test, not a
  prediction.
- **Mean chosen risk stays at the landscape mean (≈ 0.055) for every radius**,
  against `0.030` at `r = 1` in the baseline. Risk cannot enter the ranking
  because every candidate patch carries the identical prior factor. Expect at
  most a slight decline (to perhaps `0.050`) from the weak post-reset flight
  reflex. **This is the headline prediction: direct knowledge of `F` was buying
  essentially all of the risk reduction.**
- **Mean chosen profit should be slightly *higher* than baseline and flat** —
  around `8.3` at `r = 1` versus the baseline's `8.08 → 7.75`. The baseline's
  downward drift was profit being traded for safety; with no safety to buy, the
  greedy climber simply stays at its local profit maximum.
- **Mean wealth drops roughly 20 % at `r = 1`.** Treating wealth as a renewal
  process — reset at rate `F`, growing at `P` between resets — gives
  `E[W_t] = P·(1−F)(1−(1−F)^t)/F`. That estimate reproduces the measured baseline
  to within 8 % (predicts 126 vs. 136 measured), so applied to the Bayesian case:

  | | `r = 1` | `r = 2` | `r = 5` |
  |---|---|---|---|
  | predicted `E[W₂₅]` | ~108 | ~116 | ~120 |
  | measured baseline | 136.0 | 157.7 | 169.4 |

  Note the gap **widens** with `r`: better vision buys profit either way, but in
  the baseline it also bought lower risk, and that half is now gone.
- **Posterior beliefs barely move.** Mean `|F̂ − 0.055|` should stay below about
  `0.015`, and mean posterior error `|F̂ − F|` should be roughly flat over the run
  rather than declining.
- **Relocations per investor should be close to the baseline's**, since both
  models are profit-climbing early; the difference is that the baseline keeps
  making small risk-motivated moves late in the run and this one does not.

### The caveat, stated plainly

The prior here is correct by construction, so nothing above is an artefact of
starting out wrong. But two modelling choices still bound the result:

1. **The plug-in mean discards posterior variance.** Using the exact expectation
   `E[(1−F)^T]` instead would give `0.7592` at the prior rather than `0.7536` —
   a negligible difference *for this prior*, but the two diverge sharply once the
   posterior is wide. Worth checking rather than assuming.
2. **25 steps is too short for the question to be interesting.** The model as
   specified answers "can risk be learned in 25 steps?" (no). Whether it can be
   learned *at all* is a different question, and needs experiment 1 below.

## Experiments worth running

1. **Run length.** The single most informative change. At `nsteps = 200`, an
   investor that settles can accumulate `n ≈ 100+` on one patch, finally
   outweighing the prior (`F̂ → 0.024` on a clean record). Does risk selection
   re-emerge, and how much of the baseline's advantage does it recover? The
   renewal estimate says the baseline/Bayesian wealth gap should *widen* with
   time (213 vs 143 at `t = 200`) unless learning genuinely bites.
2. **Prior strength sweep at fixed mean.** Hold the mean at `0.055` and vary
   `s₀ ∈ {2, 10, 76, 300}`. Since the mean is always right, this isolates
   responsiveness from calibration — and should show that faster learning buys
   nothing but noise at 25 steps, which is the claim made above.
3. **Deliberate misspecification.** Re-run with the uniform `Beta(1,1)` prior for
   contrast. Predicted: severe incumbency lock-in (`Λ(1,0) = 4.21`), investors
   freezing after one step, and "burned bridges" where one bad step permanently
   writes off a business. Quantifies how much of a model's behaviour can be
   manufactured purely by a badly chosen prior — a useful cautionary result.
4. **Plug-in mean vs. full expectation.** Replace `(1 − F̂)^T` with
   `E[(1−F)^T] = ∏_{i=0}^{T-1} (β+i)/(α+β+i)`. Negligible for the matched prior,
   large for weak ones; pairs naturally with experiment 2.
5. **Thompson sampling.** Draw `F̃ ~ Beta(α, β)` fresh each step instead of a
   point estimate. This is the standard fix for the explore/exploit problem, and
   is the only mechanism considered here that would give investors a *reason* to
   visit a patch in order to learn about it. Likely needed before risk selection
   can appear at any run length.

## Implementation Notes (Julia / Agents.jl)

Not yet implemented. Sketch of what changes from the current code:

- `GridSpace((19, 19), periodic = false)` with `Schedulers.Randomly()` — unchanged.
- Patch properties stay as model-level matrices (`profit`, `risk`) in
  `properties`; `risk` is now read **only** by `accounting_step!`, never by the
  agent's decision rule. That separation is the whole point and should be visible
  in the code.
- `@agent struct Investor(GridAgent{2})` gains a belief field, e.g.
  `beliefs::Dict{Tuple{Int,Int}, Tuple{Int,Int}}` mapping position → `(n, k)`.
  A `Dict` rather than a matrix, since an investor touches only a handful of the
  361 patches — most stay at the prior and need no storage.
- `utility` takes `(wealth, profit, n, k, T, α₀, β₀)` and computes `F̂` internally,
  so the prior is a parameter rather than a magic constant.
- `initialize` takes `prior_mean` and `prior_strength` (both `nothing` by
  default), resolves them against `risk_range` via the moment-matching formula in
  the Parameters table, and stores the resulting `α₀, β₀` in `properties`. Keep
  the *(mean, strength)* pair as the public interface and `(α₀, β₀)` as the
  internal representation — the step functions want the latter, the experiments
  want the former.
- Assert `α₀ > 0 && β₀ > 0`; warn (don't fail) when `α₀ < 1`, since that is a
  legitimate but qualitatively different prior shape.
- Expose the data weight `w = n/(s₀ + n)` as a model-level metric so the
  prior-strength and run-length sweeps share an x-axis.
- `accounting_step!` must update `(n, k)` for the occupied patch **after** the
  wealth update, and set `k += 1` when either mechanism zeroed the wealth. Record
  the two causes separately at model level for the observability metrics, even
  though the investor sees only their union.
- **Do not draw random numbers in the belief update** — the `r = 0` regression
  test above depends on the RNG stream being identical to the baseline's.
- `add_agent_single!` for the one-investor-per-patch initial placement;
  `isempty(p, model)` enforces it during repositioning — unchanged.
- Remember the Julia gotcha from CLAUDE.md: write `agent_step! = investor_step!`
  **with spaces** around `=`.
