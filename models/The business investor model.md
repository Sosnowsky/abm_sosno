#abm

# Business Investor Model — Summary

A toy economy in which **investors** move over a grid of **businesses**, each
business offering a fixed profit — which may be a **loss** — and a fixed risk of
ruin. Investors **cannot observe risk**. They hold a Bayesian belief about each
business's failure rate, updated from the only risk signal they actually receive:
the wipeouts they **witness within their sensing radius** — their own, and those
of the investors occupying nearby businesses. A business can only fail while
someone is standing on it, so an empty business emits no evidence at all.

> **Status.** This document specifies the *Bayesian-belief* version of the model.
> The code in [`src/BusinessInvestor.jl`](../src/BusinessInvestor.jl) still
> implements the previous **omniscient-risk** version, in which investors read
> each business's true `risk` directly — documented separately in
> [`The business investor model 0.md`](The%20business%20investor%20model%200.md).
> Everything in "Predictions for the Bayesian version" is derived, not measured.
> The omniscient model is retained throughout as the control to compare against.

## Purpose

The previous version handed each investor the true failure rate *F* of every
business it could see. The results showed that mattered: chosen risk fell
steadily over a run, and **rich investors became risk-averse** without any rule
saying so. This version asks the follow-up:

- **How much of that survives when risk must be learned?** Investors see profit
  but not risk, and must infer *F* from failures they witness.
- **Is 25 steps of Bernoulli data enough to learn anything at all?** Per patch,
  no — the arithmetic below is decisive and unchanged by the observation rule.
- **Does watching other investors restore the risk *comparison*?** This is the
  question the observation radius exists to ask. Learning only from your own
  patch gives you data on exactly one candidate, which makes ranking impossible
  regardless of how good the data is. Watching neighbours breaks that — but only
  partially, and only when there are neighbours to watch.
- **What does the risk/return trade-off collapse into when the risk half is
  only sporadically observable?**

The prior is deliberately **well-specified** — moment-matched to the true risk
distribution — so that the *only* thing investors lack is patch-specific
knowledge. Nothing here is driven by starting out wrong about the landscape.

## Entities, State Variables, and Scales

| Entity | State variables |
|--------|-----------------|
| **Business** (patch) | `profit` *P* (fixed, may be negative), `risk` *F* (fixed, **hidden from agents**) |
| **Investor** (agent) | `pos`, `wealth` *W* ≥ 0, and a **belief memory**: for every patch it has seen occupied, `n` = steps it observed that patch occupied and `k` = wipeouts it observed there |

- Landscape: a **19 × 19 grid** of businesses, **non-periodic** (real edges, not
  a torus).
- **At most one investor per business** — patches are exclusive.
- Businesses never change: *P* and *F* are set at initialization and are
  constant for the whole run.
- **A business fails only while occupied.** The ruin event is drawn per
  investor, on the patch it stands on; an empty business is never tested against
  its `risk` and therefore produces no observable event.
- Belief memory is **per investor**, keyed by absolute grid position, and
  **persists for the whole run** — an investor keeps what it learned about a
  patch after moving away or after the patch empties. Investors never
  *communicate*; information flows one way, by being seen.
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
2. **Accounting, then observation** (`model_step!`, after all moves). Two
   sub-phases, in order:

   **2a. Accounting.** For every investor, on the patch *p* it now occupies:
   - `W ← W + P(p)` — a negative *P* is a loss.
   - If that leaves `W < 0` the investor **goes bankrupt**: `W` is reset to 0 and
     it keeps playing. Debts are never carried forward.
   - Independently, **with probability `F(p)` its wealth is wiped to zero** by a
     ruin event.
   - The investor is flagged as *wiped this step* if either mechanism left its
     wealth at zero.

   Investors are independent here, so order is irrelevant.

   **2b. Observation.** Only once all accounting is done — a wipeout must have
   happened before it can be witnessed. Every investor scans the patches within
   `sensing_radius`, **including its own**, and for each one that is occupied:
   `n(p) ← n(p) + 1`, and `k(p) ← k(p) + 1` if that patch's occupant was flagged
   wiped. Empty patches are skipped entirely and contribute nothing. Observation
   only reads state, so order is again irrelevant.

   At `r = 0` the only patch in range is the investor's own, so the rule reduces
   exactly to "learn from your own history" — the new scheme is a strict superset
   of the old one.

## Design Concepts

### Basic principle

A risk/return trade-off in which **only the return half is directly
observable**. Profit accrues every step, risk destroys everything accumulated so
far, and roughly a third of the landscape is loss-making. The investor sees
profit exactly and must infer risk from failures it happens to witness — its
own, and its neighbours'.

### Sensing — what is and is not observable

| Quantity | Observable? |
|----------|-------------|
| `profit` *P* of any patch in radius | **Yes**, exactly |
| Occupancy of any patch in radius | **Yes** |
| **Wipeout of the occupant of any patch in radius** | **Yes**, as a bare event — the cause is not distinguishable |
| `risk` *F* of any patch | **No** — never, not even for the patch it occupies |
| Own wealth, own wipeout history | **Yes** |
| Another investor's *wealth*, profit history, or beliefs | **No** — only its wipeouts are visible |
| Anything on an **empty** patch beyond profit and vacancy | **No** — an empty business generates no risk evidence |
| Anything outside `sensing_radius` | **No** |

Removing *F* remains the entire change to the decision rule. The observation
radius adds a channel that did not exist before: **vicarious evidence**. Note
that `sensing_radius` now does double duty — it bounds both where an investor
may move and how far it can see failures. The two are deliberately tied, since
both represent the same notion of local reach.

### Adaptive behavior — the Bayesian belief

Each wipeout is a Bernoulli trial with the patch's unknown rate *F*, so the
conjugate prior is a Beta. For a patch the investor has observed occupied on `n`
steps, of which `k` ended in a wipeout:

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

A never-observed patch has `n = k = 0`, so `F̂ = 0.055` exactly, giving survival
factor `(1 − 0.055)^5 = 0.7536`. The true-`F` factor for a typical patch is
`0.7536` as well — **investors start essentially perfectly calibrated about risk
in general.** What they lack is the ability to tell one business from another.

### Why the belief barely moves — per patch, the data is not there

This is a statement about the *data*, not about the prior, and the observation
radius does not change it.

A patch under continuous observation at `F ≈ 0.055` yields, over an entire
25-step run, **1.38 wipeouts on average**. You cannot estimate a rate from one
and a bit events. Concretely, the prior is worth `s₀ = α₀ + β₀ = 76`
pseudo-observations, so the data would need `n = 76` steps of *observed
occupancy on a single patch* merely to draw level with it. A 25-step run cannot
get a third of the way there:

| `n` observed steps on one patch | `F̂` if `k = 0` | `F̂` if `k = 1` |
|---|---|---|
| 5 | 0.0516 | 0.0640 |
| 10 | 0.0486 | 0.0602 |
| **25** (a whole run) | **0.0414** | **0.0513** |
| 50 | 0.0332 | 0.0411 |
| 100 | 0.0237 | 0.0294 |
| 200 | 0.0151 | 0.0188 |

Across a full run `F̂` moves from `0.055` to somewhere in `[0.041, 0.051]` at
best. In utility terms that is a survival factor shifting from `0.7536` to at
most `0.8095`.

**Watching neighbours does not fix this, because it cannot.** `n` on any single
patch is capped by the run length no matter how many patches you watch —
observation breadth multiplies the *number* of patches you hold a belief about,
never the evidence behind any one of them. What vicarious observation buys is
not accuracy. It is **comparability**, which is a different and more valuable
thing.

### What breadth actually buys — and what it costs

Ranking does not need accurate estimates, only estimates that are *comparable*
and better than coin-flips. Take two patches at the extremes of the risk range,
both observed for a full `n = 25`. The true `F = 0.01` patch yields `E[k] = 0.25`
→ `F̂ = 0.0439`; the true `F = 0.1` patch yields `E[k] = 2.5` → `F̂ = 0.0661`. The
utility ratio between them is

$$\left(\frac{1 - 0.0439}{1 - 0.0661}\right)^{5} = 1.12$$

so a **fully informed risk comparison is worth about 12 %** — real, and on the
same order as the profit differences that survive once `W` is large. But the
signal is noisy: with cross-patch variance `v = 0.000675` and sampling variance
`m(1−m)/n = 0.00208` at `n = 25`, the reliability of `k/n` is

$$\frac{v}{v + m(1-m)/n} = 0.245 \quad\Longrightarrow\quad \mathrm{corr}(\hat F, F) \approx 0.50$$

A ranking by `F̂` is about **half signal, half noise**. Note that `0.245` is
precisely the data weight `w` defined below: with a moment-matched prior, the
Beta shrinkage factor *is* the reliability of the evidence. That is not a
coincidence and it is why `w` is the right diagnostic.

**The structural cost: evidence and opportunity are anti-correlated.** A patch
emits evidence only while occupied, and an investor may only move to a patch that
is *empty*. So a candidate carries risk data only in the narrow case where
someone sat there earlier, within view, and has since left. Everything else in
range is either occupied — informative but unreachable — or has always been empty
— reachable but mute, sitting at the prior forever. The investor's own patch, by
contrast, is observed every single step without fail.

This is the softened version of the previous model's central claim. Risk
selection is no longer *structurally impossible*; it is **rare and biased**,
confined to recently-vacated neighbours, and set against an incumbent patch that
is always the best-measured thing in view.

### Density is the real control on learning

How much vicarious evidence exists at all is set by **investor density**
`ρ = n_investors / prod(dims)`, which was previously just a sample-size knob.
An investor observes roughly `1 + ρ·[(2r+1)² − 1]` patch-steps per step:

| `n_investors` | `ρ` | `r = 1` | `r = 2` | `r = 5` |
|---|---|---|---|---|
| 100 (single run) | 0.277 | 3.2 | 7.6 | 34.2 |
| 10 (ensemble) | 0.028 | 1.2 | 1.7 | 4.3 |

At `ρ = 0.028` an investor sees about **5 neighbour-sightings across an entire
run**, scattered over different patches — `n = 0` or `1` on each, which is
indistinguishable from the prior. Vicarious learning is effectively switched off.
At `ρ = 0.277` and `r = 1` there are typically 2–3 occupied neighbours at any
moment, and a settled investor watching settled neighbours can reach `n ≈ 25` on
several of them.

**`n_investors` is therefore now a cognitive parameter, not just a scale
parameter** — and the baseline ensemble was measured at the low-density end,
which matters for every comparison in the Outputs section.

### The incumbency effect

Because the incumbent patch is observed every step while most candidates are
never observed at all, the comparison remains rigged in favour of standing
still. An investor on a patch with history `(n, k)` relocates to a never-observed
neighbour only if

$$\frac{W + T P_{\text{new}}}{W + T P_{\text{here}}} \;>\; \left[\frac{1 - \hat F(n,k)}{1 - 0.055}\right]^{T} \;\equiv\; \Lambda(n,k)$$

With the matched prior this **incumbency threshold** stays within a few percent
of 1:

| `n` (clean steps) | 1 | 2 | 5 | 10 | 25 |
|---|---|---|---|---|---|
| `Λ(n, 0)` | 1.004 | 1.007 | 1.018 | 1.034 | 1.074 |

and one wipeout gives only `Λ(1,1) = 0.937` — a 6 % nudge to leave, not a
stampede. For comparison, a naive uniform `Beta(1,1)` prior would give
`Λ(1,0) = 4.21` and `Λ(1,1) = 0.132`: investors locked in after a single step and
fled outright when burned. **That dramatic behaviour was an artefact of prior
misspecification, not a consequence of hiding `F`.** With a correct prior it
disappears.

Against a *vacated but observed* candidate the comparison is two-sided, and both
estimates are shrunk toward `0.055` by the same factor `w`, so the achievable
spread is the 12 % computed above rather than anything dramatic.

### Vicarious evidence is confounded too — and more cheaply acquired

The investor cannot tell a ruin event from its own bankruptcy: it observes only
"my wealth went to zero," and both increment `k`. Watching a neighbour is worse,
not better — it sees only that the occupant was wiped, with no access to that
investor's wealth or profit history at all. On a loss-making business a poor
occupant is wiped every step, so an observer accumulates `k = n` and concludes
the business is dangerous (`F̂ = 0.113` at `n = 5`, `0.289` at `n = 25`) when it
is merely unprofitable.

For the investor's *own* patch this rarely gets a chance to accumulate. A poor
investor on a loss-making patch has `W + T·P < 0`, so `U` is already **negative**,
and any neighbour with positive profit beats it immediately — flight is driven by
the profit term, and the investor is usually gone within a step.

Vicarious confounding is more durable, because the observer pays no price for
being wrong and so never corrects. It writes off a loss-making patch as
*dangerous* on top of correctly rejecting it as *unprofitable* — a redundant
error with no behavioural consequence. The one case where it bites is a patch
that is mildly loss-making for a poor occupant but attractive to a rich observer,
where the mistaken risk inference can veto a move the profit term would have
allowed. (Under the old `Beta(1,1)` prior this effect was substantial; here it
is minor.)

### Objectives, prediction, learning

The horizon `T` remains the only forecast; investors do not simulate the future.
Learning is private and confined to patches actually observed occupied, and —
per the sections above — too slow per patch to matter much over a default run.
There is still no exploration drive: an investor never moves to a patch in order
to find out about it. What has changed is that some information now arrives
**free**, without occupying anything, which partially substitutes for
exploration — it just cannot be aimed.

### Interaction and stochasticity

- **Interaction:** now two channels, both indirect. **Competition** for
  exclusive patches, as before; and an **informational externality** — an
  investor generates risk evidence for everyone watching, simply by occupying a
  business and surviving or not. Investors never trade with or communicate with
  each other, and never learn another's wealth or beliefs. Two investors watching
  the same patch from different positions form beliefs from the same event stream
  but over different windows, so their beliefs generally differ.
- **Stochasticity:** the landscape (*P*, *F*), initial placement, scheduling
  order, tie-breaking, and the per-step ruin event. Bankruptcy and the belief
  update are both deterministic given the state. **The belief machinery consumes
  no random numbers**, which matters for the regression test below.

### Observation

Alongside the previous metrics (mean wealth, mean *chosen* profit, mean *chosen*
risk), the belief machinery makes several new quantities worth recording:

- **Mean posterior error** `|F̂(p) − F(p)|` over observed patches, and its trend.
- **Mean posterior shift** `|F̂ − 0.055|` — how far beliefs travel from the prior
  at all. Predicted to be small.
- **Observed patch-steps per investor per step**, split into own vs vicarious —
  the direct measure of information supply, and the thing density controls.
- **Fraction of relocations that were *informed*** — i.e. the chosen patch had
  `n > 0`. Predicted to be low, and the cleanest test of the
  evidence/opportunity anti-correlation.
- **Relocations per investor.**
- **Bankruptcies vs ruin events**, tallied separately by the *model* even though
  neither the occupant nor any observer can distinguish them, to quantify how
  badly `k` overstates *F*.

## Parameters

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `dims` | Grid size in businesses | `(19, 19)` |
| `n_investors` | Number of investors — also sets information supply via density | `100` (single run), `10` (ensemble) |
| `T` | Planning horizon in *U* | `5` |
| `sensing_radius` | Chebyshev radius an investor can see, learn from, and move to | `1` (`0` = frozen, own-patch learning only) |
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

### The knobs that actually predict behaviour

Report two derived diagnostics alongside the metrics.

The **data weight**

$$w = \frac{n}{s_0 + n}$$

is the share of a belief that came from evidence rather than from the prior, and
— as shown above — also the reliability of that evidence under a matched prior.
It is the natural x-axis for the prior-strength and run-length sweeps. At the
defaults (`s₀ = 76`, a fully observed patch over `n = 25`) it is `0.25`: three
quarters of the belief is still the prior.

The **density** `ρ = n_investors / prod(dims)` sets how many patches ever reach a
non-trivial `n` at all. `w` describes the quality of a belief; `ρ` describes how
many beliefs exist. Both must be non-trivial for risk selection to appear, and
they are independent.

Because `w` depends only on `n/s₀`, **`prior_strength` and `nsteps` are the same
lever as far as a single belief is concerned** — but they are opposites
scientifically. Lowering `s₀` makes the agent learn faster and *worse*, swinging
on noise (with 1.38 expected wipeouts, a weak prior concludes `F ≈ 0` about a
quarter of the time). Raising `nsteps` gives it more life to learn from. Given
the findings above, **`nsteps` and `n_investors` are now as interesting as
`sensing_radius`**; `prior_strength` is best understood as a *cognitive*
parameter like `T` — how much the investor presumes to know about a world it was
dropped into — rather than as a fact about the landscape.

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

**This baseline was run at 10 investors, i.e. `ρ = 0.028`** — the density at
which vicarious learning barely operates. Any Bayesian run compared directly
against this table is therefore comparing against the *information-starved* end
of the new model, which is the right control but not the interesting regime.

### Predictions for the Bayesian version

Derived, not measured. Stated so the implementation can be checked against them.

- **`r = 0` must reproduce the baseline exactly — 44.9.** Frozen investors never
  consult a belief, they observe only their own patch, and the belief update
  draws no random numbers, so with the same seeds the RNG stream is untouched.
  This is a hard regression test, not a prediction.
- **At the ensemble density (10 investors), mean chosen risk stays near the
  landscape mean (≈ 0.055) for every radius**, against `0.030` at `r = 1` in the
  baseline. Almost every candidate patch carries the identical prior factor,
  because there is nobody nearby to generate evidence about it. Expect at most a
  slight decline (to perhaps `0.050`) from the weak post-wipeout flight reflex.
  **This remains the headline prediction: direct knowledge of `F` was buying
  essentially all of the risk reduction.**
- **At the single-run density (100 investors), expect a visible but partial
  recovery** — chosen risk falling to roughly `0.045–0.050` at `r = 1`, versus
  `0.030` for the omniscient model. The ceiling is set by two multiplied
  attenuations: rankings are only `corr ≈ 0.5` with the truth, and informed
  comparisons are available only on the minority of candidates that were watched
  before they emptied. **The gap between the two densities is the cleanest
  measurement this model offers**, and it isolates vicarious learning from
  everything else.
- **Fraction of informed relocations should be low** — well under half at
  `r = 1`, even at high density, since a candidate must be occupied-then-vacated
  within view to carry any data.
- **Mean chosen profit should be slightly *higher* than baseline and roughly
  flat** — around `8.3` at `r = 1` versus the baseline's `8.08 → 7.75`. The
  baseline's downward drift was profit being traded for safety; with much less
  safety to buy, the near-greedy climber mostly stays at its local profit
  maximum. Expect a small downward drift to reappear at high density.
- **Mean wealth drops roughly 20 % at `r = 1` at ensemble density.** Treating
  wealth as a renewal process — reset at rate `F`, growing at `P` between resets
  — gives `E[W_t] = P·(1−F)(1−(1−F)^t)/F`. That estimate reproduces the measured
  baseline to within 8 % (predicts 126 vs. 136 measured), so applied to the
  Bayesian case:

  | | `r = 1` | `r = 2` | `r = 5` |
  |---|---|---|---|
  | predicted `E[W₂₅]` | ~108 | ~116 | ~120 |
  | measured baseline | 136.0 | 157.7 | 169.4 |

  Note the gap **widens** with `r`: better vision buys profit either way, but in
  the baseline it also bought lower risk, and at low density that half is nearly
  gone. At high density the shortfall should narrow.
- **Posterior beliefs barely move, per patch.** Mean `|F̂ − 0.055|` should stay
  below about `0.015`, and mean posterior error `|F̂ − F|` should be roughly flat
  over the run rather than declining. What should rise with density is the
  *number* of patches holding a non-prior belief, not the strength of any one.
- **Relocations per investor should be close to the baseline's**, since both
  models are profit-climbing early; the difference is that the baseline keeps
  making small risk-motivated moves late in the run and this one does so only
  occasionally, when a watched neighbour vacates.

### The caveat, stated plainly

The prior here is correct by construction, so nothing above is an artefact of
starting out wrong. But three modelling choices still bound the result:

1. **The plug-in mean discards posterior variance.** Using the exact expectation
   `E[(1−F)^T]` instead would give `0.7592` at the prior rather than `0.7536` —
   a negligible difference *for this prior*, but the two diverge sharply once the
   posterior is wide. Worth checking rather than assuming.
2. **25 steps is too short for the per-patch question to be interesting.** The
   model as specified answers "can risk be learned in 25 steps?" (no, per patch).
   Whether it can be learned *at all* is a different question, and needs
   experiment 1 below.
3. **Only-occupied-businesses-fail is what makes evidence scarce.** It is a
   defensible reading — a business with no investor in it has nothing to lose —
   but it is the assumption doing most of the work in the
   evidence/opportunity anti-correlation. If businesses failed whether or not
   occupied, every patch in radius would carry `n = nsteps` and the comparison
   problem would largely dissolve.

## Experiments worth running

1. **Run length.** At `nsteps = 200`, a settled investor can accumulate
   `n ≈ 100+` on its own patch and on stable neighbours, finally outweighing the
   prior (`F̂ → 0.024` on a clean record). Does risk selection re-emerge, and how
   much of the baseline's advantage does it recover? The renewal estimate says
   the baseline/Bayesian wealth gap should *widen* with time (213 vs 143 at
   `t = 200`) unless learning genuinely bites. Longer runs also let more
   candidates pass through the occupied-then-vacated state that makes them
   knowable.
2. **Density sweep.** Vary `n_investors ∈ {10, 40, 100, 200}` at fixed `r = 1`,
   holding everything else at defaults. This is the experiment the observation
   radius exists to justify: it varies the *supply of vicarious evidence* while
   leaving each investor's cognition untouched. Predicted: chosen risk declining
   monotonically with density, from ≈ 0.055 toward ≈ 0.045, with the fraction of
   informed relocations rising alongside it. Confounded by competition for
   patches getting fiercer, so report both metrics.
3. **Prior strength sweep at fixed mean.** Hold the mean at `0.055` and vary
   `s₀ ∈ {2, 10, 76, 300}`. Since the mean is always right, this isolates
   responsiveness from calibration — and should show that faster learning buys
   nothing but noise at 25 steps.
4. **Deliberate misspecification.** Re-run with the uniform `Beta(1,1)` prior for
   contrast. Predicted: severe incumbency lock-in (`Λ(1,0) = 4.21`), investors
   freezing after one step, and "burned bridges" where one bad step permanently
   writes off a business — now including businesses the investor never occupied,
   written off on the strength of watching one neighbour get unlucky. Quantifies
   how much of a model's behaviour can be manufactured purely by a badly chosen
   prior.
5. **Plug-in mean vs. full expectation.** Replace `(1 − F̂)^T` with
   `E[(1−F)^T] = ∏_{i=0}^{T-1} (β+i)/(α+β+i)`. Negligible for the matched prior,
   large for weak ones; pairs naturally with experiment 3.
6. **Thompson sampling.** Draw `F̃ ~ Beta(α, β)` fresh each step instead of a
   point estimate. This is the standard fix for the explore/exploit problem. It
   matters less than it did under own-patch-only learning, since vicarious
   evidence already arrives without being sought — but it remains the only
   mechanism here that would give investors a reason to *aim* their attention.

## Implementation Notes (Julia / Agents.jl)

Not yet implemented. Sketch of what changes from the current code:

- `GridSpace((19, 19), periodic = false)` with `Schedulers.Randomly()` — unchanged.
- Patch properties stay as model-level matrices (`profit`, `risk`) in
  `properties`; `risk` is now read **only** by the accounting phase, never by the
  agent's decision rule. That separation is the whole point and should be visible
  in the code.
- `@agent struct Investor(GridAgent{2})` gains a belief field, e.g.
  `beliefs::Dict{Tuple{Int,Int}, Tuple{Int,Int}}` mapping position → `(n, k)`.
  A `Dict` rather than a matrix, since an investor observes only a handful of the
  361 patches — most stay at the prior and need no storage. Breadth raises the
  entry count relative to the own-patch-only scheme, but not near 361.
- `Investor` also needs a transient `wiped::Bool` flag, set during accounting and
  read during observation. It is the only channel by which one investor's outcome
  reaches another's beliefs, and it must be **cleared at the start of each
  accounting phase** so a stale flag is never witnessed twice.
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
- Expose the data weight `w = n/(s₀ + n)` and the density
  `ρ = n_investors/prod(dims)` as model-level metrics so the sweeps share axes.
- `model_step!` runs **two separate loops over all agents**, not one fused loop:
  accounting first for every investor, then observation for every investor. A
  fused loop would let an investor witness neighbours who have not yet done their
  accounting, making the belief update depend on scheduler order. This ordering
  is a correctness requirement, not a style choice.
- The observation loop scans `nearby_positions(investor, model, sensing_radius)`
  **plus the investor's own position** — `nearby_positions` excludes the centre,
  and the own patch is the single most important observation. Skip any position
  that is empty; for occupied ones, increment `n` and increment `k` if the
  occupant's `wiped` flag is set.
- **Do not draw random numbers in the belief update** — the `r = 0` regression
  test above depends on the RNG stream being identical to the baseline's.
- `add_agent_single!` for the one-investor-per-patch initial placement;
  `isempty(p, model)` enforces it during repositioning — unchanged.
- Remember the Julia gotcha from CLAUDE.md: write `agent_step! = investor_step!`
  **with spaces** around `=`.
