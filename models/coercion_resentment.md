# Coercion Resentment Model — Summary

The simplest agent-based version of Schøyen (2021), *What limits the efficacy of
coercion?*, Cliometrica 15, 267–318, built to see which of the stylized facts in
Schøyen, Sirnes and Andersen (2026), *The inefficiency of moderate political
coercion*, Public Choice, it reproduces. Implemented in
`src/CoercionResentment.jl`.

## Purpose

A state penalises a non-state identity with coercion `π`. Coercion lowers the
payoff of raising a child into that identity, but it also breeds **resentment**
`C(π)`, which makes minority parents try harder. If resentment is S-shaped
(steeper than 2 somewhere), there is an **inefficient window** of moderate
coercion in which more coercion makes the minority *larger*. The 2026 paper
reports four signatures in country data:

1. legitimacy is lower at middle coercion than on either side (H1);
2. coercion is bimodally distributed, with few country-years in the middle (H2);
3. regimes survive worst at middle coercion (H3);
4. sudden crackdowns are preceded by gradual easing (Table 4).

The paper derives 1–3 from a *rational* regime with randomly drawn constraints.
It does not model the dynamics behind 4. The question here is whether boundedly
rational regimes produce all four, and which of them need the window at all.

## Entities, state variables and scales

| Entity | State variables |
|--------|-----------------|
| **Family** (agent, `N` per country) | `minority` (identity, Bool) |
| **Regime** (one per country, model-level) | coercion `π ∈ [0, 10]`, tolerance `q̄` |

- One model is one **country**; an ensemble of seeds is the cross-country panel.
- A time step is a **year**. A generation is 25 years.
- No space and no network. Every family sees the country-wide minority share
  `q`, as in the paper's "complete mixing".

## Process overview and scheduling

Each year:

1. **Families reproduce** (each with probability 1/25). The parent socialises the
   child into its own identity with probability `τ`. Otherwise the child copies a
   random member of the population (the paper's oblique transmission). Effort
   is the best reply under a quadratic cost: `τ_m = k(1 − q)(Δu − π + C(π))`,
   `τ_M = k q (Δu + π)`.
2. **The regime observes** `q`, and whether the minority is growing: `τ_m > τ_M`,
   which is the sign of the expected flow `E[Δq] ∝ q(1 − q)(τ_m − τ_M)`.
3. **The regime sets coercion**:
   - minority not growing → **ease**: `π −= 0.1`, because coercion is costly;
   - growing and `q > q̄` → **crack down**: `π += 1.5`;
   - growing but tolerated → **hold**.
4. **Collapse** with probability `0.05 · q · C(π) / C(10)`, the resentful minority
   relative to its largest possible size. The successor keeps `π` and draws a new
   tolerance `q̄ ~ U(0.15, 0.55)`.

## Design concepts

- **Satisficing, not optimising.** The regime never computes `q*(π)`. It reacts
  only to whether its current policy makes the minority grow.
- **Two time scales.** The minority share is a stock that relaxes over about a
  century. The regime moves coercion every year, so it has to read the flow, not
  the stock. A regime that waits for the stock overshoots, which was the first
  version's failure.
- **Emergence.** Stable sawtooths sit only where `q*` falls with `π`: on the low
  branch and on the high branch. Inside the window a crackdown makes the
  minority grow faster, so an alarmed regime escalates year after year until it
  is through. The window empties without anyone knowing it exists.

## Results (`scripts/coercion_ensemble.jl`)

- With `π` frozen the families settle at the paper's `q*(π)` (max error 0.015).
- **H2 reproduced, and needs the window.** 10% of country-years have
  `3 ≤ π < 7`, against 47% in a control without a window (`max C′ = 0.78`).
- **H1 only locally.** Legitimacy falls from 0.71 to 0.65 between `π ≈ 3.5`
  and `π ≈ 4.5`, and it is monotone in the control. But the paper's form (the
  whole window `[π_q, π_qe]` lower than both sides) mostly fails. It holds
  only when regimes move slowly (`ease ≤ 0.05`, `crackdown ≤ 1`). A satisficing
  regime keeps `q` near its own tolerance, so legitimacy across countries
  mostly reflects tolerances, not `q*(π)`.
- **H3 reproduced, but not discriminating.** Collapse is highest at `π ≈ 5–6`,
  but it also peaks mid-scale in the control, since the hazard `q·C` is
  hump-shaped anyway.
- **Table 4 reproduced, but not discriminating.** Coercion falls for about 10
  years before a crackdown, to about 0.3 below random high years, in both
  models. The easing comes from the regime rule, not from resentment.
- **Path dependence.** Regimes with the same tolerance end on the low or the
  high branch depending on where they start.

## Differences from the papers

- The gap is the rising part of the window, `(π_q, peak)` ≈ (3.6, 5.3), not the
  whole `(π_q, π_qe)`. Satisficing regimes do sit on `(peak, π_qe)`, which is
  locally stable but dominated: a rational regime would drop to `π_q`.
- `q*(0) = ½`, so democracies have middling legitimacy, whereas in the data
  they have the highest. The paper flags the same mismatch.
- The regime reads the expected flow, without sampling noise. Reading the
  noisy realised change in `q` instead needs a memory of decades at `N = 1000`.

## The minimal version

Switching collapse off and giving every regime the same tolerance leaves six
free parameters: the resentment curve (`A`, `s`, `π̂`) and the regime's `ease`,
`crackdown` and `q̄`.

```julia
m = initialize(; collapse_rate = 0, tolerance = (0.3, 0.3), seed = 1)
```

Countries then differ only in their starting coercion `π0`. The other settings
are a unit (`Δu`), a speed (`effort`, `generation`) or noise (`n_families`).
A one-at-a-time sweep over all of them changes no qualitative result.

In 200 countries × 1,200 years, measured with the JavaScript port of these
rules on the Minimal Coercion ABM page, not yet with this code:

- the band from `π_q` to the peak holds 5% of what an even spread would,
  against 120% without a window;
- countries starting up to `π = 4` settle near 2.4, and those starting from
  `π = 6` near 7.3;
- coercion one year before a crackdown is 0.51 below random high-coercion years.

Only the gap depends on the window.

Left out on purpose:

- **The insurrection limit `ρ(q)` of the 2021 paper.** With a sharp drop it
  splits countries into two states even with no resentment at all, so it is a
  rival explanation of bimodality.
- **Collapse.** It is only needed for H3, which does not discriminate here.
