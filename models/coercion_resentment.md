# Coercion Resentment Model — Summary

The simplest agent-based version of Schøyen (2021), *What limits the efficacy of
coercion?*, Cliometrica 15, 267–318, built to see which of the stylized facts in
Schøyen, Sirnes and Andersen (2026), *The inefficiency of moderate political
coercion*, Public Choice, it reproduces. Implemented in
`src/CoercionResentment.jl`.

## Notation

The model follows the 2021 paper's notation. The 2026 paper writes coercion as
`C ∈ [0, 1]`, which is `π / π_max` here; its `C` is not the resentment `C(π)`.

| Symbol | Meaning | Code |
|---|---|---|
| *ns*, *s* | the non-state and the state identity | `Family.ns` (Bool) |
| `q` | share of ns parents (the paper's `q_t`) | `model.q[]`, `ns_share` |
| `π`, `π_max` | coercion, on `[0, 10]` | `model.coercion[]`, `π_max` |
| `Δu = ū − u̲` | utility of a same-identity child over an other-identity one | `Δu` |
| `C(π)`, `π̂` | coercion resentment and its inflection point | `resentment`, `π̂` |
| `τ^ns`, `τ^s` | socialisation efforts | `model.τ_ns[]`, `model.τ_s[]` |
| `H(τ) = τ²/2k` | cost of effort | `effort` (= `k`) |
| `q*(π)` | steady-state ns share | `steady_state` |
| `π_q̲` | the non-confrontational level, minimum of `q*` (3.58) | |
| `π_q̄` | the maximum of `q*` past it (5.30) | |
| `π^e_q̲` | where `q*` is back at `q*(π_q̲)` (8.33) | |
| `(π_q̲, π^e_q̲)` | the inefficient interval | |
| `ρ`, `π^EC` | exogenous constraint and the authority's choice under it | `ρ`, `best_coercion` |
| `ρ(q)` | insurrection constraint (the paper's `ρ(q*(π))`) | `insurrection_constraint` |
| `L = 1 − q` | legitimacy (2026, with `L₀ = 0`, `L₁ = 1`) | `history(...).legitimacy` |
| `q_max` | the authority's tolerance (this model's, not the papers') | `q_max` |

## Purpose

A state penalises the non-state identity *ns* with coercion `π`. Coercion lowers
the payoff of raising a child into that identity, but it also breeds
**resentment** `C(π)`, which makes ns parents try harder. If resentment is
S-shaped (steeper than 2 somewhere), there is an **inefficient interval**
`(π_q̲, π^e_q̲)` of moderate coercion, on whose rising part more coercion makes the
ns group *larger*. The 2026 paper reports four signatures in country data:

1. legitimacy is lower at middle coercion than on either side (H1);
2. coercion is bimodally distributed, with few country-years in the middle (H2);
3. regimes survive worst at middle coercion (H3);
4. sudden crackdowns are preceded by gradual easing (Table 4).

The paper derives 1–3 from a *rational* regime with randomly drawn constraints.
It does not model the dynamics behind 4. The question here is whether boundedly
rational authorities produce all four, and which of them need the inefficient
interval at all.

## Entities, state variables and scales

| Entity | State variables |
|--------|-----------------|
| **Family** (agent, `N` per country) | `ns` (identity, Bool) |
| **Authority** (one per country, model-level) | coercion `π ∈ [0, 10]`, tolerance `q_max` |

- One model is one **country**; an ensemble of seeds is the cross-country panel.
- A time step is a **year**. A generation is 25 years.
- No space and no network. Every family sees the country-wide ns share `q`, as
  in the paper's "complete mixing".

## Process overview and scheduling

Each year:

1. **Families reproduce** (each with probability 1/25). The parent socialises the
   child into its own identity with probability `τ`. Otherwise the child copies a
   random member of the population (the paper's oblique transmission). Effort
   solves the paper's first-order conditions (4)–(5) under the cost
   `H(τ) = τ²/2k`: `τ^ns = k(1 − q)(Δu − π + C(π))`, `τ^s = k q (Δu + π)`.
2. **The authority observes** `q`, and whether the ns group is growing:
   `τ^ns > τ^s`, which is the sign of the expected flow
   `E[Δq] ∝ q(1 − q)(τ^ns − τ^s)`.
3. **The authority sets coercion**:
   - ns group not growing → **ease**: `π −= 0.1`, because coercion is costly;
   - growing and `q > q_max` → **crack down**: `π += 1.5`;
   - growing but tolerated → **hold**.
4. **Collapse** with probability `0.05 · q · C(π) / C(10)`, the resentful ns group
   relative to its largest possible size. The successor keeps `π` and draws a new
   tolerance `q_max ~ U(0.15, 0.55)`.

## Design concepts

- **Satisficing, not optimising.** The authority never computes `q*(π)`. It
  reacts only to whether its current policy makes the ns group grow.
- **Two time scales.** `q` is a stock that relaxes over about a century. The
  authority moves coercion every year, so it has to read the flow, not the stock.
  An authority that waits for the stock overshoots, which was the first version's
  failure.
- **Emergence.** Stable sawtooths sit only where `q*` falls with `π`: on the low
  branch (below `π_q̲`) and on the high branch (above `π_q̄`). On the rising part
  `(π_q̲, π_q̄)` a crackdown makes the ns group grow faster, so an alarmed
  authority escalates year after year until it is through. The rising part
  empties without anyone knowing it exists.

## Results (`scripts/coercion_ensemble.jl`)

- With `π` frozen the families settle at the paper's `q*(π)` (max error 0.015).
- **H2 reproduced, and needs the inefficient interval.** 10% of country-years
  have `3 ≤ π < 7`, against 47% in a control without one (`max C′ = 0.78`).
- **H1 only locally.** Legitimacy `L` falls from 0.71 to 0.65 between `π ≈ 3.5`
  and `π ≈ 4.5`, and it is monotone in the control. But the paper's form (the
  whole interval `(π_q̲, π^e_q̲)` lower than both sides) mostly fails. It holds
  only when authorities move slowly (`ease ≤ 0.05`, `crackdown ≤ 1`). A
  satisficing authority keeps `q` near its own `q_max`, so legitimacy across
  countries mostly reflects tolerances, not `q*(π)`.
- **H3 reproduced, but not discriminating.** Collapse is highest at `π ≈ 5–6`,
  but it also peaks mid-scale in the control, since the hazard `q·C` is
  hump-shaped anyway.
- **Table 4 reproduced, but not discriminating.** Coercion falls for about 10
  years before a crackdown, to about 0.3 below random high years, in both
  models. The easing comes from the authority's rule, not from resentment.
- **Path dependence.** Authorities with the same `q_max` end on the low or the
  high branch depending on where they start.

## The 2021 rules (`scripts/coercion_2021.jl`)

`initialize(; rule = …)` swaps the satisficing authority for one of the 2021
paper's. None of them has a `q_max`, collapse or turnover.

| `rule` | Each year the authority… | Reproduces |
|---|---|---|
| `:fixed` | leaves `π` at `π0` | Lemma 1: `q` settles at `q*(π)` (max error 0.011) |
| `:exogenous` | takes `π^EC`, the `π ≤ ρ` with the lowest `q*` | Restraint: every `ρ` in the inefficient interval gives `π_q̲ = 3.58` |
| `:insurrection` | the same under the insurrection constraint `ρ(q)`, a logistic drop from above the scale to `ρ_low` at `q ≈ ρ_tip` | Path dependence: from `q0 = 0.16` the authority ends at `π = 10`, from `q0 = 0.30` at 1.5 |

The choice is a table lookup: `best_coercion` reads the running minimum of `q*`
on a grid of `π_max/1000`. At the quadratic minimum `π_q̲` this misses the true
minimum of `q*` by about `1e-6`.

The paper writes the insurrection constraint as `ρ(q*(π))`, for tractability.
Here it reads the current `q`. With finite N, the paper's path dependence is
then only metastable. A fluctuation in `q` can carry a country across the drop
in `ρ(q)`. Of 20 runs from `q0 = 0.16` over 3,000 years, the share ending at low
coercion was:

| N | 250 | 500 | 1,000 | 2,000 |
|---|---|---|---|---|
| ended low | 75% | 85% | 65% | 25% |

20 runs per N is coarse, about ±0.1, so the dip at N = 250 is noise. The
reverse switch is rare: 2 of 20 runs from `q0 = 0.30` ended high, and only
at N = 250.

The default `rule = :satisficing` is unchanged. It reproduces the earlier runs
bit for bit.

## Differences from the papers

- The gap is the rising part of `q*`, `(π_q̲, π_q̄)` ≈ (3.6, 5.3), not the whole
  inefficient interval `(π_q̲, π^e_q̲)`. Satisficing authorities do sit on
  `(π_q̄, π^e_q̲)`, which is locally stable but dominated: a rational authority
  would drop to `π_q̲`.
- `q*(0) = ½`, so democracies have middling legitimacy, whereas in the data
  they have the highest. The paper flags the same mismatch.
- The authority reads the expected flow, without sampling noise. Reading the
  noisy realised change in `q` instead needs a memory of decades at `N = 1000`.

## The minimal version

Switching collapse off and giving every authority the same tolerance leaves six
free parameters: the resentment curve (`A`, `s`, `π̂`) and the authority's
`ease`, `crackdown` and `q_max`.

```julia
m = initialize(; collapse_rate = 0, q_max = (0.3, 0.3), seed = 1)
```

Countries then differ only in their starting coercion `π0`. The other settings
are a unit (`Δu`), a speed (`effort`, `generation`) or noise (`n_families`).
A one-at-a-time sweep over all of them changes no qualitative result.

In 200 countries × 1,200 years, measured with the JavaScript port of these
rules on the Minimal Coercion ABM page, not yet with this code:

- the band from `π_q̲` to `π_q̄` holds 5% of what an even spread would,
  against 120% without an inefficient interval;
- countries starting up to `π = 4` settle near 2.4, and those starting from
  `π = 6` near 7.3;
- coercion one year before a crackdown is 0.51 below random high-coercion years.

Only the gap depends on the inefficient interval.

Left out on purpose:

- **The insurrection constraint `ρ(q)` of the 2021 paper.** With a sharp drop it
  splits countries into two states even with no resentment at all, so it is a
  rival explanation of bimodality.
- **Collapse.** It is only needed for H3, which does not discriminate here.
