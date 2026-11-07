# Tourist Destinations Model — Summary

How social influence shapes where people go on holiday. Implemented in
`src/TouristDestinations.jl`.

## The model

`N = 1000` tourists each travel once a year to one of `D = 100` destinations.

- Tourist `n` holds a weight `w_nd` for every destination `d`, drawn at the start
  uniformly from `[0, 1]`: independent individual tastes.
- Each year every tourist throws a `D`-faced die with faces weighted by `w_nd` and
  goes where it lands.
- After the holidays everyone sees where everyone else went, and the weights
  move towards it by one of two rules (`update`):
  - **additive** (`:additive`, the default):

        w_nd ← w_nd + ε Σ_{n′≠n} [d_{n′} = d],

    i.e. by `ε` times the number of *other* tourists who went to `d` that year.
  - **mixing** (`:mixing`): `w_n` is kept a probability distribution (the initial
    tastes are normalised), and each year a fraction `ε ∈ [0, 1]` of it is
    replaced by the distribution of where everyone went:

        w_nd ← (1 − ε) w_nd + (ε/N) Σ_{n′} [d_{n′} = d].

  - **conformity** (`:mixing` with `β ≠ 1`): the same mixing, but towards a
    sharpened distribution of trips, in which popular destinations count more
    than their share of visitors:

        c_d = Σ_{n′} [d_{n′} = d],   s_d = c_d^β / Σ_d′ c_d′^β,
        w_nd ← (1 − ε) w_nd + ε s_d.

    `β = 1` is the mixing rule.

  `ε = 0` keeps the initial tastes for ever under every rule.

| Symbol | Meaning | Code |
|---|---|---|
| `N`, `D` | tourists, destinations | `n_tourists`, `n_destinations` |
| `w_nd` | tourist `n`'s weight on destination `d` | `Tourist.weights` |
| `ε` | influence parameter | `ε` |
| | update rule | `update` (`:additive`, `:mixing`) |
| `β` | conformity exponent (`:mixing` only) | `β` |
| `p_d` | popularity: mean over tourists of `w_nd / Σ_d′ w_nd′` | `popularity` |
| `1 / Σ p_d²` | effective number of destinations | `effective_number` |

Under the additive rule only the ratio of `ε` to the initial weights matters. Each year a tourist's
total weight grows by `ε(N − 1)`, against an initial total of about `D/2 = 50`,
so the others' choices outweigh a tourist's own taste after about `50 / (1000 ε)`
years: half a century at `ε = 0.001`, five years at `ε = 0.01`, and within the
first year at `ε ≳ 0.05`.

## The two rules differ in memory, not in normalisation

Choices depend only on the ratios of the weights, so normalising them changes
nothing by itself. What matters is how much weight each year gets. The additive
weights only grow, so year `t`'s trips are a share of about `1/t` of the total:
every past year counts the same. Normalising the additive weights after every
year gives exactly the mixing rule, with `ε_mixing = ε(N − 1)/(1 + ε(N − 1))`
(and the tourist's own trip left out): each year then has the same weight `ε`,
and older years fade as `(1 − ε)^age`.

## Additive rule: lock-in, not concentration

The reinforcement is linear, so the model is a **Pólya urn**. A destination's
expected share next year is its share of the weight accumulated so far, which is
a martingale. Popular places do not get disproportionately more popular. The
shares freeze within a few years, at values set by the luck of the first years.

- **ε = 0**: every year is a fresh draw from the same tastes. Popularity never
  changes and year-to-year differences are pure noise.
- **ε > 0**: the first years' random differences are locked in. At `ε = 1`, the
  correlation between a destination's visitors in year 1 and its popularity in
  year 100 is 0.78, although nobody prefers it.
- **No ε concentrates the tourists.** The mean-field prediction
  (`polya_prediction`) gives the effective number of destinations after `T`
  years as `D / (1 + (D − 1) S)`, with `S = Σ_t a_t² / N` and
  `a_t = εN / (W₀ + εN t)`. As `ε → ∞` it falls only to
  `D / (1 + (D − 1) Σ 1/t² / N) ≈ 86` of 100. The year-1 fluctuations are of
  relative size `√(D/N) ≈ 0.3`, because 1000 trips a year over 100 destinations
  average out the noise.

The ABM matches the prediction within 0.4 destinations and 0.03 in correlation
from `ε = 10⁻³` to `10²` (`scripts/tourist_influence_sweep.jl`).

## Mixing rule: drift towards one destination

The popularity still moves by `ΔX = ε (v/N − X)`, with expectation zero: a
popular place is not more likely to grow than an unpopular one. But the steps no
longer shrink, so the shares random-walk for ever, like gene frequencies under
neutral genetic drift (a Wright–Fisher population of effective size `N/ε²`).
Destinations die out one by one and the only end state is everyone at one
place. The heterogeneity `1 − Σ p²` decays by a factor `(1 − ε²/N)` a year
(`drift_prediction`), so the time scale is `N/ε²` years:

| ε | effective destinations after 100 years | still visited in year 100 | time scale N/ε² |
|---|---|---|---|
| 0.1 | 91 | 100 | 100 000 years |
| 0.3 | 55 | ~85 | 11 000 years |
| 0.5 | 30 | ~55 | 4 000 years |
| 1 | 9.5 | ~18 | 1 000 years |

(ABM, mean over 20 seeds. The prediction agrees within about 2 destinations,
the size of the seed-to-seed noise, from `ε = 0.05` to 1, and over 5000 years.)
Unlike the additive rule, year 1's luck does not decide the winners (correlation
about 0.1 on average over seeds): they emerge by drift over the whole century. Tastes homogenise much faster than popularity concentrates: at
`ε = 0.1` every tourist's own taste is down to `0.9¹⁰⁰ ≈ 3·10⁻⁵` of their
weights after 100 years, yet the popularity is still almost even.

Concentration within a human lifetime therefore needs a small population or
`ε` close to 1, i.e. people who replace most of their tastes each year with what
they saw. In a population of millions the same neutral drift takes millions of
years.

## Conformity: selection instead of drift

With `β > 1` the drift becomes selection. Near the even split
`X_d = 1/D + δ_d`, the signal is `s_d ≈ 1/D + β δ_d`, so

    δ ← (1 − ε) δ + ε β δ = [1 + ε(β − 1)] δ:

a destination's excess popularity grows by a factor `1 + ε(β − 1)` a year,
whatever `N` is, and every run ends with one destination for everyone.
`conformity_prediction` adds the yearly sampling noise to this linear growth,
`δ ← g δ + ε β η` with `g = 1 + ε(β − 1)` and `Var η = (D − 1)/(D² N)`. It
matches the ABM within 2 % while popularity is near even (effective number
above ~50); later the leaders' advantage compounds and the ABM concentrates
faster than the linear prediction.

- **Speed.** The years until fewer than 2 effective destinations remain fall
  close to one curve against `ε(β − 1)`, roughly `≈ 3.5 / (ε(β − 1))`: about 40
  years at `ε = 0.3, β = 1.3`, 100 at `ε = 0.3, β = 1.1`, 15 at `ε = 1, β = 1.3`
  (10 seeds each). The linear-noise prediction gets them within ~20 %.
- **Population size barely matters.** `N` only sets the size of the first
  fluctuations, so ten times more tourists delay concentration by
  `ln 10 / (2 ln g)` ≈ 13 years at `ε = 0.3, β = 1.3`. The ABM gives 11 ± 1:
  fewer than 2 effective destinations remain after 35.8 ± 0.7 years for
  `N = 1000` and 46.9 ± 0.9 for `N = 10 000` (mean ± s.e. over 100 seeds).
  Without conformity the same tenfold population slows the drift tenfold
  (effective number after 100 years 54 vs 92).
- **No lock-in.** The winner is not year 1's leader (correlation ~0.1): the
  first decades of noise, while the advantage is still small, decide it.
  Runners-up can grow to several hundred visitors a year before collapsing.

The price is a behavioural assumption the other rules do not make: people weigh
popular destinations more than their share of visitors. Any `β > 1` ends in a
monopoly unless something pushes back, such as tastes that never fully fade or
congestion.

## Figures

- `plots/tourist_destinations.png` (`scripts/tourist_destinations.jl`): the
  additive rule at ε = 0, 0.01 and 1 from the same seed. Top row: visitors to each
  destination each year, ranked by year-1 visitors. Bottom row: each
  destination's popularity over time.
- `plots/tourist_destinations_mixing.png` (same script): the mixing rule at
  ε = 0.1, 0.3 and 1, each panel ranked by its own final popularity, on log
  scales.
- `plots/tourist_destinations_conformity.png` (same script): conformity at
  ε = 0.3 with β = 1 (the mixing rule), 1.1 and 1.3, drawn like the mixing figure.
- `plots/tourist_conformity_garden.svg` (`scripts/tourist_garden_figure.jl`):
  for the garden note "Why did we choose Taiwan?". Each destination's expected
  visitors per year at ε = 0.1, β = 1.3 over 200 years, with the winner and 4
  runners-up highlighted, in the garden's colours. It follows the reader's light
  or dark theme, and stays crisp at any size.
- `plots/tourist_conformity_sweep.png` (`scripts/tourist_conformity_sweep.jl`):
  the years until the tourists concentrate against `ε(β − 1)` for three ε, and
  1000 against 10 000 tourists with and without conformity, against the
  linear-noise prediction.
- `plots/tourist_influence_sweep.png` (`scripts/tourist_influence_sweep.jl`):
  the additive rule's effective number of destinations and year-1 lock-in after
  100 years, across ε, against the Pólya-urn prediction.
- `plots/tourist_mixing_sweep.png` (`scripts/tourist_mixing_sweep.jl`): the
  mixing rule's effective number of destinations after 100 years across ε, and
  over 5000 years against the additive rule, each against its mean-field
  prediction.
