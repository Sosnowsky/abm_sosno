# Mushroom Hunt — an Area-Restricted Search ABM

An [Agents.jl](https://juliadynamics.github.io/Agents.jl/stable/) agent-based model
exploring **area-restricted search** (a.k.a. *intermittent* or *two-mode* search):
the foraging strategy where an animal searches locally after finding food and
travels in straight lines when it hasn't found any for a while.

![Mushroom hunt simulation](mushroom_hunt.gif)

## The model

**Foragers** (triangles) move at constant speed through a 2D periodic space,
eating **mushrooms** (brown circles) that come within their sight radius.
Mushrooms are spawned in a few dense **clusters**, so food is patchy rather than
uniform. Each forager switches between two movement modes depending on how long
it has gone without eating (`time_since_last_found` vs. its `time_strategy`
threshold):

- **Foraging mode** (recently fed): large random turns (±π/2) — stay and sweep
  the local patch.
- **Searching mode** (hungry a while): small turns (±π/16) — travel nearly
  straight to reach a new area.

A forager's `time_strategy` sets how long it lingers after a find before giving
up and relocating. The experiment pits a **fast** forager (`time_strategy = 0`,
never lingers) against a **slow** one (`time_strategy = 50`, does
area-restricted search) competing for the same mushrooms.

## Experiment

Across many random realizations we record each forager's **share** of the
mushrooms eaten, and histogram the slow forager's fraction. A value of 0.5 means
the two strategies tie; above 0.5 means area-restricted search wins.

![Distribution of the slow forager's share](fraction_slow_hist.png)

Whether area-restricted search pays off depends on the environment: it helps most
when patches are **rich enough to chain finds** (keeping the forager in foraging
mode), and provides little benefit when patches are sparse and tiny, where both
foragers behave like straight-line searchers most of the time.

## Butterfly corridors — a second model

A separate model (Pe'er et al. 2005) explores **virtual corridors**: narrow paths
that many butterflies follow even though nothing about the habitat there is
special. Each butterfly moves over an elevation landscape, stepping to its highest
neighbouring patch with probability `q` (hilltopping) and to a random neighbour
otherwise. The corridor is *emergent* — it appears in the cumulative visitation
map (right) purely from the interplay of this simple rule with the topography.
See [`butterfly_corridor.md`](butterfly_corridor.md) for the full model summary.

![Butterfly corridor simulation](butterfly_corridor.gif)

How sharp is the corridor? Releasing all butterflies from a single patch and
measuring the width of the trail they leave — either as `area / radial reach` or
as the transverse spread of the visited patches (PCA minor axis) — shows the
corridor tightening monotonically as the uphill bias `q` grows: a diffuse, near-
random cloud at low `q` collapses to a single deterministic track at `q = 1`.

![Corridor width vs. uphill bias q](butterfly_corridor_width_spatial.png)

## Layout

- [`src/MushroomHunt.jl`](src/MushroomHunt.jl) — the model: agent types, the
  stepping rule, and `initialize` (plotting-agnostic, all parameters are keyword
  arguments so it is reusable for scans and ensembles).
- [`scripts/single_run.jl`](scripts/single_run.jl) — one realization rendered to
  a video (space + live mushrooms-eaten panel).
- [`scripts/single_run_result.jl`](scripts/single_run_result.jl) — ensemble over
  many seeds, producing the histogram above.
- [`src/ButterflyCorridor.jl`](src/ButterflyCorridor.jl) — the butterfly-corridor
  model: the `Butterfly` agent, the hilltopping/random step rule, and `initialize`.
- [`scripts/butterfly_single_run.jl`](scripts/butterfly_single_run.jl) — one
  realization rendered to the corridor video/gif above.
- [`scripts/butterfly_q_sweep.jl`](scripts/butterfly_q_sweep.jl) — sweeps `q` and
  plots the spatial corridor-width estimators above (no single-run visualization).

## Running

```julia
using Pkg; Pkg.activate(".")   # from the repo root
include("scripts/single_run.jl")         # generates mushroom_hunt.mp4
include("scripts/single_run_result.jl")  # generates fraction_slow_hist.png
```

## References

- S. Benhamou, *Efficiency of area-concentrated searching behaviour in a
  continuous patchy environment*, J. Theor. Biol. **159**, 67–81 (1992).
- O. Bénichou, C. Loverdo, M. Moreau, R. Voituriez, *Intermittent search
  strategies*, Rev. Mod. Phys. **83**, 81 (2011).
