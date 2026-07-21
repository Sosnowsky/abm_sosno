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

## Layout

- [`src/MushroomHunt.jl`](src/MushroomHunt.jl) — the model: agent types, the
  stepping rule, and `initialize` (plotting-agnostic, all parameters are keyword
  arguments so it is reusable for scans and ensembles).
- [`scripts/single_run.jl`](scripts/single_run.jl) — one realization rendered to
  a video (space + live mushrooms-eaten panel).
- [`scripts/single_run_result.jl`](scripts/single_run_result.jl) — ensemble over
  many seeds, producing the histogram above.

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
