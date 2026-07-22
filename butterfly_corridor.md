# Butterfly Corridor Model — Summary

A summary of the **Butterfly Model** of Pe'er et al. (2005), following the ODD
(Overview, Design concepts, Details) description in Section 3.4 of Railsback &
Grimm (2012), *Agent-based and Individual-based Modeling*. This is a classic
teaching model for spatial ABM and a good target for reimplementation in
Julia / Agents.jl.

## Purpose

Explore the emergence of **virtual corridors** — relatively narrow paths along
which many butterflies move, *even though the habitat in those paths is not
otherwise special*. The model asks:

- Under what conditions do the interaction of **hilltopping behavior**
  (butterflies moving uphill to mate) and **landscape topography** produce
  virtual corridors?
- How does **variability** in the butterflies' tendency to move uphill affect
  whether corridors emerge?

The key idea is that corridors are an **emergent** property of individual
movement rules + terrain, not a designed feature of the landscape.

## Entities, State Variables, and Scales

Two kinds of entities:

| Entity | State variables |
|--------|-----------------|
| **Patch** (square cell of land) | `elevation` (only state variable) |
| **Butterfly** (agent) | location = the patch it is on (discrete x, y of patch center) |

- Landscape: a **150 × 150 grid** of square patches.
- Butterflies have **no state other than position** — no energy, age, etc.
- Positions are discrete (patch centers).
- Scale is generic. For a real landscape a patch ≈ **25 × 25 m²**.
- Simulation runs for **1000 time steps**; one step ≈ time for a butterfly to
  move 25–35 m (one cell to a neighbor cell).

## Process Overview and Scheduling

- **Only one process:** movement of butterflies.
- Each time step, **each butterfly moves once**.
- Execution order is **irrelevant** — there are **no interactions** among
  butterflies.

## Design Concepts

- **Basic principle:** virtual corridors — pathways used by many individuals
  when there is nothing intrinsically beneficial about the habitat there.
  Corridors emerge from two model parts: the adaptive movement behavior of
  butterflies and the landscape they move through.
- **Adaptive behavior:** modeled with a simple empirical rule reproducing real
  butterflies — **moving uphill (hilltopping)**. Uphill movement is assumed
  a priori to lead to mating (and thus fitness). Because the objective is
  assumed, **Objectives** and **Prediction** are not modeled explicitly, and
  there is **no learning**.
- **Sensing:** butterflies can identify which of the **surrounding patches has
  the highest elevation**, but use no elevation information at greater
  distances. (Field studies by Pe'er 2003 examined how far butterflies sense
  elevation differences.)
- **Interaction:** none. Real butterflies do interact (stopping to visit each
  other), but interaction was judged unimportant for modeling virtual
  corridors.
- **Stochasticity:** represents two hard-to-mechanize sources of movement
  variability — (1) limits in ability to sense the highest neighboring patch,
  and (2) non-topographic factors (e.g. flowers worth investigating). Modeled
  by having butterflies not always move uphill; whether an individual moves
  uphill or randomly on a given step is decided stochastically via the
  parameter **q**.
- **Observation:** to observe corridors, a **"corridor width"** measure is
  defined, characterizing the width of a butterfly's path from its starting
  patch to a hilltop.

## The Movement Submodel (core rules)

On **each time step**, for **each butterfly**:

1. Draw a random number `r` from a **uniform distribution on [0.0, 1.0]**.
2. If `r < q` → **move uphill**.
   Otherwise (`r ≥ q`) → **move randomly**.

Definitions:

- **Neighbor patches** = the **8 patches** surrounding the butterfly's current
  patch (Moore neighborhood).
- **Move uphill** = move to the neighbor patch with the **highest elevation**.
  If two neighbors tie for highest, choose one of them **at random**.
- **Move randomly** = move to one of the 8 neighbors, each with **equal
  probability**.

## Parameters

| Parameter | Meaning | Value / range |
|-----------|---------|---------------|
| `q` | Probability an individual moves **directly uphill** (vs. randomly) on a step. **Global** — same for all butterflies. | 0.0 – 1.0 |
| Grid size | Square landscape | 150 × 150 patches |
| Number of butterflies | Population | 500 |
| Time steps | Simulation length | 1000 |
| Patch size | (only for real landscapes) | 25 × 25 m² |

`q` is the central experimental knob: `q = 1` → always hilltop (deterministic
uphill); `q = 0` → pure random walk; intermediate `q` blends the two and
governs whether corridors emerge and how wide they are.

## Initialization

- **Topography** (elevation of each patch) is set at model start. Two landscape
  versions are used:
  1. a **simple artificial topography**, and
  2. the topography of a **real study site**, imported from an elevation file.
- **Butterflies:** create **500** individuals, all placed at a **single patch
  or small region** (a common starting point so paths toward hilltops can be
  compared).

## Input Data

None — the environment is assumed **constant** over the run (elevations do not
change).

## Outputs / Observations

- Spatial distribution of butterfly paths from the common start toward
  hilltop(s).
- The **corridor width** measure of the path from start patch to a hilltop.
- Emergent finding of interest: whether **narrow, heavily-used corridors** form,
  and how their existence/width responds to `q` (movement variability).

## Notes for a Julia / Agents.jl implementation

- Fits **`GridSpace((150, 150))`** with a Moore (8-cell) neighborhood.
- Patch elevation → a model-level field (e.g. a `150×150` matrix) rather than
  agent state, since patches aren't agents in Agents.jl.
- Agent (`@agent`) needs only `id` and `pos` — no extra fields required.
- `agent_step!` implements the two-branch uphill/random rule above; `q` lives in
  the model properties.
- No `model_step!` interactions are needed (agents are independent).
- Remember the Julia gotcha from CLAUDE.md: write `agent_step! = butterfly_step!`
  **with spaces** around `=`.

## References

- Pe'er, G., Saltz, D. & Frank, K. 2005. *Virtual corridors for conservation
  management.* Conservation Biology 19: 1997–2003.
- Pe'er, G. 2003. *Spatial and behavioral determinants of butterfly movement
  patterns in topographically complex landscapes.* Ph.D. thesis, Ben-Gurion
  University of the Negev.
- Railsback, S. & Grimm, V. 2012. *Agent-based and Individual-based Modeling:
  A Practical Introduction.* Princeton University Press. (Section 3.4)
