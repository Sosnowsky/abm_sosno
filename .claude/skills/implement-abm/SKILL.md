---
name: implement-abm
description: Implement an ABM model from its description in models/ as three files — a plotting-agnostic src module, a single-run visualization script, and an ensemble script.
disable-model-invocation: true
---

Turn a model description in `models/` into a working Agents.jl model, following
the conventions already established by this repo's other models. You are invoked
with the model's name (e.g. `/implement-abm business investor`).

Work through the steps in order. The **clarify** step is a gate: no `.jl` file is
written until it is satisfied.

## 1. Read the description

Find the file in `models/` matching the name you were invoked with and read it in
full. Also read `models/butterfly_corridor.md` once for the level of detail a
complete description carries.

**Done when** you can state, for this model: its purpose; each entity and its
state variables; the space type; the stepping rule(s); which quantities are
tunable parameters; and what an ensemble experiment over it would measure — or
you have listed exactly which of these the description leaves open.

## 2. Clarify (gate)

For every point from step 1 that is underdetermined for implementation, ask the
user with `AskUserQuestion`. Typical gaps: `GridSpace` vs `ContinuousSpace` and
its dimensions; each agent type's state variables; the precise step rule and
scheduling; which parameters are tunable vs fixed; and the metric the ensemble
script should compute and plot.

**Done when** no remaining ambiguity would change the code. Write no `.jl` file
before this holds.

## 3. Study the template

Read the existing model closest in space type and treat it as the source of truth
for structure, naming, and comment density:

- **Grid / `GridSpace`** → `src/ButterflyCorridor.jl`, `scripts/butterfly_single_run.jl`, `scripts/butterfly_q_sweep.jl`
- **Continuous / `ContinuousSpace`** → `src/MushroomHunt.jl`, `scripts/mushroom_single_run.jl`, `scripts/mushroom_ensemble.jl`

## 4. Write `src/<ModelName>.jl`

A module that is **realization- and visualization-agnostic**: it imports no
plotting package (no Makie/CairoMakie) and hard-codes no single run. It contains:

- A module docstring naming the source `models/<file>.md`.
- The `@agent` struct(s).
- The step function(s); a no-op method for any passive agent type the scheduler
  still visits.
- `initialize(; kwargs...) -> StandardABM` where **every tunable parameter is a
  keyword argument**, and the parameters the step functions read at run time live
  in the model's `properties`. Seed the RNG with `MersenneTwister(seed)` and set a
  scheduler.
- Choose the `properties` container by what the run does to it: a `NamedTuple` for
  values that stay fixed or are mutable *containers* mutated in place (arrays,
  like the templates' grids). If any scalar global state is *reassigned* during
  the run (e.g. `model.price = ...` each step), a `NamedTuple` cannot rebind it —
  use a `Dict{Symbol,Any}` or a `mutable struct`, accessed the same way via
  `model.key`. This case needs a `model_step!` too (see `scripts/predator_prey.jl`).
- `export` for the agent type(s), `initialize`, and the step function(s).

Guard the `!=` lexer trap: write `agent_step! = forager_step!` with spaces around
`=` (see CLAUDE.md).

## 5. Write `scripts/<name>_single_run.jl`

One realization, visualized. Follow the template: `Pkg.activate(joinpath(@__DIR__,
".."))`, `includet` the src file, `using .<Module>, Agents, CairoMakie`. Set a
`model_params` NamedTuple, build the model with `initialize(; model_params...)`,
render the run (`abmplot` or a custom figure driven by `record`) to an mp4/gif,
and `println` the output path.

## 6. Write `scripts/<name>_ensemble.jl`

Many realizations over varied seeds, summarized in a plot. Build a vector of
models with differing seeds, run them with `ensemblerun!` (or a manual loop),
aggregate the metric fixed in step 2, plot it with `Figure`/`Axis`, `save` a
`.png`, and `println` the output path.

## 7. Verify

Load the module and step it a few times to confirm it runs, e.g.
`julia --project=. -e 'include("src/<ModelName>.jl"); using .<Module>, Agents; m = initialize(); step!(m, 10)'`.
Run the ensemble script if it is quick enough. Report the outcome honestly,
including the exact error if anything fails.
