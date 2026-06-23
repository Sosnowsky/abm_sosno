# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an exploratory project for learning **Agent-Based Modeling (ABM)** and **Julia** programming language. The primary tool is the **Agents.jl** package.

### Educational Goals
- **Learn Julia**: A high-performance, dynamically-typed language designed for scientific computing and numerical work
- **Learn ABM**: Simulation technique where individual agents interact according to rules, producing emergent behavior at the system level

## Julia Basics for Newcomers

### Installation & Setup

Julia can be installed from https://julialang.org/downloads/. For development, the VSCode extension for Julia is recommended.

### Running Julia Code

- **Interactive REPL**: Type `julia` in the terminal to start an interactive session
- **Run a file**: `julia script.jl`
- **Run with arguments**: `julia script.jl arg1 arg2`
- **Execute one line**: `julia -e "println(\"hello\")"`

### Key Julia Concepts for ABM Work

**Multiple Dispatch**: Julia's core feature—functions dispatch based on *all* argument types, not just the first. This is powerful for modeling:
```julia
function step!(model::MyModel)  # dispatch on MyModel type
    # update the model
end
```

**Types**: Julia is type-aware. Define agent and model types:
```julia
mutable struct Agent
    id::Int
    pos::Tuple{Int, Int}
    energy::Float64
end
```

**Broadcasting**: The `.` operator applies operations element-wise:
```julia
agent_energies = [a.energy for a in agents]
```

**Loops and comprehensions**: Common iteration patterns:
```julia
for agent in model.agents
    step!(agent, model)
end
```

### Julia Syntax Gotchas

**`!` in identifiers vs the `!=` operator** — Julia's lexer is greedy, so when a
name ending in `!` is immediately followed by `=`, the `!=` is tokenized as the
*not-equal* operator. This silently breaks keyword arguments and assignments:

```julia
StandardABM(...; agent_step!=forager_step!)   # parsed as `agent_step != forager_step!`  → syntax error
StandardABM(...; agent_step! = forager_step!) # correct — note the spaces
```

**Always put spaces around `=` when the left side ends in `!`** (e.g.
`agent_step! = `, `model_step! = `). This applies to any `!`-suffixed binding,
not just Agents.jl keywords.

### Package Management

- **Add a package**: `] add PackageName` (from REPL) or `using Pkg; Pkg.add("PackageName")`
- **Check installed packages**: `] status`
- **Create a project environment**: `] activate .` (creates Project.toml)

## Agent-Based Modeling (ABM) Fundamentals

### Core Concepts

**Agents**: Individual entities with state (position, energy, behavior rules, etc.)

**Model/Environment**: Contains agents and defines simulation rules, time steps, and spatial structure

**Emergent Behavior**: System-level patterns that arise from agent interactions (e.g., flocking, segregation)

### ABM Workflow

1. Define agent and model structures
2. Initialize agents and model state
3. Define step functions (how agents and model evolve each time step)
4. Run the simulation for N steps
5. Collect data and analyze results

### Common ABM Scenarios
- **Spatial**: Agents in 2D/3D space interacting with neighbors
- **Network**: Agents connected via relationships
- **Economic**: Agents trading, competing for resources
- **Ecological**: Predator-prey dynamics, reproduction

## Agents.jl Package

### Installation

```julia
using Pkg
Pkg.add("Agents")
```

### Key Components

**AgentBasedModel**: The main container for your simulation
```julia
model = AgentBasedModel(agent_step!, model_step!)
```

**Agent Definition**: Use `@agent` macro or define custom structs with `id` and `pos` fields

**Space Types**: 
- `GridSpace`: 2D/3D grid with discrete positions
- `ContinuousSpace`: Continuous coordinates
- `GraphSpace`: Network topology

**Stepping Functions**:
- `agent_step!(agent, model)`: What each agent does per step
- `model_step!(model)`: Global updates after all agents move

**Data Collection**: `run!()` with callbacks to collect data during simulation

### Basic Example Structure

```julia
using Agents

@agent struct MyAgent(GridAgent)
    energy::Float64
end

function agent_step!(agent, model)
    # Agent behavior here
end

function model_step!(model)
    # Global updates
end

model = AgentBasedModel(MyAgent, GridSpace((10, 10)))
run!(model, agent_step!, model_step!, 100)  # Run 100 steps
```

## Development Workflow

When starting a new ABM model:

1. **Start simple**: Define one agent type and one basic behavior
2. **Test incrementally**: Add features one at a time and verify behavior
3. **Use the REPL**: For interactive exploration during development
4. **Visualize**: Use Makie.jl or Plots.jl for animation/plotting
5. **Collect metrics**: Track quantities of interest (agent count, average energy, etc.)

### Testing & Debugging

- Use `@show` for debugging in the REPL
- Print agent states during simulation to verify logic
- Start with small model sizes (10x10 grid, 5 agents) before scaling up
- Check conservation laws (if energy should be conserved, verify it is)

## Resources

- **Julia Official Docs**: https://docs.julialang.org/
- **Agents.jl Docs**: https://juliadynamics.github.io/Agents.jl/stable/
- **Think Complexity** (free online): Good ABM conceptual introduction
- **Active Matter & Collective Behavior Papers**: For inspiration on multi-agent systems

## File Structure (to be established)

When you start developing:
- `src/`: Core model definitions and agent types
- `scripts/`: Simulation runs and analysis
- `data/`: Output from simulations
- `plots/`: Visualizations and figures

Use `/init` again after establishing your first working model to update this guide with project-specific patterns and commands.
