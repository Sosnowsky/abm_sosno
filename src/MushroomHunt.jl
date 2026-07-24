"""
    MushroomHunt

Model logic for the mushroom-foraging ABM: agent types, the stepping rule, and
`initialize` to build a fresh model. This module is plotting-agnostic (no Makie
dependency) so it can be reused by single-run, ensemble, and parameter-scan
scripts alike.

All tunable parameters are `initialize` keyword arguments; the ones the step
function needs at run time are stored in the model's `properties`, so parameter
scans work by simply calling `initialize(; some_param = value)`.
"""
module MushroomHunt

using Agents, Random, Distributions
using StaticArrays, LinearAlgebra

export Forager, Mushroom, initialize, forager_step!

const T = Float64

"""Rotate a 2D vector by angle `θ` (radians)."""
function rotate(v::SVector{2}, θ)
    s, c = sincos(θ)
    return SMatrix{2,2}(c, s, -s, c) * v
end

@agent struct Forager(ContinuousAgent{2,T})
    time_since_last_found::Int
    eaten_mushrooms::Int
    time_strategy::Int
end

# Even though the mushrooms do not have any behavior, we can still make them
# agents to take advantage of the space and properties of the model.
@agent struct Mushroom(ContinuousAgent{2,T}) end

function forager_step!(forager::Forager, model::StandardABM)
    # First check if there is food nearby. nearby_ids returns a lazy iterator,
    # so we loop and grab the first mushroom rather than indexing into it.
    mushroom_id = nothing
    for id in nearby_ids(forager, model, model.sight_radius)
        if model[id] isa Mushroom
            mushroom_id = id
            break
        end
    end
    if !isnothing(mushroom_id)
        remove_agent!(model[mushroom_id]::Mushroom, model) # remove the mushroom
        forager.time_since_last_found = 0 # reset the counter
        forager.eaten_mushrooms += 1
    else
        forager.time_since_last_found += 1
    end

    # Update the velocity: big turns to stay in a patch while recently fed,
    # small turns (near-straight travel) once we've gone hungry a while.
    if forager.time_since_last_found < forager.time_strategy
        # Foraging for food
        θ = rand(
            abmrng(model),
            Uniform(-model.max_rotation_foraging, model.max_rotation_foraging),
        )
        forager.vel = rotate(forager.vel, θ)
    else
        # Searching for food
        θ = rand(
            abmrng(model),
            Uniform(-model.max_rotation_searching, model.max_rotation_searching),
        )
        forager.vel = rotate(forager.vel, θ)
    end

    move_agent!(forager, model, model.dt)
end

# Mushrooms don't act, but the scheduler still calls the step function on every
# agent, so they need a method too (a no-op).
forager_step!(::Mushroom, model::StandardABM) = nothing

"""
    initialize(; kwargs...) -> StandardABM

Build a fresh model with two foragers (the `time_strategies`, default `(0, 50)`)
and `n_clusters` patches of `n_mushrooms_per_cluster` mushrooms each.
"""
function initialize(;
    seed = 1234,
    dims = (64, 64),
    speed = 1,
    dt = 1,
    sight_radius = 1,
    max_rotation_searching = π / 16,
    max_rotation_foraging = π / 2,
    n_clusters = 4,
    n_mushrooms_per_cluster = 20,
    time_strategies = (0, 50),
    cluster_spread = 2,
)
    properties = (; speed, dt, sight_radius, max_rotation_searching, max_rotation_foraging)
    space = ContinuousSpace(dims, periodic = true)
    model = StandardABM(
        Union{Forager,Mushroom},
        space;
        (agent_step!) = forager_step!,
        properties,
        rng = MersenneTwister(seed),
        scheduler = Schedulers.Randomly(),
        warn = false,
    )

    for ts in time_strategies
        θ = 2π * rand(abmrng(model))
        vel = speed * SVector(cos(θ), sin(θ))
        # Start "hungry" (high time_since_last_found) so a forager begins in
        # searching mode until it finds its first mushroom. Any value above the
        # largest time_strategy works; kept finite to avoid integer overflow as
        # the counter keeps incrementing when no food is found.
        add_agent!(Forager, model, vel, 100, 0, ts)
    end

    for _ = 1:n_clusters
        center = random_position(model)
        for _ = 1:n_mushrooms_per_cluster
            offset = rand(abmrng(model), Uniform(-cluster_spread, cluster_spread), 2)
            pos = mod.(center + offset, dims)
            add_agent!(pos, Mushroom, model, SVector(zero(T), zero(T)))
        end
    end

    return model
end

end # module
