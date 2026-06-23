using Agents, Random, CairoMakie, Distributions
using StaticArrays, LinearAlgebra
const T = Float64

function rotate(v::SVector{2}, θ)
    s, c = sincos(θ)
    return SMatrix{2,2}(c, s, -s, c) * v
end

@agent struct Forager(ContinuousAgent{2, T})
    time_since_last_found::Int
    eaten_mushrooms::Int
    time_strategy::Int
end

# Even though the mushrooms to not have any behavior, we can still make them agents to take advantage of the space and properties of the model.
@agent struct Mushroom(ContinuousAgent{2, T})
end

dims = (200, 200)
speed = 1
dt = 1
sight_radius = 1
max_rotation_searching = π / 16
max_rotation_foraging = π / 2
space = ContinuousSpace(dims, periodic = true)
n_clusters = 16
n_mushrooms_per_cluster = 20

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
        remove_agent!(model[mushroom_id]::Mushroom, model) # remove the mushroom from the model
        forager.time_since_last_found = 0 # reset the counter
        forager.eaten_mushrooms += 1
    else
        forager.time_since_last_found += 1
    end

    # Update the velocity
    if forager.time_since_last_found < forager.time_strategy
        # Foraging for food
        θ = rand(abmrng(model), Uniform(-max_rotation_foraging, max_rotation_foraging))
        forager.vel = rotate(forager.vel, θ)
    else
        # Searching for food
        θ = rand(abmrng(model), Uniform(-max_rotation_searching, max_rotation_searching))
        forager.vel = rotate(forager.vel, θ)
    end

    move_agent!(forager, model, model.dt)
end

# Mushrooms don't act, but the scheduler still calls the step function on every
# agent, so they need a method too (a no-op).
forager_step!(::Mushroom, model::StandardABM) = nothing

properties = (
    speed = speed,
    sight_radius = sight_radius,
    dt = dt,
)
seed = 1234
rng = MersenneTwister(seed)

model = StandardABM(
    Union{Forager, Mushroom}, space;
    agent_step! = forager_step!, properties, rng, scheduler = Schedulers.Randomly(), warn = false
)

time_since_last_found = 100
θ_init = 2π * rand(abmrng(model))
vel = speed * SVector(cos(θ_init), sin(θ_init))
add_agent!(Forager, model, vel, time_since_last_found, 0, 0)

θ_init = 2π * rand(abmrng(model))
vel = speed * SVector(cos(θ_init), sin(θ_init))
add_agent!(Forager, model, vel, time_since_last_found, 0, 50)

for _ in 1:n_clusters
    center = random_position(model)
    for _ in 1:n_mushrooms_per_cluster
        offset = rand(abmrng(model), Uniform(-2, 2), 2)
        pos = mod.(center + offset, dims)
        add_agent!(pos, Mushroom, model, SVector(zero(T), zero(T)))
    end
end


ashape(a) = a isa Forager ? :utriangle : :circle
foragercol(a) = a.id == 1 ? :yellow : :blue
acolor(a) = a isa Forager ? foragercol(a) : :brown

plotkwargs = (;
    agent_color = acolor,
    agent_size = 25,
    agent_marker = ashape,
    agentsplotkwargs = (strokewidth = 1.0, strokecolor = :black),
)

eaten(a) = a.eaten_mushrooms
fast(a) = a isa Forager && a.time_strategy == 0
slow(a) = a isa Forager && a.time_strategy == 50
adata = [(eaten, sum, fast), (eaten, sum, slow)]

adf, mdf = run!(initialize(;seed=4111), 2000; adata)

# --- Video: ABM space (left) + mushrooms-eaten-by-strategy panel (right) ---
# `abmvideo` only renders the space, so we build the figure ourselves with
# `abmplot(...; adata)` (which keeps `abmobs.adf` updated each step) and drive
# it with Makie's `record`. A *fresh* model is built so the video starts at
# t = 0 (the `model` above was already stepped 40 times by `run!`).
function initialize(; seed = 1234)
    s = ContinuousSpace(dims, periodic = true)
    m = StandardABM(Union{Forager, Mushroom}, s;
        agent_step! = forager_step!, properties,
        rng = MersenneTwister(seed), scheduler = Schedulers.Randomly(), warn = false)
    for ts in (0, 50)
        θ = 2π * rand(abmrng(m))
        add_agent!(Forager, m, speed * SVector(cos(θ), sin(θ)), 100, 0, ts)
    end
    for _ in 1:n_clusters
        center = random_position(m)
        for _ in 1:n_mushrooms_per_cluster
            pos = mod.(center + rand(abmrng(m), Uniform(-2, 2), 2), dims)
            add_agent!(pos, Mushroom, m, SVector(zero(T), zero(T)))
        end
    end
    return m
end


vmodel = initialize()
vfig, vax, vabmobs = abmplot(vmodel;
    adata, add_controls = false, figure = (size = (1100, 560),), plotkwargs...)

vision = lift(vabmobs.model) do m
    [Circle(Point2f(a.pos), m.sight_radius) for a in allagents(m) if a isa Forager]
end
poly!(vax, vision; color = (:yellow, 0.1), strokecolor = :blue, strokewidth = 1)

# Right-hand data panel tracking cumulative mushrooms eaten, per strategy.
# Column names come from `dataname`: "sum_eaten_fast" / "sum_eaten_slow".
data_ax = Axis(vfig[1, 2]; xlabel = "time", ylabel = "mushrooms eaten")
fast_curve = @lift(Point2f.($(vabmobs.adf).time, $(vabmobs.adf)[:, "sum_eaten_fast"]))
slow_curve = @lift(Point2f.($(vabmobs.adf).time, $(vabmobs.adf)[:, "sum_eaten_slow"]))
scatterlines!(data_ax, fast_curve; color = :gold, label = "fast (time_strategy=0)")
scatterlines!(data_ax, slow_curve; color = :blue, label = "slow (time_strategy=50)")
axislegend(data_ax; position = :lt)

frames = 2000
record(vfig, "mushroom_hunt.mp4"; framerate = 50) do io
    for _ in 1:frames
        recordframe!(io)
        Agents.step!(vabmobs, 1)
        autolimits!(data_ax)   # keep the growing curves in view
    end
    recordframe!(io)
end