using Agents
using Random: Xoshiro, shuffle!
using CairoMakie

extent = (10, 10)
space = GridSpaceSingle(extent; periodic = false, metric = :chebyshev)

@agent struct SchellingAgent(GridAgent{2})
    mood::Bool # whether the agent is happy in its position
    group::Int # The group of the agent, determines mood as it interacts with neighbors
end

function agent_step!(agent::SchellingAgent, model)
    neighbors = nearby_agents(agent, model)
    same_group_neighbors = count(n -> n.group == agent.group, neighbors)

    if same_group_neighbors >= model.min_to_be_happy
        agent.mood = true
    else
        agent.mood = false
        move_agent_single!(agent, model)
    end
    return
end

properties = Dict(:min_to_be_happy => 3)
schelling = StandardABM(
    # input arguments
    SchellingAgent, space;
    # keyword arguments
    properties, # in Julia if the input variable and keyword are named the same,
    # you don't need to repeat the keyword!
    agent_step!
)

function initialize(; total_agents = 320, gridsize = (20, 20), min_to_be_happy = 3, seed = 42)
    space = GridSpaceSingle(gridsize; periodic = false)
    properties = Dict(:min_to_be_happy => min_to_be_happy)
    rng = Xoshiro(seed)
    model = StandardABM(
        SchellingAgent, space;
        agent_step! = agent_step!, properties, rng,
        container = Vector, # agents are not removed, so we us this
        scheduler = Schedulers.Randomly() # all agents are activated once at random
    )
    # populate the model with agents, adding equal amount of the two types of agents
    # at random positions in the model. At the start all agents are unhappy.
    groups = shuffle!(vcat(fill(1, total_agents ÷ 2), fill(2, total_agents ÷ 2)))
    for n in 1:total_agents
        add_agent_single!(model; mood = false, group = groups[n])
    end
    return model
end

schelling = initialize()

happy90(model, time) = count(a -> a.mood == true, allagents(model)) / nagents(model) ≥ 0.9

step!(schelling, happy90)
abmtime(schelling)

groupcolor(a) = a.group == 1 ? :blue : :orange
groupmarker(a) = a.group == 1 ? :circle : :rect

schelling = initialize(total_agents = 1200, gridsize = (40, 40))
abmvideo(
    schelling, "schelling.mp4";
    agent_color = groupcolor, agent_marker = groupmarker, agent_size = 10,
    framerate = 4, frames = 25,
    title = "Schelling's segregation model"
)

figure, _ = abmplot(schelling; agent_color = groupcolor, agent_marker = groupmarker, agent_size = 10)
figure