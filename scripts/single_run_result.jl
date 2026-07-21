# Single realization of the mushroom-hunt model, rendered to a video:
# ABM space (left) + cumulative-mushrooms-eaten-by-strategy panel (right).
#
# `abmvideo` only renders the space, so we build the figure ourselves with
# `abmplot(...; adata)` (which keeps `abmobs.adf` updated each step) and drive
# it with Makie's `record`.

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
includet(joinpath(@__DIR__, "..", "src", "MushroomHunt.jl"))
using .MushroomHunt
using Agents, CairoMakie, Random

function random_model()
    model_params = (
        seed = Random.rand(Int32),
        dims = (200, 200),
        speed = 1,
        dt = 1,
        sight_radius = 1,
        max_rotation_searching = π / 16,
        max_rotation_foraging = π / 2,
        n_clusters = 4,
        n_mushrooms_per_cluster = 100,
        time_strategies = (0, 50),
        cluster_spread = 2,
    )
    return initialize(; model_params...)
end

# --- Data to track live: cumulative mushrooms eaten per strategy ---
eaten(a) = a.eaten_mushrooms
fast(a) = a isa Forager && a.time_strategy == 0
slow(a) = a isa Forager && a.time_strategy == 50
adata = [(eaten, sum, fast), (eaten, sum, slow)]

all_eaten(model, s) = !any(a isa Mushroom for a in allagents(model))

N = 1000
models = [random_model() for _ = 1:N]
adf, _, _ = ensemblerun!(models, all_eaten; adata, when = all_eaten, init = false)
fraction_slow = adf.sum_eaten_slow ./ (adf.sum_eaten_slow .+ adf.sum_eaten_fast)

fraction_slow

# --- Histogram of the slow forager's share across the N realizations ---
fig = Figure()
ax = Axis(
    fig[1, 1];
    xlabel = "fraction of mushrooms eaten by slow forager",
    ylabel = "count",
    title = "N = $N realizations",
)
hist!(
    ax,
    fraction_slow;
    bins = 20,
    color = :steelblue,
    strokewidth = 0.5,
    strokecolor = :black,
)
vlines!(ax, [0.5]; color = :gray, linestyle = :dash)  # 0.5 = tie between strategies
save("fraction_slow_hist.png", fig)
fig

sum(fraction_slow)/N
