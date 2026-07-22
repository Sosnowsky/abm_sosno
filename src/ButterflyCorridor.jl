"""
    ButterflyCorridor

Model logic for the butterfly virtual-corridor ABM of Pe'er et al. (2005), as
described in the ODD in Section 3.4 of Railsback & Grimm (2012). See
`butterfly_corridor.md` for the full model summary.

Butterflies live on a grid of patches whose only state is `elevation`. On every
step each butterfly either moves to its highest neighbouring patch ("hilltopping",
with probability `q`) or to a random neighbour (with probability `1 - q`).
Virtual corridors — narrow, heavily-used paths from the start region to a
hilltop — emerge from the interplay of this rule with the topography.

This module is plotting-agnostic (no Makie dependency) so it can be reused by
single-run and parameter-scan scripts alike. All tunable parameters are
`initialize` keyword arguments; the ones the step function needs at run time
(`q` and the elevation array) live in the model's `properties`.
"""
module ButterflyCorridor

using Agents, Random

export Butterfly, initialize, butterfly_step!, gaussian_topography

# Butterflies are characterised only by their location (the patch they are on),
# so the agent needs no state beyond the `id` and `pos` that `GridAgent` gives it.
@agent struct Butterfly(GridAgent{2}) end

"""
    gaussian_topography(dims; hills) -> Matrix{Float64}

Build a simple artificial topography as a sum of Gaussian hills over a
`dims` grid. Each hill is a `(x, y, height, width)` tuple: centre `(x, y)` in
patch coordinates, peak `height`, and Gaussian `width` (standard deviation in
patches). This is the "simple artificial topography" variant of the model; the
alternative is to import elevations from a real study site.
"""
function gaussian_topography(dims::NTuple{2,Int}; hills)
    elevation = zeros(Float64, dims)
    for i = 1:dims[1], j = 1:dims[2]
        for (hx, hy, height, width) in hills
            elevation[i, j] += height * exp(-((i - hx)^2 + (j - hy)^2) / (2 * width^2))
        end
    end
    return elevation
end

"""
    butterfly_step!(butterfly, model)

Move one butterfly a single patch. With probability `model.q` it moves uphill
(to the neighbouring patch with the highest elevation, ties broken at random);
otherwise it moves to a uniformly random neighbour. "Neighbours" are the eight
surrounding patches (Moore neighbourhood).
"""
function butterfly_step!(butterfly::Butterfly, model::StandardABM)
    rng = abmrng(model)
    neighbours = collect(nearby_positions(butterfly, model, 1))
    isempty(neighbours) && return  # nowhere to go (shouldn't happen on this grid)

    if rand(rng) < model.q
        # Move uphill: pick the highest-elevation neighbour, breaking ties randomly.
        elev = model.elevation
        best = maximum(elev[p...] for p in neighbours)
        highest = filter(p -> elev[p...] == best, neighbours)
        target = rand(rng, highest)
    else
        # Move randomly: any neighbour with equal probability.
        target = rand(rng, neighbours)
    end

    move_agent!(butterfly, target, model)
    return
end

"""
    initialize(; kwargs...) -> StandardABM

Build a fresh butterfly-corridor model.

Keyword arguments:
- `q`: probability of moving directly uphill instead of randomly (0–1, global).
- `dims`: grid size in patches (paper uses `(150, 150)`).
- `n_butterflies`: number of butterflies (paper uses `500`).
- `start`: patch (or centre of the start region) where butterflies are released.
- `start_radius`: butterflies are placed within this Chebyshev radius of `start`
  (`0` means all on a single patch).
- `hills`: `(x, y, height, width)` tuples defining the artificial topography.
- `elevation`: optional precomputed elevation matrix (overrides `hills`), e.g.
  imported from a real study site.
- `seed`: RNG seed.
"""
function initialize(;
    q = 0.4,
    dims = (150, 150),
    n_butterflies = 500,
    start = (15, 15),
    start_radius = 4,
    hills = ((110, 40, 100.0, 25.0), (60, 120, 80.0, 20.0)),
    elevation = nothing,
    seed = 1234,
)
    elev = isnothing(elevation) ? gaussian_topography(dims; hills) : elevation
    @assert size(elev) == dims "elevation matrix must have size $dims"

    properties = (; q = float(q), elevation = elev)
    # Non-periodic space: hilltopping toward fixed peaks needs real boundaries.
    space = GridSpace(dims, periodic = false)
    model = StandardABM(
        Butterfly,
        space;
        agent_step! = butterfly_step!,
        properties,
        rng = MersenneTwister(seed),
        scheduler = Schedulers.Randomly(),
    )

    # Release all butterflies at (or near) the start patch.
    for _ = 1:n_butterflies
        if start_radius <= 0
            add_agent!(start, Butterfly, model)
        else
            dx = rand(abmrng(model), (-start_radius):start_radius)
            dy = rand(abmrng(model), (-start_radius):start_radius)
            pos = (clamp(start[1] + dx, 1, dims[1]), clamp(start[2] + dy, 1, dims[2]))
            add_agent!(pos, Butterfly, model)
        end
    end

    return model
end

end # module
