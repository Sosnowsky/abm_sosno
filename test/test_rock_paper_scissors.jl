# Tests for the rock–paper–scissors learning model.
#
# The two that matter most are the *mechanism* tests: a learner facing a frozen
# pure opponent must converge on the counter to it (that exercises sampling,
# belief update, best response and the strategy step end to end), and the
# strategy must stay on the simplex forever without anything renormalising it —
# which is only true because every update is a convex combination.
#
# Run with:  julia --project=. test/test_rock_paper_scissors.jl

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "RockPaperScissors.jl"))
using Agents, Test, Random, Statistics, StaticArrays, LinearAlgebra
using .RockPaperScissors

const RPS = RockPaperScissors

# Cyclic neighbours in the beats-order R → P → S → R.
prev_action(i) = mod1(i - 1, 3)
next_action(i) = mod1(i + 1, 3)

@testset "Rock–paper–scissors learning model" begin

    @testset "the game" begin
        # Zero-sum: swapping the roles negates the payoff.
        @test PAYOFF' == -PAYOFF
        @test all(PAYOFF[i, i] == 0 for i = 1:3)
        # R beats S, P beats R, S beats P.
        @test PAYOFF[1, 3] == 1
        @test PAYOFF[2, 1] == 1
        @test PAYOFF[3, 2] == 1

        # Every action is worth exactly 0 against the Nash mix — which is why
        # there is no gradient pulling learners back to the centre.
        @test all(isapprox.(expected_payoffs(NASH), 0; atol = 1e-15))

        # The closed form quoted in the spec.
        q = SVector(0.5, 0.3, 0.2)
        @test expected_payoffs(q) ≈ SVector(q[3] - q[2], q[1] - q[3], q[2] - q[1])
        @test expected_payoffs(q) ≈ PAYOFF * q
    end

    @testset "best response" begin
        rng = MersenneTwister(1)

        # Against a pure opponent, play the counter.
        @test best_response(VERTEX[1], rng) == 2   # vs rock → paper
        @test best_response(VERTEX[2], rng) == 3   # vs paper → scissors
        @test best_response(VERTEX[3], rng) == 1   # vs scissors → rock

        # The worked counter-example from the spec: the reply to the *mode* (S)
        # would be rock, but the actual best reply is scissors.
        @test best_response(SVector(0.2, 0.35, 0.45), rng) == 3

        # The wedge characterisation: BR = i exactly when q[prev(i)] ≥ ⅓ and
        # q[next(i)] ≤ ⅓. Checked on random beliefs.
        for _ = 1:2000
            q = RPS.random_strategy(rng)
            i = best_response(q, rng)
            @test q[prev_action(i)] ≥ 1 / 3 - 1e-12
            @test q[next_action(i)] ≤ 1 / 3 + 1e-12
        end

        # At Nash all three payoffs tie exactly, so the tie-break must be
        # uniform rather than always picking the first index.
        picks = [best_response(NASH, rng) for _ = 1:6000]
        @test sort(unique(picks)) == [1, 2, 3]
        @test all(1800 .< [count(==(i), picks) for i = 1:3] .< 2200)
    end

    @testset "sampling actions" begin
        rng = MersenneTwister(2)
        x = SVector(0.5, 0.2, 0.3)
        draws = [sample_action(x, rng) for _ = 1:200_000]
        freq = [count(==(i), draws) / length(draws) for i = 1:3]
        @test all(abs.(freq .- x) .< 0.005)

        # A pure strategy never draws anything else.
        @test all(sample_action(VERTEX[2], rng) == 2 for _ = 1:1000)
    end

    @testset "construction and validation" begin
        model = initialize(; seed = 1)
        @test nagents(model) == 2
        @test model.rounds_per_match == 3
        @test all(a.belief == NASH for a in allagents(model))   # beliefs start blank
        @test all(isapprox(sum(a.strategy), 1; atol = 1e-12) for a in allagents(model))

        # Explicit strategies, and per-player learning rates.
        m2 = initialize(;
            init = [SVector(1.0, 0.0, 0.0), SVector(0.0, 0.5, 0.5)],
            learning_rate = [0.0, 0.4],
            seed = 1,
        )
        @test m2[1].strategy == SVector(1.0, 0.0, 0.0)
        @test m2[2].learning_rate == 0.4

        @test_throws ArgumentError initialize(; n_players = 1)
        @test_throws ArgumentError initialize(; exploration = 1.5)
        @test_throws ArgumentError initialize(; memory = 0.0)
        @test_throws ArgumentError initialize(; init = [SVector(0.5, 0.4, 0.4)])
        @test_throws ArgumentError initialize(; init = [SVector(1.5, -0.5, 0.0)])
        @test_throws ArgumentError initialize(; learning_rate = [0.1, 0.2, 0.3])
    end

    @testset "a learner beats a frozen opponent to the counter" begin
        # Player 2 is pinned to pure rock (η = 0 means it never moves); player 1
        # must discover paper. This is the whole loop — sample, observe, believe,
        # best-respond — in one assertion.
        for (frozen, counter) in ((1, 2), (2, 3), (3, 1))
            model = initialize(;
                init = [NASH, VERTEX[frozen]],
                learning_rate = [0.3, 0.0],
                seed = 7,
            )
            Agents.step!(model, 200)
            @test model[1].strategy[counter] > 0.99
            @test model[2].strategy == VERTEX[frozen]      # really frozen
            @test model[1].score > 0                        # and it pays off
        end
    end

    @testset "strategies stay on the simplex" begin
        for (η, ε, λ, K) in ((0.1, 0.0, 1.0, 3), (1.0, 0.0, 0.2, 1), (0.7, 0.3, 0.5, 9))
            model = initialize(;
                learning_rate = η, exploration = ε, memory = λ, rounds_per_match = K,
                seed = 3,
            )
            xs = trajectory(model, 500)
            @test all(isapprox(sum(x), 1; atol = 1e-12) for x in xs)
            @test all(all(≥(0), x) for x in xs)
            @test all(all(≤(1), x) for x in xs)
            # Beliefs live on the simplex too.
            @test all(isapprox(sum(a.belief), 1; atol = 1e-12) for a in allagents(model))
        end
    end

    @testset "the two degenerate learning rates" begin
        # η = 0: nobody learns, strategies are frozen.
        model = initialize(; learning_rate = 0.0, seed = 4)
        start = [model[i].strategy for i = 1:2]
        Agents.step!(model, 100)
        @test [model[i].strategy for i = 1:2] == start

        # ε = 1: the exploration floor overwrites everything, pinning play at Nash.
        model = initialize(; exploration = 1.0, seed = 4)
        Agents.step!(model, 10)
        @test all(a.strategy ≈ NASH for a in allagents(model))
    end

    @testset "match accounting is zero-sum" begin
        model = initialize(; seed = 5)
        T = 300
        Agents.step!(model, T)
        a, b = model[1], model[2]
        @test a.score == -b.score
        @test a.wins == b.losses
        @test a.draws == b.draws
        @test a.wins + a.losses + a.draws == T
        # Per match, |net| ≤ K.
        @test abs(a.score) ≤ T * model.rounds_per_match
    end

    @testset "reproducibility" begin
        xs = trajectory(initialize(; seed = 11), 200)
        @test xs == trajectory(initialize(; seed = 11), 200)
        @test xs != trajectory(initialize(; seed = 12), 200)
    end

    @testset "ternary coordinates" begin
        @test ternary(VERTEX[1]) ≈ SVector(0.0, 0.0)
        @test ternary(VERTEX[2]) ≈ SVector(1.0, 0.0)
        @test ternary(VERTEX[3]) ≈ SVector(0.5, sqrt(3) / 2)
        # Nash sits at the centroid, equidistant from all three corners.
        c = ternary(NASH)
        @test all(≈(norm(c - ternary(VERTEX[1]))), [norm(c - ternary(v)) for v in VERTEX])
        # The affine map preserves midpoints, which is what makes it faithful.
        @test ternary(SVector(0.5, 0.5, 0.0)) ≈ (ternary(VERTEX[1]) + ternary(VERTEX[2])) / 2

        @test distance_to_nash(NASH) ≈ 0 atol = 1e-15
        @test distance_to_nash(VERTEX[1]) ≈ sqrt(2 / 3)
        @test exploitability(NASH) ≈ 0 atol = 1e-15
        @test exploitability(VERTEX[1]) ≈ 1        # a pure player loses a point a round
    end

    @testset "angles and winding" begin
        # Vertices sit 120° apart, with R → P → S counter-clockwise.
        @test rad2deg(angle_around_nash(VERTEX[1])) ≈ -150
        @test rad2deg(angle_around_nash(VERTEX[2])) ≈ -30
        @test rad2deg(angle_around_nash(VERTEX[3])) ≈ 90

        @test unwrap([0.0, 3.0, -3.0]) ≈ [0.0, 3.0, 2π - 3.0]
        @test unwrap([0.0, -3.0, 3.0]) ≈ [0.0, -3.0, 3.0 - 2π]

        # A synthetic exact orbit: mixing the three vertices with a cosine
        # profile 120° apart puts the point at ternary angle θ exactly, so
        # twelve evenly spaced θ trace one full counter-clockwise loop.
        θs = range(0, 2π; length = 13)[1:end-1]
        circle = map(θs) do θ
            w = SVector{3}(1 + cos(θ - a) for a in (-5π / 6, -π / 6, π / 2))
            w / sum(w)
        end
        @test all(isapprox(sum(x), 1; atol = 1e-12) for x in circle)
        @test angle_around_nash(circle[1]) ≈ 0 atol = 1e-12
        @test winding_rate(circle) ≈ 2π / 12 rtol = 1e-9
        @test winding_rate(reverse(circle)) ≈ -2π / 12 rtol = 1e-9
        @test winding_rate([NASH]) == 0.0
    end

    @testset "running mean and trajectory" begin
        model = initialize(; seed = 6)
        # Indexed by id, not `allagents`: agent iteration order is not
        # guaranteed, which is exactly why `trajectory` sorts the ids.
        x0 = [model[i].strategy for i = 1:2]
        xs = trajectory(model, 50)
        @test size(xs) == (51, 2)
        @test xs[1, :] == x0                      # first row is the initial state

        rm = running_mean(xs[:, 1])
        @test rm[1] == xs[1, 1]
        @test rm[end] ≈ mean(xs[:, 1])
        @test all(isapprox(sum(x), 1; atol = 1e-12) for x in rm)
    end

    @testset "predictions from the spec" begin
        T = 4000
        tail = (T ÷ 2 + 1):(T+1)

        # (2) The time average converges to Nash even though play does not.
        # (1) Play itself does not: it orbits at a radius well away from 0.
        for λ in (1.0, 0.2)
            radii = Float64[]
            avg_errors = Float64[]
            for seed = 1:8
                xs = trajectory(initialize(; memory = λ, seed = 100 + seed), T)
                for j = 1:2
                    push!(avg_errors, distance_to_nash(mean(xs[:, j])))
                    push!(radii, mean(distance_to_nash.(xs[tail, j])))
                end
            end
            @test mean(avg_errors) < 0.05          # time average ≈ Nash
            @test mean(radii) > 5 * mean(avg_errors)   # …but play is not
        end

        # (3) A larger learning rate means a larger orbit, monotonically.
        radius(η) = mean(
            mean(distance_to_nash.(trajectory(initialize(; learning_rate = η,
                memory = 0.2, seed = 200 + s), T)[tail, j])) for s = 1:4, j = 1:2
        )
        radii = radius.([0.02, 0.05, 0.1, 0.3])
        @test issorted(radii)
        # η = 1 pins strategies to the vertices exactly.
        @test radius(1.0) ≈ sqrt(2 / 3) atol = 1e-9

        # (5) Symmetric learners: neither wins in the long run. A persistent
        # edge would mean the synchronous-update invariant had been broken —
        # someone reacting to a strategy the other has not played yet. Measured
        # per round played, so the bound does not depend on the run length.
        matches, K = 2000, 3
        scores = Float64[]
        for seed = 1:40
            model = initialize(; memory = 0.2, rounds_per_match = K, seed = 300 + seed)
            Agents.step!(model, matches)
            push!(scores, model[1].score)
        end
        @test abs(mean(scores)) / (matches * K) < 0.01
    end
end
