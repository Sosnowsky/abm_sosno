# Tests for the tourist-destination model.
#
# The first sets pin down the rules as written: the weighted die, the weight
# update with the tourist's own trip left out, and ε = 0 as the no-influence
# limit. The last checks the whole model against a known answer: with linear
# reinforcement it must behave as a Pólya urn, so the spread of popularity and
# the share of it decided in year 1 must match `polya_prediction`.
#
# Run with:  julia --project=. test/test_tourist_destinations.jl

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "TouristDestinations.jl"))
using Agents, Test, Statistics, Random
using .TouristDestinations

const TD = TouristDestinations

@testset "Tourist-destination model" begin

    @testset "the weighted die" begin
        rng = MersenneTwister(1)
        w = [1.0, 0.0, 3.0, 6.0]
        n = 200_000
        counts = zeros(Int, 4)
        for _ = 1:n
            counts[TD.weighted_choice(rng, w)] += 1
        end
        @test counts[2] == 0                       # a zero weight is never drawn
        @test counts ./ n ≈ w ./ sum(w) atol = 0.005
    end

    @testset "construction" begin
        m = initialize(; n_tourists = 50, n_destinations = 7, seed = 3)
        @test nagents(m) == 50
        @test all(t -> length(t.weights) == 7 && all(0 .<= t.weights .<= 1), allagents(m))
        @test sum(popularity(m)) ≈ 1
        @test_throws ArgumentError initialize(; ε = -1)
        @test_throws ArgumentError initialize(; n_destinations = 3, initial_weights = rng -> zeros(3))
        @test_throws ArgumentError initialize(; n_destinations = 3, initial_weights = rng -> ones(4))
    end

    @testset "one year: everyone travels, then weights grow by ε × others' trips" begin
        ε = 0.5
        m = initialize(; n_tourists = 200, n_destinations = 10, ε, seed = 4)
        before = Dict(t.id => copy(t.weights) for t in allagents(m))
        step!(m, 1)
        v = m.visits
        @test sum(v) == 200
        @test v == [count(t -> t.destination == d, allagents(m)) for d = 1:10]
        for t in allagents(m)
            others = v .- [d == t.destination for d = 1:10]   # n′ ≠ n
            @test t.weights ≈ before[t.id] .+ ε .* others
            # Each weight grows by ε × (N − 1) in total, and none falls.
            @test sum(t.weights) - sum(before[t.id]) ≈ ε * 199
            @test all(t.weights .>= before[t.id])
        end
    end

    @testset "ε = 0 keeps the initial tastes" begin
        m = initialize(; ε = 0, seed = 5)
        w0 = Dict(t.id => copy(t.weights) for t in allagents(m))
        h = history(m, 20)
        @test all(t -> t.weights == w0[t.id], allagents(m))
        @test all(c -> c ≈ h.popularity[:, 1], eachcol(h.popularity))
        @test all(==(1000), sum(h.visits, dims = 1))
        # Each year is a fresh draw from the same tastes: no memory of year 1.
        @test abs(cor(h.visits[:, 1], h.visits[:, 20])) < 0.3
    end

    @testset "reproducible from the seed" begin
        a = history(initialize(; ε = 0.1, seed = 6), 10)
        b = history(initialize(; ε = 0.1, seed = 6), 10)
        @test a.visits == b.visits
        # Same seed, different ε: same tastes, so the same first year.
        c = history(initialize(; ε = 2.0, seed = 6), 10)
        @test c.visits[:, 1] == a.visits[:, 1]
        @test c.visits[:, 2] != a.visits[:, 2]
    end

    @testset "behaves as a Pólya urn" begin
        @test polya_prediction(0, 100) == (effective_number = 100.0, lock_in = 0.0)
        @test polya_prediction(1.0, 0).effective_number == 100
        # The strong-influence limit, D / (1 + (D − 1) Σ 1/t² / N).
        limit = 100 / (1 + 99 * sum(1 / t^2 for t = 1:100) / 1000)
        @test polya_prediction(1e6, 100).effective_number ≈ limit rtol = 1e-3
        @test polya_prediction(1e6, 100).lock_in ≈ 1 / sqrt(sum(1 / t^2 for t = 1:100)) rtol = 1e-3

        # The model against the prediction, averaged over seeds.
        for ε in (0.01, 0.1, 1.0)
            runs = map(1:12) do s
                m = initialize(; ε, seed = 100 + s)
                h = history(m, 100)
                p = popularity(m)
                (effective_number(p), cor(h.visits[:, 1], p))
            end
            pred = polya_prediction(ε, 100)
            @test mean(first.(runs)) ≈ pred.effective_number atol = 1.5
            @test mean(last.(runs)) ≈ pred.lock_in atol = 0.08
        end
    end

    @testset "mixing rule: w ← (1 − ε) w + (ε/N) Σ d stays a distribution" begin
        @test_throws ArgumentError initialize(; update = :mixing, ε = 1.5)
        @test_throws ArgumentError initialize(; update = :copying)

        ε = 0.3
        m = initialize(; n_tourists = 200, n_destinations = 10, ε, update = :mixing, seed = 7)
        @test all(t -> sum(t.weights) ≈ 1, allagents(m))
        before = Dict(t.id => copy(t.weights) for t in allagents(m))
        step!(m, 1)
        for t in allagents(m)
            @test t.weights ≈ (1 - ε) .* before[t.id] .+ ε .* m.visits ./ 200
        end
        step!(m, 50)
        @test all(t -> sum(t.weights) ≈ 1 && all(>=(0), t.weights), allagents(m))

        # ε = 1: everyone forgets their taste and copies last year's crowd.
        m = initialize(; n_tourists = 200, n_destinations = 10, ε = 1, update = :mixing, seed = 8)
        step!(m, 1)
        @test all(t -> t.weights ≈ m.visits ./ 200, allagents(m))
        @test popularity(m) ≈ m.visits ./ 200
    end

    @testset "mixing rule behaves as neutral drift" begin
        @test drift_prediction(0, 100) ≈ 100
        @test drift_prediction(1, 10^6) ≈ 1 atol = 1e-6
        # The model against the prediction for 1 / E[Σ p²], averaged over seeds.
        for (ε, rtol) in ((0.5, 0.1), (1.0, 0.2))
            s2 = map(1:12) do s
                m = initialize(; ε, update = :mixing, seed = 200 + s)
                history(m, 100)
                sum(abs2, popularity(m))
            end
            @test 1 / mean(s2) ≈ drift_prediction(ε, 100) rtol = rtol
        end
    end

    @testset "conformity: s_d ∝ c_d^β" begin
        @test_throws ArgumentError initialize(; update = :mixing, β = 0)
        @test_throws ArgumentError initialize(; update = :additive, β = 1.5)

        # β = 1 is the mixing rule, draw for draw.
        a = history(initialize(; ε = 0.3, update = :mixing, seed = 9), 20)
        b = history(initialize(; ε = 0.3, update = :mixing, β = 1, seed = 9), 20)
        @test a.visits == b.visits

        # One year: w ← (1 − ε) w + ε c^β / Σ c^β, still a distribution.
        ε, β = 0.3, 1.5
        m = initialize(; n_tourists = 200, n_destinations = 10, ε, β, update = :mixing, seed = 10)
        before = Dict(t.id => copy(t.weights) for t in allagents(m))
        step!(m, 1)
        c = m.visits
        for t in allagents(m)
            @test t.weights ≈ (1 - ε) .* before[t.id] .+ ε .* c .^ β ./ sum(c .^ β)
            @test sum(t.weights) ≈ 1
        end
    end

    @testset "conformity: early growth at rate 1 + ε(β − 1)" begin
        # At β = 1 the linear-noise prediction is the first-order drift prediction.
        @test conformity_prediction(0.3, 1, 100) ≈ drift_prediction(0.3, 100) rtol = 0.01
        # The model against it while popularity is still near even (N_eff ≳ 50).
        for (ε, β, T) in ((0.3, 1.3, 10), (0.3, 1.1, 30), (0.1, 1.3, 40))
            s2 = map(1:12) do s
                m = initialize(; ε, β, update = :mixing, seed = 300 + s)
                history(m, T)
                sum(abs2, popularity(m))
            end
            @test 1 / mean(s2) ≈ conformity_prediction(ε, β, T) rtol = 0.05
        end
        # Any β > 1 ends with one destination for everyone.
        m = initialize(; ε = 0.3, β = 1.5, update = :mixing, seed = 11)
        history(m, 100)
        @test maximum(popularity(m)) > 0.99
    end
end
