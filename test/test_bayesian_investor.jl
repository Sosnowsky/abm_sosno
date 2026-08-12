# Tests for the Bayesian business-investor model.
#
# The headline test is the `sensing_radius = 0` regression: with risk hidden but
# nobody moving, the Bayesian model must reproduce the omniscient predecessor
# *exactly*. That only holds because the belief machinery draws no random
# numbers, so it is a sharp check that no randomness leaked into it.
#
# Run with:  julia --project=. test/test_bayesian_investor.jl

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "BusinessInvestor.jl"))
include(joinpath(@__DIR__, "..", "src", "BayesianInvestor.jl"))
using Agents, Test, Statistics
using .BusinessInvestor: BusinessInvestor
using .BayesianInvestor: BayesianInvestor

const BI = BusinessInvestor
const BY = BayesianInvestor

base = (
    dims = (19, 19),
    n_investors = 10,
    T = 5,
    profit_range = (-5.0, 10.0),
    risk_range = (0.01, 0.1),
)

@testset "Bayesian investor model" begin

    @testset "prior resolution" begin
        # Moment-matched to U(0.01, 0.1): m = 0.055, v = 0.000675, s₀ = 76.
        α₀, β₀ = BY.resolve_prior(nothing, nothing, (0.01, 0.1))
        @test α₀ ≈ 4.18 atol = 1e-9
        @test β₀ ≈ 71.82 atol = 1e-9
        @test α₀ + β₀ ≈ 76.0 atol = 1e-9

        # An unobserved patch sits exactly at the landscape mean risk.
        @test BY.posterior_mean(0, 0, α₀, β₀) ≈ 0.055 atol = 1e-12

        # (mean, strength) is a bijection with (α₀, β₀).
        @test all(BY.resolve_prior(0.5, 2, (0.01, 0.1)) .≈ (1.0, 1.0))

        # Overriding the mean must not disturb the derived strength.
        a2, b2 = BY.resolve_prior(0.01, nothing, (0.01, 0.1))
        @test a2 + b2 ≈ 76.0 atol = 1e-9

        # α₀ < 1 is a shape change and must warn rather than fail.
        @test_logs (:warn,) BY.resolve_prior(0.055, 2, (0.01, 0.1))
    end

    @testset "posterior arithmetic matches the spec table" begin
        α₀, β₀ = BY.resolve_prior(nothing, nothing, (0.01, 0.1))
        @test BY.posterior_mean(5, 0, α₀, β₀) ≈ 0.0516 atol = 5e-5
        @test BY.posterior_mean(25, 0, α₀, β₀) ≈ 0.0414 atol = 5e-5
        @test BY.posterior_mean(25, 1, α₀, β₀) ≈ 0.0513 atol = 5e-5
        @test BY.posterior_mean(100, 0, α₀, β₀) ≈ 0.0237 atol = 5e-5
    end

    @testset "r = 0 reproduces the omniscient model exactly" begin
        # The hard regression test. Same seeds, same RNG stream, so the wealth
        # trajectories must agree bit for bit — not merely to a tolerance.
        for seed in 1:20
            m_old = BI.initialize(; base..., sensing_radius = 0, seed = seed)
            m_new = BY.initialize(; base..., sensing_radius = 0, seed = seed)
            @test m_old.profit == m_new.profit
            @test m_old.risk == m_new.risk
            for _ = 1:25
                step!(m_old, 1)
                step!(m_new, 1)
            end
            w_old = sort([a.wealth for a in allagents(m_old)])
            w_new = sort([a.wealth for a in allagents(m_new)])
            @test w_old == w_new           # exact, not approximate
            @test sort([a.pos for a in allagents(m_old)]) ==
                  sort([a.pos for a in allagents(m_new)])
        end
    end

    @testset "r = 0 ensemble matches the published baseline (44.9)" begin
        finals = Float64[]
        for seed in 1:100
            m = BY.initialize(; base..., sensing_radius = 0, seed = seed)
            step!(m, 25)
            push!(finals, BY.mean_wealth(m))
        end
        @test mean(finals) ≈ 44.9 atol = 0.5
    end

    @testset "hiding risk changes behaviour at r = 1" begin
        # Sanity: the models must NOT agree once investors can move, or the
        # belief layer would be doing nothing at all.
        m_old = BI.initialize(; base..., sensing_radius = 1, seed = 7)
        m_new = BY.initialize(; base..., sensing_radius = 1, seed = 7)
        step!(m_old, 25)
        step!(m_new, 25)
        @test sort([a.pos for a in allagents(m_old)]) !=
              sort([a.pos for a in allagents(m_new)])
    end

    @testset "observation records only occupied patches" begin
        # Deterministic landscape: no risk, uniform profit. Nobody is wiped and
        # nobody moves (no neighbour strictly beats staying), so the only thing
        # that changes is the belief memory.
        dims = (5, 5)
        m = BY.initialize(;
            dims,
            n_investors = 2,
            sensing_radius = 1,
            profit = fill(1.0, dims),
            risk = zeros(dims),
            seed = 3,
        )
        a, b = collect(allagents(m))
        move_agent!(a, (2, 2), m)
        move_agent!(b, (2, 3), m)   # adjacent to a
        for _ = 1:3
            step!(m, 1)
        end

        @test a.pos == (2, 2) && b.pos == (2, 3)      # nobody moved
        # Each sees its own patch and its one occupied neighbour, and nothing else.
        @test sort(collect(keys(a.beliefs))) == [(2, 2), (2, 3)]
        @test sort(collect(keys(b.beliefs))) == [(2, 2), (2, 3)]
        @test BY.belief(a, (2, 2)) == (3, 0)          # own patch, every step
        @test BY.belief(a, (2, 3)) == (3, 0)          # vicarious, every step
        @test BY.belief(a, (1, 1)) == (0, 0)          # empty neighbour: no evidence
        @test BY.mean_observed_patches(m) == 2.0
    end

    @testset "wipeouts are witnessed, by self and by neighbours" begin
        # A fully occupied 2×2 grid: nowhere to move, and at r = 1 every investor
        # sees every patch. Ruin fires every step, so k must track n everywhere —
        # on the own patch and on all three watched neighbours alike.
        dims = (2, 2)
        m = BY.initialize(;
            dims,
            n_investors = 4,            # every patch occupied, so nobody can move
            sensing_radius = 1,
            profit = fill(1.0, dims),
            risk = ones(dims),          # ruin fires every step
            seed = 3,
        )
        for _ = 1:4
            step!(m, 1)
        end
        for a in allagents(m)
            @test sort(collect(keys(a.beliefs))) ==
                  [(1, 1), (1, 2), (2, 1), (2, 2)]
            @test all(v == (4, 4) for v in values(a.beliefs))
        end
        @test m.n_ruins[] == 16         # 4 investors × 4 steps
        @test m.n_bankruptcies[] == 0   # profit is positive, so never insolvent
        @test m.n_moves[] == 0          # every patch occupied: no escape
    end

    @testset "post-wipeout flight reflex" begin
        # Being repeatedly wiped raises F̂ on the *occupied* patch above the prior
        # every unvisited patch still carries, so the investor eventually leaves —
        # the only risk-driven motion in the model. Uniform profit, so the profit
        # term cannot be what moves it.
        dims = (5, 5)
        m = BY.initialize(;
            dims,
            n_investors = 1,
            sensing_radius = 1,
            profit = fill(1.0, dims),
            risk = ones(dims),          # wiped every step wherever it stands
            seed = 3,
        )
        a = first(allagents(m))
        move_agent!(a, (3, 3), m)
        step!(m, 3)
        @test a.pos != (3, 3)           # fled the patch that kept burning it
        @test m.n_moves[] > 0
        # It flees onto patches it has no evidence about, by construction.
        @test BY.informed_move_fraction(m) == 0.0
    end

    @testset "bankruptcy is confounded with ruin" begin
        # Loss-making patch, zero risk: the investor is wiped every step by
        # bankruptcy alone, and its belief must climb as if the business were
        # dangerous. This is the confounding the model is meant to exhibit.
        dims = (3, 3)
        m = BY.initialize(;
            dims,
            n_investors = 1,
            sensing_radius = 0,         # pinned, so it cannot flee on profit
            profit = fill(-1.0, dims),
            risk = zeros(dims),
            seed = 5,
        )
        a = first(allagents(m))
        step!(m, 5)
        n, k = BY.belief(a, a.pos)
        @test (n, k) == (5, 5)
        @test m.n_bankruptcies[] == 5
        @test m.n_ruins[] == 0
        # F̂ = (4.18 + 5) / (76 + 5) = 0.1133 — misjudged as risky when merely poor.
        @test BY.posterior_mean(n, k, m.α₀, m.β₀) ≈ 0.1133 atol = 1e-4
    end

    @testset "beliefs are private" begin
        dims = (5, 5)
        m = BY.initialize(; dims, n_investors = 2, sensing_radius = 0, seed = 11)
        a, b = collect(allagents(m))
        @test a.beliefs !== b.beliefs      # not a shared Dict
        step!(m, 2)
        # At r = 0 each investor knows only its own patch, and nothing of the other.
        @test collect(keys(a.beliefs)) == [a.pos]
        @test collect(keys(b.beliefs)) == [b.pos]
    end

    @testset "metrics are well defined" begin
        m = BY.initialize(; base..., sensing_radius = 1, seed = 42)
        step!(m, 25)
        @test BY.density(m) ≈ 10 / 361
        @test 0 <= BY.mean_data_weight(m) < 1
        @test 0 <= BY.informed_move_fraction(m) <= 1
        @test BY.mean_posterior_shift(m) >= 0
        @test BY.mean_posterior_error(m) >= 0
        @test BY.mean_observed_patches(m) >= 1
    end
end
