# Tests for the coercion-resentment model.
#
# The one that matters most is the limiting case: with coercion frozen, the
# families alone must settle at the steady state q*(π) of Schøyen (2021), inefficient
# interval included. Everything the authority does is built on that, so if it fails the ABM
# is not an extension of the paper's model. The others pin down the payoffs and
# each branch of the authority’s rule.
#
# Run with:  julia --project=. test/test_coercion_resentment.jl

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "CoercionResentment.jl"))
using Agents, Test, Statistics
using .CoercionResentment

const CR = CoercionResentment

@testset "Coercion-resentment model" begin

    @testset "payoffs and steady state" begin
        m = initialize(; n_families = 10)
        @test resentment(0.0, m) ≈ 0 atol = 1e-12
        # No coercion: both gains are Δu and the population splits evenly.
        @test gains(0.0, m) == (m.Δu, m.Δu)
        @test steady_state(0.0, m) ≈ 0.5

        # Without resentment, q* falls linearly to zero at π = Δu.
        flat = initialize(; n_families = 10, resentment_scale = 0)
        for π in (0.0, 1.0, 2.5, 4.0)
            @test steady_state(π, flat) ≈ (flat.Δu - π) / (2flat.Δu)
        end
        @test steady_state(8.0, flat) == 0

        # The default S-shape has an inefficient interval: q* falls, rises, falls again.
        q = [steady_state(π, m) for π = 0:0.01:10]
        dq = diff(q)
        @test any(dq .> 0)
        @test dq[1] < 0 && dq[end] < 0
        # Steeper than 2 somewhere, as Assumption 3 of the paper requires.
        @test maximum(diff([resentment(π, m) for π = 0:0.01:10]) ./ 0.01) > 2

        # At q*, the two efforts balance, τ^ns = τ^s, so the expected flow vanishes.
        for π in (1.0, 5.0, 9.0)
            τ_ns, τ_s = CR.efforts(steady_state(π, m), π, m)
            @test τ_ns ≈ τ_s
        end
    end

    @testset "families reproduce q*(π) with coercion frozen" begin
        for (i, π) in enumerate((0.0, 3.5, 5.3, 10.0))
            m = initialize(; π0 = π, q0 = 0.5, ease = 0, crackdown = 0, collapse_rate = 0,
                n_families = 2000, seed = i)
            h = history(m, 1500)
            @test all(==(π), h.coercion)
            @test mean(h.q[750:end]) ≈ steady_state(π, m) atol = 0.03
        end
    end

    @testset "the authority’s rule" begin
        # ns share above q*(π): it is shrinking, so the authority eases.
        m = initialize(; π0 = 9.0, q0 = 0.4, q_max = (0.5, 0.5), collapse_rate = 0)
        step!(m, 1)
        @test m.coercion[] ≈ 9.0 - m.ease
        @test !m.cracked_down[]

        # Growing but tolerated: hold.
        m = initialize(; π0 = 0.0, q0 = 0.2, q_max = (0.5, 0.5), collapse_rate = 0)
        step!(m, 1)
        @test m.coercion[] == 0.0

        # Growing and above q_max: crack down.
        m = initialize(; π0 = 1.0, q0 = 0.3, q_max = (0.1, 0.1), collapse_rate = 0)
        step!(m, 1)
        @test m.coercion[] ≈ 1.0 + m.crackdown
        @test m.cracked_down[]

        # Coercion stays on the scale.
        m = initialize(; π0 = 10.0, q0 = 0.05, q_max = (0.0, 0.0), collapse_rate = 0)
        step!(m, 5)
        @test m.coercion[] == m.π_max
        m = initialize(; π0 = 0.0, q0 = 0.6, q_max = (1.0, 1.0), collapse_rate = 0)
        step!(m, 5)
        @test m.coercion[] == 0.0
    end

    @testset "collapse" begin
        h = history(initialize(; collapse_rate = 0, seed = 3), 500)
        @test all(==(1), h.authority) && !any(h.collapse)
        # No resentment at π = 0, so no one to rise up.
        h = history(initialize(; π0 = 0.0, q_max = (1.0, 1.0), collapse_rate = 1, seed = 3), 200)
        @test !any(h.collapse)
        h = history(initialize(; π0 = 6.0, collapse_rate = 1, seed = 3), 200)
        @test count(h.collapse) == h.authority[end] - 1 > 0
    end

    @testset "reproducible from the seed" begin
        a = history(initialize(; seed = 11), 300)
        b = history(initialize(; seed = 11), 300)
        @test a.coercion == b.coercion && a.q == b.q
    end

    @testset "2021 rule: fixed coercion" begin
        m = initialize(; rule = :fixed, π0 = 5.0, q0 = 0.5, n_families = 2000, seed = 2)
        h = history(m, 3000)
        @test all(==(5.0), h.coercion)
        @test !any(h.crackdown) && !any(h.collapse)
        @test mean(h.q[500:end]) ≈ steady_state(5.0, m) atol = 0.02
    end

    @testset "2021 rule: exogenous constraint ρ, the theorem of restraint" begin
        p = initialize(; n_families = 10)
        πq = best_coercion(5.0, p)                 # π_q̲; ρ = 5 is inside the interval
        @test πq ≈ 3.58 atol = 0.01
        # Below the interval ρ binds; inside it the authority holds back at
        # π_q̲; past π^e_q̲ ≈ 8.3 ρ binds again; above the scale it is 10.
        @test best_coercion(2.0, p) ≈ 2.0
        @test all(c -> best_coercion(c, p) == πq, 3.7:0.1:8.2)
        @test best_coercion(9.0, p) ≈ 9.0
        @test best_coercion(12.0, p) == p.π_max
        # It really is the lowest q* left of ρ, up to the grid spacing:
        # at the quadratic minimum π_q̲ a 0.01 grid misses the true minimum by ~1e-6.
        for c in (1.0, 4.0, 6.5, 9.5)
            best = steady_state(best_coercion(c, p), p)
            @test all(π -> best <= steady_state(π, p) + 1e-5, range(0, c, length = 400))
        end
        # In the ABM: π jumps to its choice at once and q relaxes to q*(π_q̲).
        m = initialize(; rule = :exogenous, ρ = 6.0, q0 = 0.5, n_families = 2000, seed = 3)
        @test m.coercion[] == πq
        h = history(m, 2000)
        @test all(==(πq), h.coercion)
        @test mean(h.q[500:end]) ≈ steady_state(πq, m) atol = 0.02
    end

    @testset "2021 rule: insurrection constraint, path dependence" begin
        p = initialize(; n_families = 10)
        ρ = q -> insurrection_constraint(q, p)
        @test ρ(0.0) ≈ p.π_max + 0.5 atol = 1e-6   # a small ns group sets no constraint
        @test ρ(0.6) ≈ p.ρ_low atol = 1e-6
        @test issorted([ρ(q) for q = 0:0.01:1], rev = true)
        # Same parameters, two starting ns shares, two long-run states.
        a = history(initialize(; rule = :insurrection, q0 = 0.16, n_families = 3000, seed = 4), 1000)
        b = history(initialize(; rule = :insurrection, q0 = 0.30, n_families = 3000, seed = 5), 1000)
        @test a.coercion[end] == p.π_max
        @test b.coercion[end] ≈ p.ρ_low atol = 0.02
        @test mean(a.q[end-199:end]) < p.ρ_tip < mean(b.q[end-199:end])
    end

    @testset "rule is checked" begin
        @test_throws ArgumentError initialize(; rule = :rational)
    end
end
