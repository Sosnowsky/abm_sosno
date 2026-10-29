# Tests for the coercion-resentment model.
#
# The one that matters most is the limiting case: with coercion frozen, the
# families alone must settle at the steady state q*(π) of Schøyen (2021), window
# included. Everything the regime does is built on that, so if it fails the ABM
# is not an extension of the paper's model. The others pin down the payoffs and
# each branch of the regime's rule.
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

        # The default S-shape has a window: q* falls, rises, falls again.
        q = [steady_state(π, m) for π = 0:0.01:10]
        dq = diff(q)
        @test any(dq .> 0)
        @test dq[1] < 0 && dq[end] < 0
        # Steeper than 2 somewhere, as Assumption 3 of the paper requires.
        @test maximum(diff([resentment(π, m) for π = 0:0.01:10]) ./ 0.01) > 2

        # At q*, the two efforts balance, so the expected flow vanishes.
        for π in (1.0, 5.0, 9.0)
            τ_m, τ_M = CR.efforts(steady_state(π, m), π, m)
            @test τ_m ≈ τ_M
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

    @testset "the regime's rule" begin
        # Minority above q*(π): it is shrinking, so the regime eases.
        m = initialize(; π0 = 9.0, q0 = 0.4, tolerance = (0.5, 0.5), collapse_rate = 0)
        step!(m, 1)
        @test m.coercion[] ≈ 9.0 - m.ease
        @test !m.cracked_down[]

        # Growing but tolerated: hold.
        m = initialize(; π0 = 0.0, q0 = 0.2, tolerance = (0.5, 0.5), collapse_rate = 0)
        step!(m, 1)
        @test m.coercion[] == 0.0

        # Growing and above tolerance: crack down.
        m = initialize(; π0 = 1.0, q0 = 0.3, tolerance = (0.1, 0.1), collapse_rate = 0)
        step!(m, 1)
        @test m.coercion[] ≈ 1.0 + m.crackdown
        @test m.cracked_down[]

        # Coercion stays on the scale.
        m = initialize(; π0 = 10.0, q0 = 0.05, tolerance = (0.0, 0.0), collapse_rate = 0)
        step!(m, 5)
        @test m.coercion[] == m.π_max
        m = initialize(; π0 = 0.0, q0 = 0.6, tolerance = (1.0, 1.0), collapse_rate = 0)
        step!(m, 5)
        @test m.coercion[] == 0.0
    end

    @testset "collapse" begin
        h = history(initialize(; collapse_rate = 0, seed = 3), 500)
        @test all(==(1), h.regime) && !any(h.collapse)
        # No resentment at π = 0, so no one to rise up.
        h = history(initialize(; π0 = 0.0, tolerance = (1.0, 1.0), collapse_rate = 1, seed = 3), 200)
        @test !any(h.collapse)
        h = history(initialize(; π0 = 6.0, collapse_rate = 1, seed = 3), 200)
        @test count(h.collapse) == h.regime[end] - 1 > 0
    end

    @testset "reproducible from the seed" begin
        a = history(initialize(; seed = 11), 300)
        b = history(initialize(; seed = 11), 300)
        @test a.coercion == b.coercion && a.q == b.q
    end
end
