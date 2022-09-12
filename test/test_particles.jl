@testset verbose = true "Particles" begin
    @testset "construction" begin
        npar = 10
        pos = rand(3, npar)
        mom = rand(3, npar)
        new_pos = reinterpret(reshape, SVector{3,Float64}, pos)
        new_mom = reinterpret(reshape, SVector{3,Float64}, mom)
        particles = Particles(;pos=pos, mom=mom)
        @test particles.npar == npar
        @test particles.positions == new_pos
        @test particles.momenta == new_mom
        @test particles.efields == reinterpret(reshape, SVector{3,Float64}, zeros(3,npar))
        @test particles.efields == reinterpret(reshape, SVector{3,Float64}, zeros(3,npar))
    end

    @testset "`Beam` as a type alias" begin
        @test Beam isa Type{Particles}

        pos = zeros(3,10)
        mom = zeros(3,10)
        beam = Beam(;pos=pos, mom=mom)
        @test beam isa Beam
        @test beam isa Particles
    end
end
