@testset verbose = true "Particles" begin
    @testset "construction" begin
        npar = 10
        pos = rand(3, npar)
        mom = rand(3, npar)
        new_pos = Vector{SVector{3,Float64}}(undef, npar)
        new_mom = Vector{SVector{3,Float64}}(undef, npar)
        for i in 1:npar
            new_pos[i] = SVector{3,Float64}(pos[:,i])
            new_mom[i] = SVector{3,Float64}(mom[:,i])
        end
        particles = Particles(;pos=pos, mom=mom)
        @test particles.npar == npar
        @test particles.positions == new_pos
        @test particles.momenta == new_mom
        @test particles.efields == fill(SVector{3,Float64}(0,0,0), npar)
        @test particles.efields == fill(SVector{3,Float64}(0,0,0), npar)
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
