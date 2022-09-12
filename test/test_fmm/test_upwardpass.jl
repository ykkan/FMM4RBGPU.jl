@testset verbose = true "upwardpass" begin
    @testset "P2M!" begin
        N = 100
        N0 = 200

        pos = rand(3,N)
        mom = zeros(3,N)
        mom[3,:] .= 2.0
        particles = Particles(;pos=pos, mom=mom)
        ct = ClusterTree(particles; N0=N0, stretch=SVector(1.0,1.0,1.0))

        n = 2
        mp = MacroParticles(ct.clusters,2)

        P2M!(mp, leafindexrange(N,N0), ct)
        @test isapprox(sum(mp.momenta[:,:,:,1]), N * mom[:,1], rtol = 0.01)
    end

    @testset "upwardpass" begin
        N=1000
        n=2
        N0=80
        pos = rand(3,N)
        mom = zeros(3,N)
        gamma = 30.0
        p0 = sqrt(gamma^2 - 1.0)
        mom[3,:] .= p0
        particles = Particles(;pos=pos, mom=mom)
        ct = ClusterTree(particles; N0=N0, stretch=SVector(1.0,1.0,1.0))

        n = 2
        mp = MacroParticles(ct.clusters,2)
        max_level = maxlevel(N,N0)
        upwardpass!(mp, ct; max_level)
        @test isapprox(sum(mp.gammas[:,:,:,1]), N * gamma)
        @test isapprox(sum(mp.momenta[:,:,:,1]), N * mom[:,1])
    end
end
