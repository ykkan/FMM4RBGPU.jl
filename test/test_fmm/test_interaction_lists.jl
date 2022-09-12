@testset verbose=true "InteractionLists" begin
    @testset "admissible" begin
        bbox1 = BBox(SVector(0.0,0.0,0.0), SVector(0.5,0.5,0.5))
        bbox2 = BBox(SVector(0.5,0.5,0.5), SVector(1.0,1.0,1.0))
        @test admissible(bbox1, bbox2; eta=0.50, stretch=SVector(1.0,1.0,1.0)) == false
        @test admissible(bbox1, bbox2; eta=0.55, stretch=SVector(1.0,1.0,2.0)) == true
    end

    @testset "interaction lists from two extrem etas" begin
        N=1000
        n=2
        N0=400
        pos = rand(3,N)
        mom = zeros(3,N)
        gamma = 30.0
        p0 = sqrt(gamma^2 - 1.0)
        mom[3,:] .= p0
        particles = Particles(;pos=pos, mom=mom)
        ct = ClusterTree(particles; N0=N0, stretch=SVector(1.0,1.0,1.0))
        itlists = InteractionLists(ct.clusters; stretch=SVector(1.0,1.0,1.0), eta=999.0)
        @test itlists.nm2l == 2

        itlists = InteractionLists(ct.clusters; stretch=SVector(1.0,1.0,1.0), eta=0.0)
        @test itlists.np2p == 16
    end
end
