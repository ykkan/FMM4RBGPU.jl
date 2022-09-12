@testset "Clusters" begin
    @testset "contrution" begin
        nelem = 10
        cs = Clusters(10)
        @test length(cs.levels) == nelem
        @test length(cs.parlohis) == nelem
        @test length(cs.parents) == nelem
    end

    @testset "contrution from particles" begin
       N = 10000
       N0 = 32

       parindices = [1:N;]
       pos = rand(3,N)
       mom = zeros(3,N)
       particles = Beam(;pos=pos, mom=mom)
       nc = ncluster(N, N0)
       clusters = Clusters(nc)
       fillclusters!(clusters, 1, particles, parindices, (1,N), 0; N0=N0, stretch=SVector(1.0,1.0,1.0))
       @test length(clusters.levels) .== nc

       ml = maxlevel(N, N0)

       lrg = leafindexrange(N,N0)
       @test all(x->x== ml, clusters.levels[lrg]) == true
       @test all(x->x== (-1,-1), clusters.children[lrg]) == true
    end
end
