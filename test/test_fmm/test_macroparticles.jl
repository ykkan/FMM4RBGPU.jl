@testset verbose = true "MacroParticles" begin
    @testset "construction" begin
        N = 10000
        N0 = 32

        parindices = [1:N;]
        pos = rand(3,N)
        mom = zeros(3,N)
        particles = Beam(;pos=pos, mom=mom)
        nc = ncluster(N, N0)
        clusters = Clusters(nc)
        fillclusters!(clusters, 1, particles, parindices, (1,N), 0; N0=N0, stretch=SVector(1.0,1.0,1.0))

        n = 3
        mp = MacroParticles(clusters, n)
        xc = mp.xcoords[:,1]
        yc = mp.ycoords[:,1]
        zc = mp.zcoords[:,1]
        @test xc[1] > 0
        @test xc[n+1] < 1
        @test size(mp.gammas[:,:,:,4]) == (n+1,n+1,n+1)
    end
end
