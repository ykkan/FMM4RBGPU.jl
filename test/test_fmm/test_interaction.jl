@testset verbose=true "Interaction" begin
    @testset "M2L" begin
        clusters = Clusters(2)

        # test by considering two static clusters with large separate distance
        distance=10000.0
        n = 3
        bbox1 = BBox(SVector(-1.0,-1.0,-1.0),SVector(1.0,1.0,1.0))
        bbox2 = BBox(SVector(-1.0,-1.0,-1.0+distance),SVector(1.0,1.0,1.0+distance))
        clusters.bboxes[1] = bbox1
        clusters.bboxes[2] = bbox2

        mp = MacroParticles(clusters, n)
        mp.gammas[:,:,:,1] .= 1.0
        mp.gammas[:,:,:,2] .= 1.0
        m2l_lists = [(1,2), (2,1)]
        nm2l = 2
        p_avg = SVector(0.0,0.0,0.0)
        M2L!(mp, m2l_lists, nm2l; p_avg=p_avg)
        yy = SVector(0.0,0.0,0.0)
        xx = SVector(0.0,0.0,distance)
        far_efield = (n+1)^3 * (xx - yy) / norm(xx-yy)^3
        # experienced fields of cluster 2
        @test isapprox(mp.efields[1,1,1,2], far_efield; rtol=1e-3)
        @test isapprox(mp.bfields[1,1,1,2], SVector(0.0,0.0,0.0); rtol=1e-3)
        # experienced fields of cluster 1
        @test isapprox(mp.efields[1,1,1,1], -1.0 * far_efield; rtol=1e-3)
        @test isapprox(mp.bfields[1,1,1,1], SVector(0.0,0.0,0.0); rtol=1e-3)
    end

    @testset "P2P" begin
        npar = 200
        separation=10000.0
        pos = rand(3, npar)
        mom = zeros(3, npar)
        pos[3,101:npar] .+= separation
        particles = Particles(;pos=pos, mom=mom)
        parindices = [1:npar;]



        clusters = Clusters(2)
        clusters.parlohis[1] = (1,100)
        clusters.parlohis[2] = (101,npar)

        p2p_lists = [(1,2), (2,1)]
        np2p = 2

        P2P!(particles, parindices, clusters, p2p_lists, np2p)
        yy = SVector(0.0,0.0,0.0)
        xx = SVector(0.0,0.0,separation)
        far_efield = npar/2 * (xx - yy) / norm(xx-yy)^3
        # test field of one particle from cluster 1
        @test isapprox(particles.efields[1], -1.0 * far_efield; rtol=1e-3)
        @test isapprox(particles.bfields[1], SVector(0.0,0.0,0.0); rtol=1e-3)
        # test field of one particle from cluster 2
        @test isapprox(particles.efields[101], far_efield; rtol=1e-3)
        @test isapprox(particles.bfields[101], SVector(0.0,0.0,0.0); rtol=1e-3)
    end
end
