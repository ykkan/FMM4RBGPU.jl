@testset verbose = true "gpu interaction" begin
    @testset "gpu M2L" begin
        clusters = Clusters(2)

        # test by considering two static clusters with large separate distance
        distance=10000.0
        n = 3
        bbox1 = BBox(SVector(-1.0,-1.0,-1.0),SVector(1.0,1.0,1.0))
        bbox2 = BBox(SVector(-1.0,-1.0,-1.0+distance),SVector(1.0,1.0,1.0+distance))
        clusters.bboxes[1] = bbox1
        clusters.bboxes[2] = bbox2
        ct = ClusterTree(Particles(;pos=zeros(3,5),mom=zeros(3,5)), [1:5;], clusters)

        mp = MacroParticles(clusters, n)
        mp.gammas[:,:,:,1] .= 1.0
        mp.gammas[:,:,:,2] .= 1.0
        m2l_lists = [(1,2), (2,1)]
        m2l_lists_ptrs = [1,2,3]
        nm2l = 2
        nm2lgroup = 2
        p_avg = SVector(0.0,0.0,0.0)

        d_mp_gammas = CuArray(mp.gammas)
        d_mp_momenta = CuArray(mp.momenta)
        d_mp_efields = CuArray(mp.efields)
        d_mp_bfields = CuArray(mp.bfields)

        d_cl_bboxes = CuArray(ct.clusters.bboxes)

        d_m2l_lists = CuArray(m2l_lists)
        d_m2l_lists_ptrs = CuArray(m2l_lists_ptrs)

        @cuda blocks=nm2lgroup threads=(n+1,n+1,n+1) gpu_M2L!(d_mp_gammas, d_mp_momenta, d_mp_efields, d_mp_bfields, Val(n),
                                                        d_cl_bboxes, d_m2l_lists, d_m2l_lists_ptrs,
                                                        p_avg)

        copyto!(mp.efields, d_mp_efields)
        copyto!(mp.bfields, d_mp_bfields)
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

    @testset "gpu P2P" begin
        npar = 200
        separation=10000.0
        pos = rand(3, npar)
        mom = zeros(3, npar)
        pos[3,1:div(npar,2)] .+= separation
        particles = Particles(;pos=pos, mom=mom)
        parindices = [1:npar;]



        clusters = Clusters(2)
        clusters.parlohis[1] = (1,div(npar,2))
        clusters.parlohis[2] = ((div(npar,2)+1),npar)

        p2p_lists = [(1,2), (2,1)]
        p2p_lists_ptrs = [1,2,3]
        nm2l = 2
        np2pgroup = 2

        d_pr_positions = CuArray(particles.positions)
        d_pr_momenta = CuArray(particles.momenta)
        d_pr_efields = CuArray(particles.efields)
        d_pr_bfields = CuArray(particles.bfields)

        d_ct_parindices = CuArray(parindices)

        d_cl_parlohis = CuArray(clusters.parlohis)

        d_p2p_lists = CuArray(p2p_lists)
        d_p2p_lists_ptrs = CuArray(p2p_lists_ptrs)
        block_size = 128
        @cuda blocks=np2pgroup threads=block_size shmem=(block_size*sizeof(particles.positions[1])) gpu_P2P!(d_pr_positions, d_pr_momenta, d_pr_efields, d_pr_bfields, d_ct_parindices, d_cl_parlohis, Val(block_size),d_p2p_lists,d_p2p_lists_ptrs)

        copyto!(particles.efields, d_pr_efields)
        copyto!(particles.bfields, d_pr_bfields)

        yy = SVector(0.0,0.0,0.0)
        xx = SVector(0.0,0.0,separation)
        far_efield = npar/2 * (xx - yy) / norm(xx-yy)^3
        # test field of one particle from cluster 1
        @test isapprox(particles.efields[1], far_efield; rtol=1e-3)
        @test isapprox(particles.bfields[1], SVector(0.0,0.0,0.0); rtol=1e-3)
        # test field of one particle from cluster 2
        @test isapprox(particles.efields[npar], (-1)*far_efield; rtol=1e-3)
        @test isapprox(particles.bfields[npar], SVector(0.0,0.0,0.0); rtol=1e-3)
    end
end
