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

        d_ct_parindices = CuArray(ct.parindices)

        d_pr_positions = CuArray(ct.particles.positions)
        d_pr_momenta = CuArray(ct.particles.momenta)

        d_cl_bboxes = CuArray(ct.clusters.bboxes)
        d_cl_parlohis = CuArray(ct.clusters.parlohis)

        nc = ncluster(N,N0)
        d_mp_gammas = CuArray{Float64}(undef,n+1,n+1,n+1,nc)
        d_mp_momenta = CuArray{SVector{3,Float64}}(undef,n+1,n+1,n+1,nc)

        lfindices = leafindexrange(N, N0)
        nleafnode = 1
        @cuda threads=(n+1,n+1,n+1) blocks=nleafnode gpu_P2M!(d_pr_positions, d_pr_momenta, d_ct_parindices,
                                                    d_mp_gammas, d_mp_momenta, Val(n),
                                                    d_cl_bboxes, d_cl_parlohis, lfindices)


        copyto!(mp.gammas, d_mp_gammas)
        copyto!(mp.momenta, d_mp_momenta)
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

        degree = Val(n)
        mp = MacroParticles(ct.clusters,2)

        d_ct_parindices = CuArray(ct.parindices)

        d_pr_positions = CuArray(ct.particles.positions)
        d_pr_momenta = CuArray(ct.particles.momenta)

        d_cl_bboxes = CuArray(ct.clusters.bboxes)
        d_cl_parlohis = CuArray(ct.clusters.parlohis)
        d_cl_parents = CuArray(ct.clusters.parents)
        d_cl_children = CuArray(ct.clusters.children)

        nc = ncluster(N,N0)
        d_mp_gammas = CuArray{Float64}(undef,n+1,n+1,n+1,nc)
        d_mp_momenta = CuArray{SVector{3,Float64}}(undef,n+1,n+1,n+1,nc)
        max_level = maxlevel(N,N0)

        leafindicies = nodeindexrangeat(max_level)
        @cuda blocks=length(leafindicies) threads=(n+1,n+1,n+1) gpu_P2M!(d_pr_positions, d_pr_momenta, d_ct_parindices,
                                                    d_mp_gammas, d_mp_momenta, degree,
                                                    d_cl_bboxes, d_cl_parlohis, leafindicies)

        for l in (max_level-1):-1:0
            nodeindicies = nodeindexrangeat(l)
            nc_in_level = 2^l
            @cuda blocks=nc_in_level threads=(n+1,n+1,n+1) gpu_M2M!(d_mp_gammas, d_mp_momenta, degree,
                                                                d_cl_bboxes, d_cl_children, nodeindicies)
        end

        copyto!(mp.gammas, d_mp_gammas)
        copyto!(mp.momenta, d_mp_momenta)
        @test isapprox(sum(mp.gammas[:,:,:,1]), N * gamma)
        @test isapprox(sum(mp.momenta[:,:,:,1]), N * mom[:,1])
    end
end
