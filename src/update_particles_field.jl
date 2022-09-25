export BruteForce
export FMM, FMMGPU
export update_particles_field!

import Base.@kwdef

struct BruteForce end

@kwdef struct FMM{T}
    n::Int
    N0::Int
    eta::T
end

@kwdef struct FMMGPU{T}
    n::Int
    N0::Int
    eta::T
end

function update_particles_field!(particles::Particles{T}, alg::BruteForce; lambda) where {T}
    q = particles.charge
    npar = particles.npar
    @inbounds for i in 1:npar
        particles.efields[i] = SVector(0.0, 0.0, 0.0)
        particles.bfields[i] = SVector(0.0, 0.0, 0.0)
        xi = particles.positions[i]
        amp = 2.8179403699772166e-15 * q / lambda
        for j in 1:npar
            xj = particles.positions[j]
            pj = particles.momenta[j]
            R = xi-xj
            Kij = R / sqrt(dot(R, R) + dot(pj, R)^2 + eps())^3
            particles.efields[i] += amp * sqrt(1.0 + dot(pj, pj)) * Kij
            particles.bfields[i] += amp * cross(pj, Kij)
        end
    end
end

function update_particles_field!(particles::Particles{T}, alg::FMM; lambda) where {T}
    (;n, N0, eta) = alg

    q = particles.charge
    N = particles.npar
    p_avg = sum(particles.momenta) / particles.npar
    g_avg = sqrt(1.0 + dot(p_avg, p_avg))
    stretch = SVector(1.0,1.0,g_avg)

    max_level = maxlevel(N, N0)
    ct = ClusterTree(particles; N0=N0, stretch=stretch)
    mp = MacroParticles(ct.clusters, n)
    upwardpass!(mp, ct; max_level=max_level)
    itlists = InteractionLists(ct.clusters; stretch=stretch, eta=eta)
    interact!(mp, ct, itlists; p_avg=p_avg)
    downwardpass!(mp, ct; max_level=max_level)

    efields = particles.efields
    bfields = particles.bfields
    amp = 2.8179403699772166e-15 * q / lambda
    @inbounds for i in 1:N
        efields[i] *= amp
        bfields[i] *= amp
    end
end

function update_particles_field!(particles::Particles{T}, alg::FMMGPU; lambda) where {T}
    (;n, N0, eta) = alg

    q = particles.charge
    N = particles.npar
    p_avg = sum(particles.momenta) / particles.npar
    g_avg = sqrt(1.0 + dot(p_avg, p_avg))
    stretch = SVector(1.0,1.0,g_avg)

    nc = ncluster(N,N0)
    ct = ClusterTree(particles; N0=N0, stretch=stretch)
    cl = ct.clusters
    pr = particles

    d_ct_parindices = CuArray(ct.parindices)

    d_cl_levels = CuArray(cl.levels)
    d_cl_parlohis = CuArray(cl.parlohis)
    d_cl_parents = CuArray(cl.parents)
    d_cl_children = CuArray(cl.children)
    d_cl_bboxes = CuArray(cl.bboxes)

    d_pr_positions = CuArray(pr.positions)
    d_pr_momenta = CuArray(pr.momenta)

    d_mp_xcoords = CuArray{T}(undef, n+1, nc)
    d_mp_ycoords = CuArray{T}(undef, n+1, nc)
    d_mp_zcoords = CuArray{T}(undef, n+1, nc)
    d_mp_gammas = CuArray{T}(undef,n+1,n+1,n+1,nc)
    d_mp_momenta = CuArray{SVector{3,T}}(undef,n+1,n+1,n+1,nc)
    d_mp_efields = CuArray{SVector{3,T}}(undef,n+1,n+1,n+1,nc)
    d_mp_bfields = CuArray{SVector{3,T}}(undef,n+1,n+1,n+1,nc)

    lfindices = leafindexrange(N, N0)
    nleafnode = lfindices[2] - lfindices[1] + 1
    @cuda blocks=nleafnode threads=(n+1,n+1,n+1) gpu_P2M!(d_pr_positions, d_pr_momenta, d_ct_parindices,
                                                d_mp_xcoords, d_mp_ycoords, d_mp_zcoords, d_mp_gammas, d_mp_momenta, n,
                                                d_cl_parlohis, lfindices)
    # upwardpass!(mp, ct; max_level=max_level)
    # itlists = InteractionLists(ct.clusters; stretch=stretch, eta=eta)
    # interact!(mp, ct, itlists; p_avg=p_avg)
    # downwardpass!(mp, ct; max_level=max_level)

    # efields = particles.efields
    # bfields = particles.bfields
    # amp = 2.8179403699772166e-15 * q / lambda
    # @inbounds for i in 1:N
    #     efields[i] *= amp
    #     bfields[i] *= amp
    # end
end
