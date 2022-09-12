export BruteForce
export FMM
export update_particles_field!

import Base.@kwdef

struct BruteForce end

@kwdef struct FMM{T}
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
