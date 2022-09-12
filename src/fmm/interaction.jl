export M2L!, P2P!, interact!

function M2L!(mp::MacroParticles{I,T}, m2l_lists::Vector{Tuple{I,I}}, nm2l::I; p_avg::SVector{3,T}) where {I,T}
    n = mp.n
    for i in 1:nm2l
        tindex, sindex = m2l_lists[i]
        s_xcoords = @view mp.xcoords[:,sindex]
        s_ycoords = @view mp.ycoords[:,sindex]
        s_zcoords = @view mp.zcoords[:,sindex]
        s_gammas = @view mp.gammas[:,:,:,sindex]
        s_momenta = @view mp.momenta[:,:,:,sindex]

        t_xcoords = @view mp.xcoords[:,tindex]
        t_ycoords = @view mp.ycoords[:,tindex]
        t_zcoords = @view mp.zcoords[:,tindex]
        t_efields = @view mp.efields[:,:,:,tindex]
        t_bfields = @view mp.bfields[:,:,:,tindex]

        for tk in 1:(n+1)
            tz = t_zcoords[tk]
            for tj in 1:(n+1)
                ty = t_ycoords[tj]
                for ti in 1:(n+1)
                    tx = t_xcoords[ti]
                    for sk in 1:(n+1)
                        sz = s_zcoords[sk]
                        for sj in 1:(n+1)
                            sy = s_ycoords[sj]
                                for si in 1:(n+1)
                                sx = s_xcoords[si]
                                R = SVector(tx-sx, ty-sy, tz-sz)
                                kernel = R / sqrt(dot(R, R) + dot(p_avg, R)^2 + eps())^3
                                t_efields[ti,tj,tk] += s_gammas[si,sj,sk] * kernel
                                t_bfields[ti,tj,tk] += cross(s_momenta[si,sj,sk], kernel)
                            end
                        end
                    end
                end
            end
        end
    end
end


function P2P!(particles::Particles{T}, parindices::Vector{Int}, clusters::Clusters{I,T}, p2p_lists::Vector{Tuple{I,I}}, np2p::I) where {I,T}
    positions = particles.positions
    momenta = particles.momenta
    efields = particles.efields
    bfields = particles.bfields
    parlohis = clusters.parlohis

    for l in 1:np2p
        tindex, sindex = p2p_lists[l]
        t_parlo, t_parhi = parlohis[tindex]
        s_parlo, s_parhi = parlohis[sindex]
        for i in t_parlo:t_parhi
            pari = parindices[i]
            xi = positions[pari]
            for j in s_parlo:s_parhi
                parj = parindices[j]
                xj = positions[parj]
                pj = momenta[parj]
                gj = sqrt(1.0 + dot(pj,pj))
                R = xi-xj
                Kij = R / sqrt(dot(R, R) + dot(pj, R)^2 + eps())^3
                efields[pari] += gj * Kij
                bfields[pari] += cross(pj, Kij)
            end
        end
    end
end

function interact!(mp::MacroParticles{I,T}, ct::ClusterTree{I,T}, itlists::InteractionLists{I}; p_avg) where {I,T}
    M2L!(mp, itlists.m2l_lists, itlists.nm2l; p_avg=p_avg)
    P2P!(ct.particles, ct.parindices, ct.clusters, itlists.p2p_lists, itlists.np2p)
end
