export L2L!, L2P!, downwardpass!

function L2L!(mp::MacroParticles{I,T}, nodeindices::UnitRange{I}, ct::ClusterTree{I,T}) where {I,T}
    n = mp.n
    bboxes = ct.clusters.bboxes
    for nodeindex in nodeindices
        c_bmin, c_bmax = bboxes[nodeindex]
        c_xcoords = cheb2(n, c_bmin[1], c_bmax[1])
        c_ycoords = cheb2(n, c_bmin[2], c_bmax[2])
        c_zcoords = cheb2(n, c_bmin[3], c_bmax[3])
        c_efields = @view mp.efields[:,:,:,nodeindex]
        c_bfields = @view mp.bfields[:,:,:,nodeindex]

        parent_index = ct.clusters.parents[nodeindex]
        p_bmin, p_bmax = bboxes[parent_index]
        p_xcoords = cheb2(n, p_bmin[1], p_bmax[1])
        p_ycoords = cheb2(n, p_bmin[2], p_bmax[2])
        p_zcoords = cheb2(n, p_bmin[3], p_bmax[3])
        p_efields = @view mp.efields[:,:,:,parent_index]
        p_bfields = @view mp.bfields[:,:,:,parent_index]

        for ck in 1:(n+1)
            cz = c_zcoords[ck]
            weights_z = lgweight(cz, p_zcoords)
            for cj in 1:(n+1)
                cy = c_ycoords[cj]
                weights_y = lgweight(cy, p_ycoords)
                for ci in 1:(n+1)
                    cx = c_xcoords[ci]
                    weights_x = lgweight(cx, p_xcoords)

                    for pk in 1:(n+1)
                        for pj in 1:(n+1)
                            for pi in 1:(n+1)
                                weight = weights_x[pi]*weights_y[pj]*weights_z[pk]
                                c_efields[ci,cj,ck] += p_efields[pi,pj,pk] * weight
                                c_bfields[ci,cj,ck] += p_bfields[pi,pj,pk] * weight
                            end
                        end
                    end
                end
            end
        end
    end
end

function L2P!(mp::MacroParticles{I,T}, lfindices::UnitRange{I}, ct::ClusterTree{I,T}) where {I,T}
    n = mp.n
    particles = ct.particles
    parindices = ct.parindices
    parlohis = ct.clusters.parlohis
    bboxes = ct.clusters.bboxes

    efields = particles.efields
    bfields = particles.bfields
    for nodeindex in lfindices
        lo, hi = parlohis[nodeindex]
        bmin, bmax = bboxes[nodeindex]
        mp_xcoords = cheb2(n, bmin[1], bmax[1])
        mp_ycoords = cheb2(n, bmin[2], bmax[2])
        mp_zcoords = cheb2(n, bmin[3], bmax[3])
        for p = lo:hi
            parindex = parindices[p]
            x,y,z = particles.positions[parindex]
            weights_x = lgweight(x, mp_xcoords)
            weights_y = lgweight(y, mp_ycoords)
            weights_z = lgweight(z, mp_zcoords)

            for k in 1:(n + 1)
                for j in 1:(n + 1)
                    for i in 1:(n + 1)
                        weight = weights_x[i] * weights_y[j] * weights_z[k]
                        efields[parindex] += mp.efields[i,j,k,nodeindex] * weight
                        bfields[parindex] += mp.bfields[i,j,k,nodeindex] * weight
                    end
                end
            end
        end
    end
end

function downwardpass!(mp::MacroParticles{I,T}, ct::ClusterTree{I,T}; max_level) where {I,T}
    for l in 1:max_level
        nodeindicies = nodeindexrangeat(l)
        L2L!(mp, nodeindicies, ct)
    end
    leafindicies = nodeindexrangeat(max_level)
    L2P!(mp, leafindicies, ct)
end
