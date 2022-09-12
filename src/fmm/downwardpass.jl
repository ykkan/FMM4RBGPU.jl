export L2L!, L2P!, downwardpass!

function L2L!(mp::MacroParticles{I,T}, nodeindices::UnitRange{I}, ct::ClusterTree{I,T}) where {I,T}
    n = mp.n
    for nodeindex in nodeindices
        c_xcoords = @view mp.xcoords[:,nodeindex]
        c_ycoords = @view mp.ycoords[:,nodeindex]
        c_zcoords = @view mp.zcoords[:,nodeindex]
        c_efields = @view mp.efields[:,:,:,nodeindex]
        c_bfields = @view mp.bfields[:,:,:,nodeindex]

        parent_index = ct.clusters.parents[nodeindex]
        p_xcoords = mp.xcoords[:,parent_index]
        p_ycoords = mp.ycoords[:,parent_index]
        p_zcoords = mp.zcoords[:,parent_index]
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

    efields = particles.efields
    bfields = particles.bfields
    for nodeindex in lfindices
        lo, hi = parlohis[nodeindex]
        p_xcoords = mp.xcoords[:,nodeindex]
        p_ycoords = mp.ycoords[:,nodeindex]
        p_zcoords = mp.zcoords[:,nodeindex]
        for p = lo:hi
            parindex = parindices[p]
            x,y,z = particles.positions[parindex]
            weights_x = lgweight(x, p_xcoords)
            weights_y = lgweight(y, p_ycoords)
            weights_z = lgweight(z, p_zcoords)

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
