export P2M!, M2M!, upwardpass!
# values of Lagragian polynomials on chevyschev point evaluated at x

function lgweight(x::T, nodes::SVector{T}) where {T}
    n = length(nodes) - 1
    bitarr = abs.((x .- nodes)) .< eps(T)
    if any(bitarr)
        return convert.(Float64, bitarr)
    end
    ws = [(-1.0)^(j%2) for j in 0:n]
    ws[1] = 0.5
    ws[n + 1] = 0.5 * ws[n + 1]
    values = ws ./ (x .- nodes)
    return values ./ sum(values)
end

function P2M!(mp::MacroParticles{I,T}, lfindices::UnitRange{I}, ct::ClusterTree{I,T}) where {I,T}
    particles = ct.particles
    parindices = ct.parindices
    parlohis = ct.clusters.parlohis

    for nodeindex in lfindices
        lo, hi = parlohis[nodeindex]
        mp_xcoords = mp.xcoords[:,nodeindex]
        mp_ycoords = mp.ycoords[:,nodeindex]
        mp_zcoords = mp.zcoords[:,nodeindex]
        for p = lo:hi
            pindex = parindices[p]
            pos = particles.positions[pindex]
            mom = particles.momenta[pindex]
            ga = sqrt(1.0 + dot(mom,mom))

            x, y, z = pos
            n = mp.n
            #deposite to macro paritcles
            weights_x = lgweight(x, mp_xcoords)
            weights_y = lgweight(y, mp_ycoords)
            weights_z = lgweight(z, mp_zcoords)
            for k in 1:(n + 1)
                for j in 1:(n + 1)
                    for i in 1:(n + 1)
                        weight = weights_x[i] * weights_y[j] * weights_z[k]
                        mp.gammas[i,j,k,nodeindex] += ga * weight
                        mp.momenta[i,j,k,nodeindex] += mom * weight
                    end
                end
            end
        end
    end
end



function M2M!(mp::MacroParticles{I,T}, nodeindices::UnitRange{I}, ct::ClusterTree{I,T}) where {I,T}
    n = mp.n
    for nodeindex in nodeindices
        parent_index = ct.clusters.parents[nodeindex]
        c_xcoords = @view mp.xcoords[:,nodeindex]
        c_ycoords = @view mp.ycoords[:,nodeindex]
        c_zcoords = @view mp.zcoords[:,nodeindex]
        c_gammas = @view mp.gammas[:,:,:,nodeindex]
        c_momenta = @view mp.momenta[:,:,:,nodeindex]

        p_xcoords = mp.xcoords[:,parent_index]
        p_ycoords = mp.ycoords[:,parent_index]
        p_zcoords = mp.zcoords[:,parent_index]
        p_gammas = @view mp.gammas[:,:,:,parent_index]
        p_momenta = @view mp.momenta[:,:,:,parent_index]
        for ck in 1:(n+1)
            cz = c_zcoords[ck]
            weights_z = lgweight(cz, p_zcoords)
            for cj in 1:(n+1)
                cy = c_ycoords[cj]
                weights_y = lgweight(cy, p_ycoords)
                for ci in 1:(n+1)
                    cx = c_xcoords[ci]
                    weights_x = lgweight(cx, p_xcoords)
                    c_ga = c_gammas[ci,cj,ck]
                    c_mom = c_momenta[ci,cj,ck]
                    for pk in 1:(n+1)
                        for pj in 1:(n+1)
                            for pi in 1:(n+1)
                                weight = weights_x[pi]*weights_y[pj]*weights_z[pk]
                                p_gammas[pi,pj,pk] += c_ga * weight
                                p_momenta[pi,pj,pk] += c_mom * weight
                            end
                        end
                    end
                end
            end
        end
    end
end

function upwardpass!(mp::MacroParticles{I,T}, ct::ClusterTree{I,T}; max_level) where {I,T}
    leafindicies = nodeindexrangeat(max_level)
    P2M!(mp, leafindicies, ct)
    for l in max_level:-1:1
        nodeindicies = nodeindexrangeat(l)
        M2M!(mp, nodeindicies, ct)
    end
end
