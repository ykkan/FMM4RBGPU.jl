export P2M!, M2M!, upwardpass!
# values of Lagragian polynomials on chevyschev point evaluated at x
function lgweight(x::T, nodes::Vector{T}) where {T}
    n = length(nodes) - 1
    w = [(-1.0)^(j) for j in 0:n]
    w[1] = 0.5
    w[n + 1] = 0.5 * w[n + 1]
    diffs = x .- nodes
    flag = findfirst(x->abs(x) < eps(T), diffs)
    values = zeros(n+1)
    if flag == nothing
        values .= w ./ diffs
        sum_v = sum(values)
        return values ./ sum_v
    else
        values[flag] = 1.0
        return values
    end
end

function lgweight(x::T, idx::I, nodes::Vector{T}, n::I) where {I,T}
    values = zeros(n+1)
    epsi = eps(T)
    values[1] = 0.5 / (abs(x - nodes[1]) + epsi)
    for i in 1:(n-1)
        values[i+1] = (-1.0)^(i%2)/(abs(x - nodes[i+1]) + epsi)
    end
    values[n+1] = 0.5*(-1.0)^(n%2) / (abs(x - nodes[n+1]) + epsi)
    return values[idx] / sum(values)
end


function P2M!(mp::MacroParticles{I,T}, lfindices::UnitRange{I}, ct::ClusterTree{I,T}) where {I,T}
    particles = ct.particles
    parindices = ct.parindices
    parlohis = ct.clusters.parlohis

    for nodeindex in lfindices
        lo, hi = parlohis[nodeindex]
        mpxcoords = mp.xcoords[:,nodeindex]
        mpycoords = mp.ycoords[:,nodeindex]
        mpzcoords = mp.zcoords[:,nodeindex]
        for p = lo:hi
            pindex = parindices[p]
            pos = particles.positions[pindex]
            mom = particles.momenta[pindex]
            ga = sqrt(1.0 + dot(mom,mom))

            x, y, z = pos
            # deposite to macro paritcles
            weights_x = lgweight(x, mp.xcoords[:,nodeindex])
            weights_y = lgweight(y, mp.ycoords[:,nodeindex])
            weights_z = lgweight(z, mp.zcoords[:,nodeindex])
            n = mp.n
            for k in 1:(n + 1)
                wz = lgweight(z, k, mpxcoords, n)
                for j in 1:(n + 1)
                    wy = lgweight(y, j, mpycoords, n)
                    for i in 1:(n + 1)
                        wx = lgweight(x, i, mpxcoords, n)
                        weight = wx * wy * wz
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
        for pk in 1:(n+1)
            for pj in 1:(n+1)
                for pi in 1:(n+1)
                    for ck in 1:(n+1)
                        cz = c_zcoords[ck]
                        wz = lgweight(cz, pk, p_zcoords, n)
                        for cj in 1:(n+1)
                            cy = c_ycoords[cj]
                            wy = lgweight(cy, pj, p_ycoords, n)
                            for ci in 1:(n+1)
                                cx = c_xcoords[ci]
                                wx = lgweight(cx, pi, p_xcoords, n)
                                weight = wx * wy * wz
                                p_gammas[pi,pj,pk] += c_gammas[ci,cj,ck] * weight
                                p_momenta[pi,pj,pk] += c_momenta[ci,cj,ck] * weight
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
