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

# function deposite!(mp::MacroParticles{I,T}, nodeindex, pos, gamma, mom) where {I,T}
#     x, y, z = pos
#     weights_x = weight(x, mp.xcoords[:,nodeindex])
#     weights_y = weight(y, mp.ycoords[:,nodeindex])
#     weights_z = weight(z, mp.zcoords[:,nodeindex])
#     n = mp.n
#     for k in 1:(n + 1)
#         for j in 1:(n + 1)
#             for i in 1:(n + 1)
#                 weight = weights_x[i] * weights_y[j] * weights_z[k]
#                 mp.gammas[i,j,k,nodeindex] += gamma * weight
#                 mp.momenta[i,j,k,nodeindex] += mom * weight
#             end
#         end
#     end
# end

function P2M!(mp::MacroParticles{I,T}, lfindices::UnitRange{I}, ct::ClusterTree{I,T}) where {I,T}
    particles = ct.particles
    parindices = ct.parindices
    parlohis = ct.clusters.parlohis

    for nodeindex in lfindices
        lo, hi = parlohis[nodeindex]
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
        # should be considered in gpu implementation
        # single node weight needs to be implemented
        # for pk in 1:(n+1)
        #     for pj in 1:(n+1)
        #         for pi in 1:(n+1)
        #             for ck in 1:(n+1)
        #                 cz = c_zcoords[ck]
        #                 weights_z = lgweight(cz, p_zcoords)
        #                 for cj in 1:(n+1)
        #                     cy = c_ycoords[cj]
        #                     weights_y = lgweight(cy, p_ycoords)
        #                     for ci in 1:(n+1)
        #                         cx = c_xcoords[ci]
        #                         weights_x = lgweight(cx, p_xcoords)

        #                         weight = weights_x[pi]*weights_y[pj]*weights_z[pk]
        #                         p_gammas[pi,pj,pk] += c_gammas[ci,cj,ck] * weight
        #                         p_momenta[pi,pj,pk] += c_momenta[ci,cj,ck] * weight
        #                     end
        #                 end
        #             end
        #         end
        #     end
        # end






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
