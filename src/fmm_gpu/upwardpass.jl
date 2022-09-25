export P2M!, M2M!, upwardpass!

function gpu_lgweight!(x, nodes::AbstractVector{T}, n, values::AbstractVector{T}) where {T}
    bitarr = abs.((x .- nodes)) .< eps(T)
    if any(bitarr)
        return convert.(Float64, bitarr)
    end
    ws = [(-1.0)^(j%2) for j in 0:n]
    ws[1] = 0.5
    ws[n + 1] = 0.5 * ws[n + 1]
    for i in 1:(n+1)
        values[i] = ws[i] / (x - nodes[i])
    end
    sum_values = zero(T)
    for i in 1:(n+1)
        sum_values += values[i]
    end
    for i in 1:(n+1)
        values[i] = values[i] / sum_values
    end
    return nothing
end


function gpu_P2M!(pr_positions::CuDeviceVector{SVector{3,T},1}, pr_momenta::CuDeviceVector{SVector{3,T},1}, ct_parindices::CuDeviceVector{I,1},
    mp_xcoords::CuDeviceArray{T,2,1}, mp_ycoords::CuDeviceArray{T,2,1}, mp_zcoords::CuDeviceArray{T,2,1}, mp_gammas::CuDeviceArray{T,4,1}, mp_momenta::CuDeviceArray{SVector{3,T},4,1}, n,
    cl_parlohis::CuDeviceVector{Tuple{I,I},1}, lfindices::UnitRange{I}) where {I,T}
    bid = blockIdx().x
    tid = threadIdx()

    nodeindex = lfindices[1] + bid -1
    lo, hi = cl_parlohis[nodeindex]
    cxcoords = @view mp_xcoords[:,nodeindex]
    cycoords = @view mp_ycoords[:,nodeindex]
    czcoords = @view mp_zcoords[:,nodeindex]
    
    for p = lo:hi
        pindex = ct_parindices[p]
        pos = pr_positions[pindex]
        mom = pr_momenta[pindex]
        ga = sqrt(1.0 + dot(mom,mom))

        x, y, z = pos
        weights_x = CuArray{T}(undef, n+1)
        weights_y = CuArray{T}(undef, n+1)
        weights_z = CuArray{T}(undef, n+1)
        #deposite to macro paritcles
        gpu_lgweight!(x, cxcoords, weights_x, n)
        gpu_lgweight!(y, cycoords, weights_x, n)
        gpu_lgweight!(z, czcoords, weights_x, n)
        i, j ,k = tid
        weight = weights_x[i] * weights_y[j] * weights_z[k]
        mp_gammas[i,j,k,nodeindex] += ga * weight
        mp_momenta[i,j,k,nodeindex] += mom * weight
    end
    return nothing
end



function M2M!(mp_xcoords::CuDeviceArray{T,2,1}, mp_ycoords::CuDeviceArray{T,2,1}, mp_zcoords::CuDeviceArray{T,2,1},
              mp_gammas::CuDeviceArray{SVector{3,T},2,1}, mp_momenta::CuDeviceArray{SVector{3,T},2,1},
              cl_parents::CuDeviceVector{I,1}, nodeindices::UnitRange{I}) where {I,T}
    n = mp.n
    bid = blockIdx()
    tid =
    nodeindex = bid + nodeindices[1] - 1
    parent_index = cl_parents[nodeindex]
    c_xcoords = @view mp_xcoords[:,nodeindex]
    c_ycoords = @view mp_ycoords[:,nodeindex]
    c_zcoords = @view mp_zcoords[:,nodeindex]
    c_gammas = @view mp_gammas[:,:,:,nodeindex]
    c_momenta = @view mp_momenta[:,:,:,nodeindex]

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
    return nothing
end

function upwardpass!(mp_xcoords::CuDeviceArray{T,2,1}, mp_ycoords::CuDeviceArray{T,2,1}, mp_zcoords::CuDeviceArray{T,2,1},
    mp_gammas::CuDeviceArray{SVector{3,T},2,1}, mp_momenta::CuDeviceArray{SVector{3,T},2,1}, cl_parents::CuDeviceVector{I,1}; max_level::I) where {I,T}
    leafindicies = nodeindexrangeat(max_level)
    nc = ncluster(max_level)
    @cuda blocks=nc threads=(n+1,n+1,n+1) gpu_P2M!(pr_positions, pr_momenta, ct_parindices,
    mp_xcoords, mp_ycoords, mp_zcoords, mp_gammas, mp_momenta,
    cl_parlohis, leafindicies)
    for l in max_level:-1:1
        nodeindicies = nodeindexrangeat(l)
        nc_in_level = 2^l
        @cuda blocks=nc_in_level threads=(n+1,n+1,n+1) M2M!(mp_xcoords, mp_ycoords, mp_zcoords,
        mp_gammas, mp_momenta,
        cl_parents, nodeindicies)
    end
    return nothing
end
