export gpu_P2M!, gpu_M2M!

function gpu_P2M!(pr_positions::CuDeviceVector{SVector{3,T},1}, pr_momenta::CuDeviceVector{SVector{3,T},1}, ct_parindices::CuDeviceVector{I,1},
    mp_gammas::CuDeviceArray{T,4,1}, mp_momenta::CuDeviceArray{SVector{3,T},4,1}, degree::Val{D},
    cl_bboxes::CuDeviceVector{Tuple{SVector{3,T},SVector{3,T}},1}, cl_parlohis::CuDeviceVector{Tuple{I,I},1}, lfindices::UnitRange{I}) where {I,T,D}
    bid = blockIdx().x
    tid = threadIdx()

    nodeindex = lfindices[bid]
    lo, hi = cl_parlohis[nodeindex]
    mp_bmin, mp_bmax = cl_bboxes[nodeindex]

    xnodes = @MVector zeros(T,D+1)
    ynodes = @MVector zeros(T,D+1)
    znodes = @MVector zeros(T,D+1)
    for i in 1:(D+1)
        xnodes[i] = mp_bmin[1] + (cos(pi*(i-1)/D) + 1.0)/2.0 * (mp_bmax[1] - mp_bmin[1])
        ynodes[i] = mp_bmin[2] + (cos(pi*(i-1)/D) + 1.0)/2.0 * (mp_bmax[2] - mp_bmin[2])
        znodes[i] = mp_bmin[3] + (cos(pi*(i-1)/D) + 1.0)/2.0 * (mp_bmax[3] - mp_bmin[3])
    end

    mp_i, mp_j, mp_k = tid
    mp_x = xnodes[mp_i]
    mp_y = ynodes[mp_j]
    mp_z = znodes[mp_k]

    mp_ga = zero(T)
    mp_mom = SVector{3,T}(0.0,0.0,0.0)
    ws = @MVector zeros(T,D+1)
    for i in 1:(D+1)
        ws[i] = (i%2)==1 ? 1.0 : -1.0
    end
    ws[1] *= 0.5
    ws[D+1] *= 0.5
    for p in lo:hi
        pindex = ct_parindices[p]
        x,y,z = pr_positions[pindex]
        mom = pr_momenta[pindex]
        ga = sqrt(1.0 + dot(mom,mom))

        sum_x = zero(T)
        sum_y = zero(T)
        sum_z = zero(T)
        xflag = -1
        yflag = -1
        zflag = -1
        for i in 1:(D+1)
            dx = x-xnodes[i]
            dy = y-ynodes[i]
            dz = z-znodes[i]
            xflag = abs(dx) < eps(T) ? i : xflag
            yflag = abs(dy) < eps(T) ? i : yflag
            zflag = abs(dz) < eps(T) ? i : zflag
            sum_x += ws[i] / dx
            sum_y += ws[i] / dy
            sum_z += ws[i] / dz
        end

		if xflag == -1
		    wx = (ws[mp_i]/(x - mp_x))/sum_x
	    else
            wx = (mp_i == xflag) ? one(T) : zero(T)
		end

        if yflag == -1
		    wy = (ws[mp_j]/(y - mp_y))/sum_y
	    else
            wy = (mp_j == yflag) ? one(T) : zero(T)
		end

        if zflag == -1
		    wz = (ws[mp_k]/(z - mp_z))/sum_z
	    else
            wz = (mp_k == zflag) ? one(T) : zero(T)
		end

        weight = wx*wy*wz
        mp_ga += ga * weight
        mp_mom += mom * weight
    end
    mp_gammas[mp_i,mp_j,mp_k,nodeindex] += mp_ga
    mp_momenta[mp_i,mp_j,mp_k,nodeindex] += mp_mom
    return nothing
end

function gpu_M2M!(mp_gammas::CuDeviceArray{T,4,1}, mp_momenta::CuDeviceArray{SVector{3,T},4,1}, degree::Val{D},
        cl_bboxes::CuDeviceVector{Tuple{SVector{3,T},SVector{3,T}},1}, cl_children::CuDeviceVector{Tuple{I,I},1}, nodeindices::UnitRange{I}) where {I,T,D}
    bid = blockIdx().x
    p_i, p_j, p_k = threadIdx()

    parent_index = nodeindices[bid]
    child_indicies = cl_children[parent_index]
    p_bmin, p_bmax = cl_bboxes[parent_index]

    ws = @MVector zeros(T,D+1)
    for i in 1:(D+1)
        ws[i] = (i%2)==1 ? 1.0 : -1.0
    end
    ws[1] *= 0.5
    ws[D+1] *= 0.5

    p_xcoords = @MVector zeros(T,D+1)
    p_ycoords = @MVector zeros(T,D+1)
    p_zcoords = @MVector zeros(T,D+1)
    for i in 1:(D+1)
        p_xcoords[i] = p_bmin[1] + (cos(pi*(i-1)/D) + 1.0)/2.0 * (p_bmax[1] - p_bmin[1])
        p_ycoords[i] = p_bmin[2] + (cos(pi*(i-1)/D) + 1.0)/2.0 * (p_bmax[2] - p_bmin[2])
        p_zcoords[i] = p_bmin[3] + (cos(pi*(i-1)/D) + 1.0)/2.0 * (p_bmax[3] - p_bmin[3])
    end

    p_x = p_xcoords[p_i]
    p_y = p_ycoords[p_j]
    p_z = p_zcoords[p_k]
    p_ga = zero(T)
    p_mom = SVector{3,T}(0.0,0.0,0.0)
    for child_index in child_indicies
        c_bmin, c_bmax = cl_bboxes[child_index]
        for c_k in 1:(D+1)
            c_z = c_bmin[3] + (cos(pi*(c_k-1)/D) + 1.0)/2.0 * (c_bmax[3] - c_bmin[3])
            for c_j in 1:(D+1)
                c_y = c_bmin[2] + (cos(pi*(c_j-1)/D) + 1.0)/2.0 * (c_bmax[2] - c_bmin[2])
                for c_i in 1:(D+1)
                    c_x = c_bmin[1] + (cos(pi*(c_i-1)/D) + 1.0)/2.0 * (c_bmax[1] - c_bmin[1])
                    c_ga = mp_gammas[c_i,c_j,c_k,child_index]
                    c_mom = mp_momenta[c_i,c_j,c_k,child_index]

                    sum_x = zero(T)
                    sum_y = zero(T)
                    sum_z = zero(T)
                    xflag = -1
                    yflag = -1
                    zflag = -1
                    for i in 1:(D+1)
                        dx = c_x - p_xcoords[i]
                        dy = c_y - p_ycoords[i]
                        dz = c_z - p_zcoords[i]
                        xflag = abs(dx) < eps(T) ? i : xflag
                        yflag = abs(dy) < eps(T) ? i : yflag
                        zflag = abs(dz) < eps(T) ? i : zflag
                        sum_x += ws[i] / dx
                        sum_y += ws[i] / dy
                        sum_z += ws[i] / dz
                    end

                    if xflag == -1
                        wx = (ws[p_i]/(c_x - p_x)) / sum_x
                    else
                        wx = (p_i == xflag) ? one(T) : zero(T)
                    end

                    if yflag == -1
                        wy = (ws[p_j]/(c_y - p_y)) / sum_y
                    else
                        wy = (p_j == yflag) ? one(T) : zero(T)
                    end

                    if zflag == -1
                        wz = (ws[p_k]/(c_z - p_z)) / sum_z
                    else
                        wz = (p_k == zflag) ? one(T) : zero(T)
                    end
                    weight = wx*wy*wz
                    p_ga += c_ga * weight
                    p_mom += c_mom * weight
                end
            end
        end
    end
    mp_gammas[p_i,p_j,p_k,parent_index] += p_ga
    mp_momenta[p_i,p_j,p_k,parent_index] += p_mom
    return nothing
end

# function gpu_M2M!(mp_gammas::CuDeviceArray{T,4,1}, mp_momenta::CuDeviceArray{SVector{3,T},4,1}, degree::Val{D},
#         cl_bboxes::CuDeviceVector{Tuple{SVector{3,T},SVector{3,T}},1}, cl_parents::CuDeviceVector{I,1}, nodeindices::UnitRange{I}) where {I,T,D}
#     bid = blockIdx().x
#     p_i, p_j, p_k = threadIdx()

#     nodeindex = nodeindices[bid]

#     c_bmin, c_bmax = cl_bboxes[nodeindex]

#     parent_index = cl_parents[nodeindex]
#     p_bmin, p_bmax = cl_bboxes[parent_index]

#     ws = @MVector zeros(T,D+1)
#     for i in 1:(D+1)
#         ws[i] = (i%2)==1 ? 1.0 : -1.0
#     end
#     ws[1] *= 0.5
#     ws[D+1] *= 0.5

#     p_xcoords = @MVector zeros(T,D+1)
#     p_ycoords = @MVector zeros(T,D+1)
#     p_zcoords = @MVector zeros(T,D+1)
#     for i in 1:(D+1)
#         p_xcoords[i] = p_bmin[1] + (cos(pi*(i-1)/D) + 1.0)/2.0 * (p_bmax[1] - p_bmin[1])
#         p_ycoords[i] = p_bmin[2] + (cos(pi*(i-1)/D) + 1.0)/2.0 * (p_bmax[2] - p_bmin[2])
#         p_zcoords[i] = p_bmin[3] + (cos(pi*(i-1)/D) + 1.0)/2.0 * (p_bmax[3] - p_bmin[3])
#     end

#     p_x = p_xcoords[p_i]
#     p_y = p_ycoords[p_j]
#     p_z = p_zcoords[p_k]
#     p_ga = zero(T)
#     p_mom = SVector{3,T}(0.0,0.0,0.0)
#     for c_k in 1:(D+1)
#         c_z = c_bmin[3] + (cos(pi*(c_k-1)/D) + 1.0)/2.0 * (c_bmax[3] - c_bmin[3])
#         for c_j in 1:(D+1)
#             c_y = c_bmin[2] + (cos(pi*(c_j-1)/D) + 1.0)/2.0 * (c_bmax[2] - c_bmin[2])
#             for c_i in 1:(D+1)
#                 c_x = c_bmin[1] + (cos(pi*(c_i-1)/D) + 1.0)/2.0 * (c_bmax[1] - c_bmin[1])
#                 c_ga = mp_gammas[c_i,c_j,c_k,nodeindex]
#                 c_mom = mp_momenta[c_i,c_j,c_k,nodeindex]

#                 sum_x = zero(T)
#                 sum_y = zero(T)
#                 sum_z = zero(T)
#                 xflag = -1
#                 yflag = -1
#                 zflag = -1
#                 for i in 1:(D+1)
#                     dx = c_x - p_xcoords[i]
#                     dy = c_y - p_ycoords[i]
#                     dz = c_z - p_zcoords[i]
#                     xflag = abs(dx) < eps(T) ? i : xflag
#                     yflag = abs(dy) < eps(T) ? i : yflag
#                     zflag = abs(dz) < eps(T) ? i : zflag
#                     sum_x += ws[i] / dx
#                     sum_y += ws[i] / dy
#                     sum_z += ws[i] / dz
#                 end

#                 if xflag == -1
#                     wx = (ws[p_i]/(c_x - p_x)) / sum_x
#                 else
#                     wx = (p_i == xflag) ? one(T) : zero(T)
#                 end

#                 if yflag == -1
#                     wy = (ws[p_j]/(c_y - p_y)) / sum_y
#                 else
#                     wy = (p_j == yflag) ? one(T) : zero(T)
#                 end

#                 if zflag == -1
#                     wz = (ws[p_k]/(c_z - p_z)) / sum_z
#                 else
#                     wz = (p_k == zflag) ? one(T) : zero(T)
#                 end
#                 weight = wx*wy*wz
#                 p_ga += c_ga * weight
#                 p_mom += c_mom * weight
#             end
#         end
#     end
#     mp_gammas[p_i,p_j,p_k,parent_index] += p_ga
#     mp_momenta[p_i,p_j,p_k,parent_index] += p_mom
#     return nothing
# end

# function upwardpass!(mp_xcoords::CuDeviceArray{T,2,1}, mp_ycoords::CuDeviceArray{T,2,1}, mp_zcoords::CuDeviceArray{T,2,1},
#     mp_gammas::CuDeviceArray{SVector{3,T},2,1}, mp_momenta::CuDeviceArray{SVector{3,T},2,1}, cl_parents::CuDeviceVector{I,1}; max_level::I) where {I,T}
#     leafindicies = nodeindexrangeat(max_level)
#     nc = ncluster(max_level)
#     @cuda blocks=nc threads=(n+1,n+1,n+1) gpu_P2M!(pr_positions, pr_momenta, ct_parindices,
#     mp_xcoords, mp_ycoords, mp_zcoords, mp_gammas, mp_momenta,
#     cl_parlohis, leafindicies)
#     for l in max_level:-1:1
#         nodeindicies = nodeindexrangeat(l)
#         nc_in_level = 2^l
#         @cuda blocks=nc_in_level threads=(n+1,n+1,n+1) M2M!(mp_xcoords, mp_ycoords, mp_zcoords,
#         mp_gammas, mp_momenta,
#         cl_parents, nodeindicies)
#     end
#     return nothing
# end
