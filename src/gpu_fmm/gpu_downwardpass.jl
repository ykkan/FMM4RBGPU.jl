export gpu_L2L!, gpu_L2P!


function gpu_L2L!(mp_efields::CuDeviceArray{SVector{3,T},4,1}, mp_bfields::CuDeviceArray{SVector{3,T},4,1}, degree::Val{D},
    cl_bboxes::CuDeviceVector{Tuple{SVector{3,T},SVector{3,T}},1},cl_parents::CuDeviceVector{I,1}, nodeindices::UnitRange{I}) where {I,T,D}
    bid = blockIdx().x
    c_i, c_j, c_k = threadIdx()
    cindex = nodeindices[bid]
    pindex = cl_parents[cindex]

    p_bmin, p_bmax = cl_bboxes[pindex]
    c_bmin, c_bmax = cl_bboxes[cindex]

    c_x = c_bmin[1] + (cos(pi*(c_i-1)/D) + 1.0)/2.0 * (c_bmax[1] - c_bmin[1])
    c_y = c_bmin[2] + (cos(pi*(c_j-1)/D) + 1.0)/2.0 * (c_bmax[2] - c_bmin[2])
    c_z = c_bmin[3] + (cos(pi*(c_k-1)/D) + 1.0)/2.0 * (c_bmax[3] - c_bmin[3])

    ws = baryweights(Val(D+1), T)

    p_xcoords = nsvector(i->cheb2coord(i, p_bmin[1], p_bmax[1], Val(D)), Val(D+1), T)
    p_ycoords = nsvector(i->cheb2coord(i, p_bmin[2], p_bmax[2], Val(D)), Val(D+1), T)
    p_zcoords = nsvector(i->cheb2coord(i, p_bmin[3], p_bmax[3], Val(D)), Val(D+1), T)

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
        wdx = ws[i] / dx
        wdy = ws[i] / dy
        wdz = ws[i] / dz
        xflag = abs(dx) < eps(T) ? i : xflag
        yflag = abs(dy) < eps(T) ? i : yflag
        zflag = abs(dz) < eps(T) ? i : zflag
        sum_x += wdx
        sum_y += wdy
        sum_z += wdz
    end

    c_e_sum = SVector{3,T}(0.0,0.0,0.0)
    c_b_sum = SVector{3,T}(0.0,0.0,0.0)
    for p_k in 1:(D+1)
        if zflag == -1
            wz = ws[p_k] /(c_z - p_zcoords[p_k]) / sum_z
        else
            wz = (p_k == zflag) ? 1.0 : 0.0
        end
        for p_j in 1:(D+1)
            if yflag == -1
                wy = ws[p_j] /(c_y - p_ycoords[p_j]) / sum_y
            else
                wy = (p_j == yflag) ? 1.0 : 0.0
            end
            for p_i in 1:(D+1)
                if xflag == -1
                    wx = ws[p_i] /(c_x - p_xcoords[p_i]) / sum_x
                else
                    wx = (p_i == xflag) ? 1.0 : 0.0
                end
                weight = wx * wy * wz
                c_e_sum += mp_efields[p_i,p_j,p_k,pindex] * weight
                c_b_sum += mp_bfields[p_i,p_j,p_k,pindex] * weight
            end
        end
    end
    mp_efields[c_i,c_j,c_k,cindex] += c_e_sum
    mp_bfields[c_i,c_j,c_k,cindex] += c_b_sum
    return nothing
end

function gpu_L2P!(pr_positions::CuDeviceVector{SVector{3,T},1}, pr_efields::CuDeviceVector{SVector{3,T},1}, pr_bfields::CuDeviceVector{SVector{3,T},1}, ct_parindices::CuDeviceVector{I,1},
    mp_efields::CuDeviceArray{SVector{3,T},4,1}, mp_bfields::CuDeviceArray{SVector{3,T},4,1}, degree::Val{D},
    cl_bboxes::CuDeviceVector{Tuple{SVector{3,T},SVector{3,T}},1}, cl_parlohis::CuDeviceVector{Tuple{I,I},1}, lfindices::UnitRange{I}) where {I,T,D}
    bid = blockIdx().x
    tid = threadIdx().x
    nodeindex = lfindices[bid]
    parlo, parhi = cl_parlohis[nodeindex]
    npar = parhi - parlo + 1
    if tid <= npar
        mp_bmin, mp_bmax = cl_bboxes[nodeindex]
        mp_xcoords = nsvector(i->cheb2coord(i, mp_bmin[1], mp_bmax[1], Val(D)), Val(D+1), T)
        mp_ycoords = nsvector(i->cheb2coord(i, mp_bmin[2], mp_bmax[2], Val(D)), Val(D+1), T)
        mp_zcoords = nsvector(i->cheb2coord(i, mp_bmin[3], mp_bmax[3], Val(D)), Val(D+1), T)

        i = ct_parindices[parlo + tid - 1]
        x, y, z = pr_positions[i]

        ws = baryweights(Val(D+1), T)
        sum_x = zero(T)
        sum_y = zero(T)
        sum_z = zero(T)
        xflag = -1
        yflag = -1
        zflag = -1
        for i in 1:(D+1)
            dx = x - mp_xcoords[i]
            dy = y - mp_ycoords[i]
            dz = z - mp_zcoords[i]
            wdx = ws[i] / dx
            wdy = ws[i] / dy
            wdz = ws[i] / dz
            xflag = abs(dx) < eps(T) ? i : xflag
            yflag = abs(dy) < eps(T) ? i : yflag
            zflag = abs(dz) < eps(T) ? i : zflag
            sum_x += wdx
            sum_y += wdy
            sum_z += wdz
        end

        e_sum = SVector{3,T}(0.0,0.0,0.0)
        b_sum = SVector{3,T}(0.0,0.0,0.0)
        for mp_k in 1:(D+1)
            if zflag == -1
                wz = ws[mp_k] /(z - mp_zcoords[mp_k]) / sum_z
            else
                wz = (mp_k == zflag) ? 1.0 : 0.0
            end
            for mp_j in 1:(D+1)
                if yflag == -1
                    wy = ws[mp_j] /(y - mp_ycoords[mp_j]) / sum_y
                else
                    wy = (mp_j == yflag) ? 1.0 : 0.0
                end
                for mp_i in 1:(D+1)
                    if xflag == -1
                        wx = ws[mp_i] /(x - mp_xcoords[mp_i]) / sum_x
                    else
                        wx = (mp_i == xflag) ? 1.0 : 0.0
                    end
                    weight = wx * wy * wz
                    e_sum += mp_efields[mp_i,mp_j,mp_k,nodeindex] * weight
                    b_sum += mp_bfields[mp_i,mp_j,mp_k,nodeindex] * weight
                end
            end
        end
        pr_efields[i] += e_sum
        pr_bfields[i] += b_sum
    end
    return nothing
end
