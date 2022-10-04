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

    weight_x = @MVector zeros(T,D+1)
    weight_y = @MVector zeros(T,D+1)
    weight_z = @MVector zeros(T,D+1)
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
        weight_x[i] = wdx
        weight_y[i] = wdy
        weight_z[i] = wdz
        xflag = abs(dx) < eps(T) ? i : xflag
        yflag = abs(dy) < eps(T) ? i : yflag
        zflag = abs(dz) < eps(T) ? i : zflag
        sum_x += wdx
        sum_y += wdy
        sum_z += wdz
    end

    if xflag == -1
        weight_x ./= sum_x
    else
        weight_x .= 0.0
        weight_x[xflag] = 1.0
    end

    if yflag == -1
        weight_y ./= sum_y
    else
        weight_y .= 0.0
        weight_y[yflag] = 1.0
    end

    if zflag == -1
        weight_z ./= sum_z
    else
        weight_z .= 0.0
        weight_z[zflag] = 1.0
    end

    c_e_sum = SVector{3,T}(0.0,0.0,0.0)
    c_b_sum = SVector{3,T}(0.0,0.0,0.0)
    for p_k in 1:(D+1)
        for p_j in 1:(D+1)
            for p_i in 1:(D+1)
                weight = weight_x[p_i] * weight_y[p_j] * weight_z[p_k]
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

    mp_bmin, mp_bmax = cl_bboxes[nodeindex]
    mp_xcoords = @MVector zeros(T,D+1)
    mp_ycoords = @MVector zeros(T,D+1)
    mp_zcoords = @MVector zeros(T,D+1)
    for i in 1:(D+1)
        mp_xcoords[i] = mp_bmin[1] + (cos(pi*(i-1)/D) + 1.0)/2.0 * (mp_bmax[1] - mp_bmin[1])
        mp_ycoords[i] = mp_bmin[2] + (cos(pi*(i-1)/D) + 1.0)/2.0 * (mp_bmax[2] - mp_bmin[2])
        mp_zcoords[i] = mp_bmin[3] + (cos(pi*(i-1)/D) + 1.0)/2.0 * (mp_bmax[3] - mp_bmin[3])
    end

    npar = parhi - parlo + 1
    if tid <= npar
        ws = @MVector zeros(T,D+1)
        for i in 1:(D+1)
            ws[i] = (i%2)==1 ? 1.0 : -1.0
        end
        ws[1] *= 0.5
        ws[D+1] *= 0.5

        i = ct_parindices[parlo + tid - 1]
        x, y, z = pr_positions[i]

        weight_x = @MVector zeros(T,D+1)
        weight_y = @MVector zeros(T,D+1)
        weight_z = @MVector zeros(T,D+1)
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
            weight_x[i] = wdx
            weight_y[i] = wdy
            weight_z[i] = wdz
            xflag = abs(dx) < eps(T) ? i : xflag
            yflag = abs(dy) < eps(T) ? i : yflag
            zflag = abs(dz) < eps(T) ? i : zflag
            sum_x += wdx
            sum_y += wdy
            sum_z += wdz
        end

        if xflag == -1
            weight_x ./= sum_x
        else
            weight_x .= 0.0
            weight_x[xflag] = 1.0
        end

        if yflag == -1
            weight_y ./= sum_y
        else
            weight_y .= 0.0
            weight_y[yflag] = 1.0
        end

        if zflag == -1
            weight_z ./= sum_z
        else
            weight_z .= 0.0
            weight_z[zflag] = 1.0
        end

        e_sum = SVector{3,T}(0.0,0.0,0.0)
        b_sum = SVector{3,T}(0.0,0.0,0.0)
        for mp_k in 1:(D+1)
            for mp_j in 1:(D+1)
                for mp_i in 1:(D+1)
                    weight = weight_x[mp_i] * weight_y[mp_j] * weight_z[mp_k]
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
