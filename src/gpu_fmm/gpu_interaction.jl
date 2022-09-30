export gpu_M2L!, gpu_P2P!

function gpu_M2L!(mp_gammas::CuDeviceArray{T,4,1}, mp_momenta::CuDeviceArray{SVector{3,T},4,1}, mp_efields::CuDeviceArray{SVector{3,T},4,1}, mp_bfields::CuDeviceArray{SVector{3,T},4,1}, degree::Val{D},
    cl_bboxes::CuDeviceVector{Tuple{SVector{3,T},SVector{3,T}},1},
    m2l_lists::CuDeviceVector{Tuple{I,I},1}, p_avg::SVector{3,T}) where {I,T,D}
    bid = blockIdx().x
    t_i, t_j, t_k = threadIdx()
    tindex, sindex = m2l_lists[bid]
    t_bmin, t_bmax = cl_bboxes[tindex]
    t_x = t_bmin[1] + (cos(pi*(t_i-1)/D) + 1.0)/2.0 * (t_bmax[1] - t_bmin[1])
    t_y = t_bmin[2] + (cos(pi*(t_j-1)/D) + 1.0)/2.0 * (t_bmax[2] - t_bmin[2])
    t_z = t_bmin[3] + (cos(pi*(t_k-1)/D) + 1.0)/2.0 * (t_bmax[3] - t_bmin[3])

    shared_s_gammas = CuStaticSharedArray(T, (D+1,D+1,D+1))
    shared_s_momenta = CuStaticSharedArray(SVector{3,T}, (D+1,D+1,D+1))
    shared_s_gammas[t_i, t_j, t_k] = mp_gammas[t_i, t_j, t_k, sindex]
    shared_s_momenta[t_i, t_j, t_k] = mp_momenta[t_i, t_j, t_k, sindex]
    sync_threads()

    s_bmin, s_bmax = cl_bboxes[sindex]
    t_e_sum = SVector{3,T}(0.0,0.0,0.0)
    t_b_sum = SVector{3,T}(0.0,0.0,0.0)
    for s_k in 1:(D+1)
        s_z = s_bmin[3] + (cos(pi*(s_k-1)/D) + 1.0)/2.0 * (s_bmax[3] - s_bmin[3])
        for s_j in 1:(D+1)
            s_y = s_bmin[2] + (cos(pi*(s_j-1)/D) + 1.0)/2.0 * (s_bmax[2] - s_bmin[2])
            for s_i in 1:(D+1)
                s_x = s_bmin[1] + (cos(pi*(s_i-1)/D) + 1.0)/2.0 * (s_bmax[1] - s_bmin[1])
                R = SVector(t_x-s_x, t_y-s_y, t_z-s_z)
                kernel = R / sqrt(dot(R, R) + dot(p_avg, R)^2 + eps())^3
                t_e_sum += shared_s_gammas[s_i,s_j,s_k] * kernel
                t_b_sum += cross(shared_s_momenta[s_i,s_j,s_k], kernel)
            end
        end
    end
    mp_efields[t_i,t_j,t_k,tindex] += t_e_sum
    mp_bfields[t_i,t_j,t_k,tindex] += t_b_sum
    return nothing
end

function gpu_P2P!(pr_positions::CuDeviceVector{SVector{3,T},1}, pr_momenta::CuDeviceVector{SVector{3,T},1}, pr_efields::CuDeviceVector{SVector{3,T},1}, pr_bfields::CuDeviceVector{SVector{3,T},1}, ct_parindices::CuDeviceVector{I,1},
    cl_parlohis::CuDeviceVector{Tuple{I,I},1}, VN0::Val{N0}, p2p_lists::CuDeviceVector{Tuple{I,I},1}) where {I,T,N0}
    bid = blockIdx().x
    block_size = blockDim().x
    tid = threadIdx().x
    tindex, sindex = p2p_lists[bid]

    t_parlo, t_parhi = cl_parlohis[tindex]
    s_parlo, s_parhi = cl_parlohis[sindex]
    shared_s_positions = CuStaticSharedArray(SVector{3,T}, N0)
    shared_s_momenta = CuStaticSharedArray(SVector{3,T}, N0)


    t_npar = t_parhi - t_parlo +1
    s_npar = s_parhi - s_parlo +1

    if tid <= s_npar
        j = ct_parindices[tid + s_parlo - 1]
        shared_s_positions[tid] = pr_positions[j]
        shared_s_momenta[tid] = pr_momenta[j]
        # @cuprintln(bid, " ", tid," ", pr_positions[j][1], " ", pr_positions[j][2], " ", pr_positions[j][3], " ", shared_s_positions[tid][1]," ",shared_s_positions[tid][2]," ",shared_s_positions[tid][3])
    end
    sync_threads()

    if tid <= t_npar
        t_e_sum = SVector{3,T}(0.0,0.0,0.0)
        t_b_sum = SVector{3,T}(0.0,0.0,0.0)
        i = ct_parindices[tid + t_parlo - 1]
        xi = pr_positions[i]
        for s_j in 1:s_npar
            xj = shared_s_positions[s_j]
            pj = shared_s_momenta[s_j]
            gj = sqrt(1.0 + dot(pj,pj))
            R = xi - xj
            Kij = R / sqrt(dot(R, R) + dot(pj, R)^2 + eps())^3
            t_e_sum += gj * Kij
            t_b_sum += cross(pj, Kij)
        end
        pr_efields[i] += t_e_sum
        pr_bfields[i] += t_b_sum
    end
    return nothing
end
