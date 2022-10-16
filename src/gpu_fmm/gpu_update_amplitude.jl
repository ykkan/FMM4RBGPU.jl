export gpu_update_amplitude!

function gpu_update_amplitude!(pr_efields::CuDeviceVector{SVector{3,T},1}, pr_bfields::CuDeviceVector{SVector{3,T},1}, N::I, amp::T) where {I,T}
    tid = threadIdx().x
    bid = blockIdx().x
    blocksize = blockDim().x
    gid = (bid - 1) * blocksize + tid
    if gid <= N
        pr_efields[gid] *= amp
        pr_bfields[gid] *= amp
    end
    return nothing
end
