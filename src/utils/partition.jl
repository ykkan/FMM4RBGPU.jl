export @swap!, partition!, kpermute!

macro swap!(a, b)
    return quote
        $(esc(a)), $(esc(b)) = $(esc(b)), $(esc(a))
    end
end

# Luuto partition
function partition!(labels::Vector{Int}, lo::Int, hi::Int, data::AbstractVector{T}; pvtindex=rand(lo:hi)) where {T}
pvtval = data[labels[pvtindex]]
    @swap!(labels[pvtindex], labels[hi])
    j = lo
    for i in lo:(hi - 1)
        if data[labels[i]] < pvtval
            @swap!(labels[j], labels[i])
            j = j + 1
        end
    end
    @swap!(labels[hi], labels[j])
    return j
end

"""
    kpermute!(labels::Vector{Int}, lo::Int, hi::Int, data::AbstractVector{T}; k) where {T}
Permute `labels` of `data` to `labels = [labels1, labels]` by the k-th largest
data value `kval` such that `data[labels1] .<= kval` and `data[labels2] .> kval`
"""
function kpermute!(labels::Vector{Int}, lo::Int, hi::Int, data::AbstractVector{T}; k) where {T}
    while true
        if lo == hi
            return lo
        end
        pindex = partition!(labels, lo, hi, data)
        if pindex == k
            return k
        elseif pindex > k
            hi = pindex - 1
        else
            lo = pindex + 1
        end
    end
end
