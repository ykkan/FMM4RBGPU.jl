export InteractionListsGPU

struct InteractionListsGPU{I}
    np2p::I
    nm2l::I
    np2pgroup::I
    nm2lgroup::I
    p2p_lists::Vector{Tuple{I,I}}
    m2l_lists::Vector{Tuple{I,I}}
    p2p_lists_ptrs::Vector{I}
    m2l_lists_ptrs::Vector{I}
end

function ilistptrs(lists::Vector{Tuple{I,I}}, nelem::I) where {I}
    lists_ptrs = [1]
    val = lists[1][1]
    for i in 1:nelem
        if lists[i][1] != val
            val = lists[i][1]
            push!(lists_ptrs, i)
        end
    end
    push!(lists_ptrs, nelem+1)
    return lists_ptrs
end

function InteractionListsGPU(clusters::Clusters{I,T}; stretch=stretch, eta=eta) where {I,T}
    p2p_lists = Vector{Tuple{I,I}}()
    m2l_lists = Vector{Tuple{I,I}}()
    dualtraversefill!(p2p_lists, m2l_lists, clusters, 1, 1; stretch=stretch, eta=eta)
    np2p = length(p2p_lists)
    nm2l = length(m2l_lists)
    sort!(p2p_lists, by = x -> x[1])
    sort!(m2l_lists, by = x -> x[1])
    p2p_lists_ptrs = ilistptrs(p2p_lists, np2p)
    m2l_lists_ptrs = ilistptrs(m2l_lists, nm2l)
    np2pgroup = length(p2p_lists_ptrs) - 1
    nm2lgroup = length(m2l_lists_ptrs) - 1
    return InteractionListsGPU(np2p, nm2l, np2pgroup, nm2lgroup, p2p_lists, m2l_lists, p2p_lists_ptrs, m2l_lists_ptrs)
end
