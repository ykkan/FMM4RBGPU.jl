export InteractionLists, admissible, dualtraversefill!

struct InteractionLists{I}
    np2p::I
    nm2l::I
    p2p_lists::Vector{Tuple{I,I}}
    m2l_lists::Vector{Tuple{I,I}}
end

function admissible(bbox1::BBox{T}, bbox2::BBox{T}; stretch=stretch, eta=eta) where {T}
    bmin1 = bbox1[1]
    bmax1 = bbox1[2]
    bmin2 = bbox2[1]
    bmax2 = bbox2[2]
    xc1 = (bmin1 + bmax1)/2.0
    xc2 = (bmin2 + bmax2)/2.0
    r1 = norm(stretch .* (bmax1 - xc1))
    r2 = norm(stretch .* (bmax2 - xc2))
    R = norm(stretch .* (xc1 - xc2))
    return max(r1,r2)/R < eta
end

function dualtraversefill!(p2p_lists::Vector{Tuple{I,I}}, m2l_lists::Vector{Tuple{I,I}},clusters::Clusters{I,T}, tindex::I, sindex::I; stretch, eta) where {I,T}
    children = clusters.children
    bboxes = clusters.bboxes
    tchildindicies = children[tindex]
    schildindicies = children[sindex]
    if (tchildindicies == (-1,-1) && schildindicies == (-1,-1))
        push!(p2p_lists, (tindex, sindex))
    else
        if admissible(bboxes[tindex], bboxes[sindex]; stretch=stretch, eta=eta)
            push!(m2l_lists, (tindex, sindex))
        elseif tchildindicies == (-1,-1)
            dualtraversefill!(p2p_lists, m2l_lists, clusters, tindex, schildindicies[1]; stretch=stretch, eta=eta)
            dualtraversefill!(p2p_lists, m2l_lists, clusters, tindex, schildindicies[2]; stretch=stretch, eta=eta)
        elseif schildindicies == (-1,-1)
            dualtraversefill!(p2p_lists, m2l_lists, clusters, tchildindicies[1], sindex; stretch=stretch, eta=eta)
            dualtraversefill!(p2p_lists, m2l_lists, clusters, tchildindicies[2], sindex; stretch=stretch, eta=eta)
        else
            if diameter(bboxes[tindex]; stretch=stretch) > diameter(bboxes[sindex]; stretch=stretch)
                dualtraversefill!(p2p_lists, m2l_lists, clusters, tchildindicies[1], sindex; stretch=stretch, eta=eta)
                dualtraversefill!(p2p_lists, m2l_lists, clusters, tchildindicies[2], sindex; stretch=stretch, eta=eta)
            else
                dualtraversefill!(p2p_lists, m2l_lists, clusters, tindex, schildindicies[1]; stretch=stretch, eta=eta)
                dualtraversefill!(p2p_lists, m2l_lists, clusters, tindex, schildindicies[2]; stretch=stretch, eta=eta)
            end
        end
    end
end

function InteractionLists(clusters::Clusters{I,T}; stretch=stretch, eta=eta) where {I,T}
    p2p_lists = Vector{Tuple{I,I}}()
    m2l_lists = Vector{Tuple{I,I}}()
    dualtraversefill!(p2p_lists, m2l_lists, clusters, 1, 1; stretch=stretch, eta=eta)
    np2p = length(p2p_lists)
    nm2l = length(m2l_lists)
    return InteractionLists(np2p, nm2l, p2p_lists, m2l_lists)
end
