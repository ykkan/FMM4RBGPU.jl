export Clusters
export fillclusters!

struct Clusters{I<:Integer,T}
    levels::Vector{I}
    parlohis::Vector{Tuple{I,I}}
    parents::Vector{I}
    children::Vector{Tuple{I,I}}
    bboxes::Vector{Tuple{SVector{3,T},SVector{3,T}}}
end

function Clusters(nc::I) where {I<:Integer}
    levels = Vector{I}(undef, nc)
    parlohis = Vector{Tuple{I,I}}(undef, nc)
    parents = Vector{I}(undef, nc)
    children = Vector{Tuple{I,I}}(undef, nc)
    bboxes = fill((SVector(0.0,0.0,0.0),SVector(0.0,0.0,0.0)), nc)
    return Clusters(levels, parlohis, parents, children, bboxes)
end

function fillclusters!(clusters::Clusters{I,T}, nodeindex::I, particles::Particles{T}, parindices::Vector{I}, lohi, level; N0, stretch) where {I,T}
    lo, hi = lohi
    npar = (hi - lo + 1)
    bbox = BBox(particles, parindices, lo, hi)

    levels = clusters.levels
    parlohis = clusters.parlohis
    parents = clusters.parents
    children = clusters.children
    bboxes = clusters.bboxes
    levels[nodeindex] = level
    parlohis[nodeindex] = lohi
    bboxes[nodeindex] = bbox
    parents[nodeindex] = floor(Int, nodeindex/2)

    if npar <= N0
        children[nodeindex] = (-1,-1)
    else
        splitdir = argmax( stretch .* (bbox[2] - bbox[1]) )
        pvtindex = kpermute!(parindices, lo, hi, pick(particles.positions, splitdir); k=floor(Int, (lo+hi)/2))
        childindx1 = nodeindex * 2
        childindx2 = nodeindex * 2 + 1
        children[nodeindex] = (childindx1,childindx2)
        lohi1 = (lo, pvtindex)
        lohi2 = (pvtindex + 1, hi)
        fillclusters!(clusters, childindx1, particles, parindices, lohi1, level+1; N0, stretch)
        fillclusters!(clusters, childindx2, particles, parindices, lohi2, level+1; N0, stretch)
    end
end

function BBox(particles::Particles{T}, parindices::Vector{I}, lo::I, hi::I) where {I,T}
    pos = particles.positions
    tmin = typemin(T)
    tmax = typemax(T)
    xmin, xmax = tmax, tmin
    ymin, ymax = tmax, tmin
    zmin, zmax = tmax, tmin
    for i in lo:hi
        (x, y, z) = pos[parindices[i]]
        xmin = x < xmin ? x : xmin
        xmax = x > xmax ? x : xmax
        ymin = y < ymin ? y : ymin
        ymax = y > ymax ? y : ymax
        zmin = z < zmin ? z : zmin
        zmax = z > zmax ? z : zmax
    end
    bmin = SVector(xmin, ymin, zmin)
    bmax = SVector(xmax, ymax, zmax)
    return BBox(bmin, bmax)
end
