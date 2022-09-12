export BBox, diameter, inside

const BBox{T} = Tuple{SVector{3,T}, SVector{3,T}}

function BBox(bmin::SVector{3,T}, bmax::SVector{3,T}) where {T}
    return (bmin, bmax)
end

function diameter(bbox::BBox{T}; stretch=SVector(1.0,1.0,1.0)) where {T}
    return norm(stretch .* (bbox[2] - bbox[1]) ./ 2.0)
end

function inside(child::BBox{T}, parent::BBox{T}) where {T}
    return all(parent[1] .<= child[1]) && all(parent[2] .>= child[2])
end
