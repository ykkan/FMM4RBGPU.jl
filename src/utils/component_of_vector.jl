export ComponentOfSVectors, pick

struct ComponentOfSVectors{T,A<:AbstractVector{SVector{3,T}}} <: AbstractVector{T}
    component::Int
    data::A
end

function ComponentOfSVectors(svectors::A, component::Int) where {T,A<:AbstractVector{SVector{3,T}}}
    @assert dim <= length(first(svectors))
    return ComponentOfSVectors(component, svectors)
end

Base.getindex(cvectors::ComponentOfSVectors, i::Int) = cvectors.data[i][cvectors.component]

Base.size(cvectors::ComponentOfSVectors) = size(cvectors.data)

function pick(svectors::A, component::Int) where {T,A<:AbstractVector{SVector{3,T}}}
    return ComponentOfSVectors(component, svectors)
end
