export nsvector, baryweights, cheb2coord

nsvector(f::F,::Val{2},::Type{T}) where {F,T} = SVector{2,T}(f(1),f(2))
nsvector(f::F,::Val{3},::Type{T}) where {F,T} = SVector{3,T}(f(1),f(2),f(3))
nsvector(f::F,::Val{4},::Type{T}) where {F,T} = SVector{4,T}(f(1),f(2),f(3),f(4))
nsvector(f::F,::Val{5},::Type{T}) where {F,T} = SVector{5,T}(f(1),f(2),f(3),f(4),f(5))
nsvector(f::F,::Val{6},::Type{T}) where {F,T} = SVector{6,T}(f(1),f(2),f(3),f(4),f(5),f(6))
nsvector(f::F,::Val{7},::Type{T}) where {F,T} = SVector{7,T}(f(1),f(2),f(3),f(4),f(5),f(6),f(7))
nsvector(f::F,::Val{8},::Type{T}) where {F,T} = SVector{8,T}(f(1),f(2),f(3),f(4),f(5),f(6),f(7),f(8))
nsvector(f::F,::Val{9},::Type{T}) where {F,T} = SVector{9,T}(f(1),f(2),f(3),f(4),f(5),f(6),f(7),f(8),f(9))
nsvector(f::F,::Val{10},::Type{T}) where {F,T} = SVector{10,T}(f(1),f(2),f(3),f(4),f(5),f(6),f(7),f(8),f(9),f(10))
nsvector(f::F,::Val{11},::Type{T}) where {F,T} = SVector{11,T}(f(1),f(2),f(3),f(4),f(5),f(6),f(7),f(8),f(9),f(10),f(11))

baryweights(::Val{2},::Type{T}) where {T} =   SVector{2,T}(0.5,-1)
baryweights(::Val{3},::Type{T}) where {T} =   SVector{3,T}(0.5,-1,0.5)
baryweights(::Val{4},::Type{T}) where {T} =   SVector{4,T}(0.5,-1,1,-0.5)
baryweights(::Val{5},::Type{T}) where {T} =   SVector{5,T}(0.5,-1,1,-1,0.5)
baryweights(::Val{6},::Type{T}) where {T} =   SVector{6,T}(0.5,-1,1,-1,1,-0.5)
baryweights(::Val{7},::Type{T}) where {T} =   SVector{7,T}(0.5,-1,1,-1,1,-1,0.5)
baryweights(::Val{8},::Type{T}) where {T} =   SVector{8,T}(0.5,-1,1,-1,1,-1,1,-0.5)
baryweights(::Val{9},::Type{T}) where {T} =   SVector{9,T}(0.5,-1,1,-1,1,-1,1,-1,0.5)
baryweights(::Val{10},::Type{T}) where {T} = SVector{10,T}(0.5,-1,1,-1,1,-1,1,-1,1,-0.5)
baryweights(::Val{11},::Type{T}) where {T} = SVector{11,T}(0.5,-1,1,-1,1,-1,1,-1,1,-1,0.5)

function cheb2coord(i::I, a::T, b::T, ::Val{D}) where {I,D,T}
    return a + (cos(pi*(i-1)/D) + 1.0)/2.0 * (b - a)
end
