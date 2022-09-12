struct Particles{T,SV<:AbstractVector{SVector{3,T}}}
    npar::Int
    charge::T
    mass::T
    positions::SV
    momenta::SV
    efields::SV
    bfields::SV
end

function Particles(;pos::Matrix{T}, mom::Matrix{T}, charge=-1.0, mass=1.0) where {T}
    @assert size(pos) == size(mom)
    @assert size(pos)[1] == 3
    npar = size(pos)[2]
    new_pos = reinterpret(reshape, SVector{3,T}, pos)
    new_mom = reinterpret(reshape, SVector{3,T}, mom)
    return Particles(npar, charge, mass, new_pos, new_mom, reinterpret(reshape, SVector{3,T}, zeros(3, npar)), reinterpret(reshape, SVector{3,T}, zeros(3, npar)))
end

# type alias for Particles
const Beam{T} = Particles{T}
