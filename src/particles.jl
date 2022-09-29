struct Particles{T,SV<:Vector{SVector{3,T}}}
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
    new_pos = Vector{SVector{3,T}}(undef,npar)
    new_mom = Vector{SVector{3,T}}(undef,npar)
    for i in 1:npar
        new_pos[i] = SVector{3,T}(pos[:,i])
        new_mom[i] = SVector{3,T}(mom[:,i])
    end
    return Particles(npar, charge, mass, new_pos, new_mom, fill(SVector{3,T}(0,0,0),npar), fill(SVector{3,T}(0,0,0),npar))
end

# type alias for Particles
const Beam{T} = Particles{T}
