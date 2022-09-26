export MacroParticles

struct MacroParticles{I,T}
    n::I
    gammas::Array{T,4}
    momenta::Array{SVector{3,T},4}
    efields::Array{SVector{3,T},4}
    bfields::Array{SVector{3,T},4}
end

function MacroParticles(clusters::Clusters{I,T}, n::I) where {I,T}
    nc = length(clusters.levels)
    gammas = zeros(T,n+1,n+1,n+1,nc)
    momenta = fill(SVector{3,T}(0,0,0),n+1,n+1,n+1,nc)
    efielfs = fill(SVector{3,T}(0,0,0),n+1,n+1,n+1,nc)
    bfields = fill(SVector{3,T}(0,0,0),n+1,n+1,n+1,nc)
    return MacroParticles(n, gammas, momenta, efielfs, bfields)
end
