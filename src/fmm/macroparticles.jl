export MacroParticles

struct MacroParticles{I,T}
    n::I
    xcoords::Array{T,2}
    ycoords::Array{T,2}
    zcoords::Array{T,2}
    gammas::Array{T,4}
    momenta::Array{SVector{3,T},4}
    efields::Array{SVector{3,T},4}
    bfields::Array{SVector{3,T},4}
end

function MacroParticles(clusters::Clusters{I,T}, n::I) where {I,T}
    nc = length(clusters.levels)
    xcoords = zeros(T,n+1,nc)
    ycoords = zeros(T,n+1,nc)
    zcoords = zeros(T,n+1,nc)
    gammas = zeros(T,n+1,n+1,n+1,nc)
    momenta = fill(SVector{3,T}(0,0,0),n+1,n+1,n+1,nc)
    efielfs = fill(SVector{3,T}(0,0,0),n+1,n+1,n+1,nc)
    bfields = fill(SVector{3,T}(0,0,0),n+1,n+1,n+1,nc)
    for i in 1:nc
        bbox = clusters.bboxes[i]
        xmin, ymin, zmin = bbox[1]
        xmax, ymax, zmax = bbox[2]
        xcoords[:,i] .= cheb2(n, xmin, xmax)
        ycoords[:,i] .= cheb2(n, ymin, ymax)
        zcoords[:,i] .= cheb2(n, zmin, zmax)
    end
    return MacroParticles(n, xcoords, ycoords, zcoords, gammas, momenta, efielfs, bfields)
end
