include("clusters.jl")
include("treeinfo.jl")

export CusterTree

struct ClusterTree{I<:Integer,T}
    particles::Particles{T}
    parindices::Vector{I}
    clusters::Clusters{I,T}
end

function ClusterTree(particles::Particles{T}; N0, stretch) where {T}
    N = particles.npar
    parindices = [1:N;]
    clusters = Clusters(ncluster(N,N0))
    fillclusters!(clusters, 1, particles, parindices, (1,N), 0; N0=N0, stretch=stretch)
    return ClusterTree(particles, parindices, clusters)
end
