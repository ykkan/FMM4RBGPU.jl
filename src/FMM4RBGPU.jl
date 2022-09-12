module FMM4RBGPU
using LinearAlgebra
using StaticArrays

export Particles, Beam

include("particles.jl")
include("utils/utils.jl")
include("cluster_tree/cluster_tree.jl")

end
