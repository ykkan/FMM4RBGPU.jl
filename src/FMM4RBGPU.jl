module FMM4RBGPU
using LinearAlgebra
using StaticArrays
using CUDA

export Particles, Beam

include("particles.jl")
include("utils/utils.jl")
include("cluster_tree/cluster_tree.jl")
include("fmm/fmm.jl")
include("fmm_gpu/fmm_gpu.jl")
include("update_particles_field.jl")


end
