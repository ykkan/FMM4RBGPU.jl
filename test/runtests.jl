using FMM4RBGPU
using Test
using LinearAlgebra
using StaticArrays
using CUDA

include("test_particles.jl")
include("test_utils/test_utils.jl")
include("test_cluster_tree/test_cluster_tree.jl")
include("test_fmm/test_fmm.jl")
include("test_fmm_gpu/test_fmm_gpu.jl")
