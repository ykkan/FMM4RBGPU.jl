@testset verbose=true "FMMGPU" begin
    include("test_gpu_upwardpass.jl")
    include("test_gpu_interaction.jl")
end
