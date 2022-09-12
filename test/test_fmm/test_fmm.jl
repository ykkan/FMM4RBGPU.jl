@testset verbose=true "FMM" begin
    include("test_macroparticles.jl")
    include("test_upwardpass.jl")
    include("test_interaction_lists.jl")
    include("test_interaction.jl")
end
