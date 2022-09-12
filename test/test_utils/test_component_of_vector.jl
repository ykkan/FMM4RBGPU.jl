@testset "ComponentOfSVectors" begin
    @testset "construction" begin
        svectors = [SVector{3,Float64}(0,0,i) for i in 1:10]
        k = 3
        cps = ComponentOfSVectors(k, svectors)
        @test cps[5] == svectors[5][k]
        @test cps[8] == svectors[8][k]
    end

    @testset "pick component" begin
        svectors = [SVector{3,Float64}(0,0,i) for i in 1:10]
        k = 3
        cvectors = pick(svectors, k)
        @test cvectors == [1:10;]
    end
end
