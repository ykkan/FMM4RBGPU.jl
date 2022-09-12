@testset "tree info estimation" begin
    @testset "nclusters" begin
        N, N0 = 4, 2
        @test ncluster(N,N0) == 3
        N, N0 = 11, 3
        @test ncluster(N,N0) == 7
        N, N0 = 18, 3
        @test ncluster(N,N0) == 15
        N, N0 = 2, 4
        @test ncluster(N,N0) == 1
    end
end
