@testset "cheb2" begin
    @testset "generate chevyschev points of 2nd kind" begin
        n = 5
        @test cheb2(n) == [cos(pi*i/n) for i=0:n]
        @test cheb2(n,0.0,1.0) == ([cos(pi*i/n) for i=0:n] .+ 1.0)/2.0
    end
end
