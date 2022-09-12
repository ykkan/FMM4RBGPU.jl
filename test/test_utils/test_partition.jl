@testset "partition" begin
    @testset "Lomuto partition" begin
        data = [1, 1, 5, 2, 4, 4, 3,3]
        n = length(data)
        labels = [1:n;]
        partition!(labels, 1, n, data; pvtindex=n)
        @test all(data[labels[1:3]] .< 3)
        @test all(data[labels[4:n]] .>= 3)
    end

    @testset "k-th largest perumation of labels from data" begin
        data = [2, 1, 3, 4, 4, 6]
        n = length(data)

        # k = 3
        labels = [1:n;]
        kpermute!(labels, 1, n, data; k=3)
        kval = 3
        @test all(data[labels[1:3]] .<= kval)
        @test all(data[labels[4:end]] .>= kval)

        # k = 4
        labels = [1:n;]
        kpermute!(labels, 1, n, data; k=4)
        kval = 4
        @test all(data[labels[1:4]] .<= kval)
        @test all(data[labels[5:end]] .>= kval)
    end
end
