@testset "BBox" begin
    @testset "construction" begin
        bmin = SVector(0.0,0.0,0.0)
        bmax = SVector(1.0,1.0,1.0)
        bbox = BBox(bmin, bmax)
        @test bbox[1] == bmin
        @test bbox[2] == bmax
        @test diameter(bbox) == 0.8660254037844386
    end

    @testset "bbox1 `inside` bbox2" begin
        bmin1 = SVector(0.0,0.0,0.0)
        bmax1 = SVector(1.0,1.0,1.0)
        bmin2 = SVector(0.0,0.0,0.0)
        bmax2 = SVector(0.5,0.9,1.0)
        bmin3 = SVector(0.0,-0.2,0.0)
        bmax3 = SVector(0.5,0.5,0.5)
        bbox1 = BBox(bmin1, bmax1)
        bbox2 = BBox(bmin2, bmax2)
        bbox3 = BBox(bmin3, bmax3)
        @test inside(bbox2, bbox1)
        @test ~ inside(bbox3, bbox1)
    end
end
