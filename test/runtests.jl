using ScalarKernelFunctions
using Test

@testset "ScalarKernelFunctions" begin
    x = rand(100)
    y = rand(50)
    M1 = zeros(100, 50)
    M2 = zeros(100, 50)
    v1 = zeros(50)
    v2 = zeros(50)
    @testset "Consistency with KernelFunctions.jl" begin
        @testset "$k2" for (k1, k2) in (
            SEKernel() => ScalarSEKernel(),
            LinearKernel() => ScalarLinearKernel(),
            PeriodicKernel() => ScalarPeriodicKernel(),
            PeriodicKernel(; r = [2.]) => ScalarPeriodicKernel(2.)
        )
            @testset for (kernel1, kernel2) in (
                (k1, k2),
                with_lengthscale.((k1, k2), 2.),
                2 .* (k1, k2)
            )
                @test k1(1., 4.) ≈ k2(1., 4.)
                @test kernelmatrix(k1, x) ≈ kernelmatrix(k2, x)
                @test kernelmatrix(k1, x, y) ≈ kernelmatrix(k2, x, y)
                @test kernelmatrix_diag(k1, x) ≈ kernelmatrix_diag(k2, x)
                @test kernelmatrix_diag(k1, x, y) ≈ kernelmatrix_diag(k2, x, y)

                kernelmatrix!(M1, k1, x, y)
                kernelmatrix!(M2, k2, x, y)
                kernelmatrix_diag!(v1, k1, x, y)
                kernelmatrix_diag!(v2, k2, x, y)
                @test M1 ≈ M2
                @test v1 ≈ v2
            end
        end
    end
end
