using ScalarKernelFunctions
using Test

@testset "ScalarKernelFunctions" begin
    x0 = rand(100)
    x1 = rand(100)
    x2 = rand(50)
    K1 = zeros(100, 100)
    K2 = zeros(100, 100)
    K3 = zeros(100, 50)
    K4 = zeros(100, 50)
    v1 = zeros(100)
    v2 = zeros(100)
    @testset "Consistency with KernelFunctions.jl" begin
        @testset "$kernel2" for (kernel1, kernel2) in (
            SEKernel() => ScalarSEKernel(),
            LinearKernel() => ScalarLinearKernel(),
            PeriodicKernel() => ScalarPeriodicKernel(),
            PeriodicKernel(; r = [2.]) => ScalarPeriodicKernel(2.),
            2. * SEKernel() + 3. * LinearKernel() => 2. * ScalarSEKernel() + 3. * ScalarLinearKernel(),
            SEKernel() * PeriodicKernel() => ScalarSEKernel() * ScalarPeriodicKernel()
        )
            @testset for (k1, k2) in (
                (kernel1, kernel2),
                with_lengthscale.((kernel1, kernel2), 2.),
                2 .* (kernel1, kernel2)
            )
                @test k1(1., 4.) ≈ k2(1., 4.)
                @test kernelmatrix(k1, x0) ≈ kernelmatrix(k2, x0)
                @test kernelmatrix(k1, x0, x1) ≈ kernelmatrix(k2, x0, x1)
                @test kernelmatrix(k1, x0, x2) ≈ kernelmatrix(k2, x0, x2)
                @test kernelmatrix(k1, x2, x0) ≈ kernelmatrix(k2, x2, x0)

                @test kernelmatrix_diag(k1, x0) ≈ kernelmatrix_diag(k2, x0)
                @test kernelmatrix_diag(k1, x0, x1) ≈ kernelmatrix_diag(k2, x0, x1)

                kernelmatrix!(K1, k1, x0)
                kernelmatrix!(K2, k2, x0)
                @test K1 ≈ K2

                kernelmatrix!(K1, k1, x0, x1)
                kernelmatrix!(K2, k2, x0, x1)
                @test K1 ≈ K2

                kernelmatrix!(K3, k1, x0, x2)
                kernelmatrix!(K4, k2, x0, x2)
                @test K3 ≈ K4

                kernelmatrix_diag!(v1, k1, x0, x1)
                kernelmatrix_diag!(v2, k2, x0, x1)
                @test v1 ≈ v2
            end
        end
    end
end
