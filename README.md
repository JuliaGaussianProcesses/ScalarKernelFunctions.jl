# ScalarKernelFunctions.jl

[![CI](https://github.com/JuliaGaussianProcesses/ScalarKernelFunctions.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaGaussianProcesses/ScalarKernelFunctions.jl/actions/workflows/CI.yml)
[![Codecov](https://codecov.io/gh/JuliaGaussianProcesses/ScalarKernelFunctions.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaGaussianProcesses/ScalarKernelFunctions.jl)

This package implements kernel functions that are optimized for one-dimensional input and
that are GPU-compatible by default.

## Usage
This package re-exports KernelFunctions.jl and uses the same interface. New kernels have the
`Scalar` prefix, but have the same behavior on scalar input:
```julia
using ScalarKernelFunctions

k1 = ScalarSEKernel() # from this package
k2 = SEKernel() # from KernelFunctions.jl
x = rand(100)
kernelmatrix(k1, x) ≈ kernelmatrix(k2, x) # true
```

### GPU
Easy:
```julia
using CUDA
x = CUDA.rand(100)
k = ScalarPeriodicKernel() |> gpu # ScalarPeriodicKernel{Float32}(1.0f0)
kernelmatrix(k, x) # returns 100×100 CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}
```
Omitting the `gpu` function will use `Float64`, which still works, but is much slower.

## Supported Kernels
- [x] `ScalarSEKernel`
- [x] `ScalarLinearKernel`
- [x] `ScalarPeriodicKernel`
- [x] `ScalarKernelSum`
- [x] `ScalarTransformedKernel`
- [x] `ScalarScaledKernel`
<!-- - [ ] `ScalarConstantKernel`
- [ ] `WhiteKernel`
- [ ] `EyeKernel`
- [ ] `ZeroKernel`
- [ ] `WienerKernel`
- [ ] `CosineKernel`
- [ ] `GaussianKernel`
- [ ] `LaplacianKernel`
- [ ] `ExponentialKernel`
- [ ] `GammaExponentialKernel`
- [ ] `ExponentiatedKernel`
- [ ] `FBMKernel`
- [ ] `MaternKernel`
- [ ] `Matern12Kernel`
- [ ] `Matern32Kernel`
- [ ] `Matern52Kernel`
- [ ] `PolynomialKernel`
- [ ] `RationalKernel`
- [ ] `RationalQuadraticKernel`
- [ ] `GammaRationalKernel`
- [ ] `PiecewisePolynomialKernel`
- [ ] `NeuralNetworkKernel`
- [ ] `KernelProduct`
- [ ] `KernelTensorProduct`
- [ ] `NormalizedKernel`
- [ ] `GibbsKernel` -->
