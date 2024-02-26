# ScalarKernelFunctions.jl

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![CI](https://github.com/JuliaGaussianProcesses/ScalarKernelFunctions.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaGaussianProcesses/ScalarKernelFunctions.jl/actions/workflows/CI.yml)
[![Codecov](https://codecov.io/gh/JuliaGaussianProcesses/ScalarKernelFunctions.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaGaussianProcesses/ScalarKernelFunctions.jl/tree/master)

This package implements kernel functions that are optimized for one-dimensional input and
that are GPU-compatible by default.

## Usage
This package expands the KernelFunctions.jl package (which is automatically loaded and
reexported) by a new abstract type `ScalarKernel`.
Its subtypes include a couple of base kernels with the `Scalar` prefix.
On 1-dimensional inputs, they give exactly the same output as their KernelFunctions.jl
counterparts, e.g.:
```julia
using ScalarKernelFunctions

k1 = ScalarSEKernel() # from this package
k2 = SEKernel() # from KernelFunctions.jl
x = rand(100)
kernelmatrix(k1, x) ≈ kernelmatrix(k2, x) # true
```
When combining subtypes of `ScalarKernel` using `+`, `with_lengthscale`, another subtype of
`ScalarKernel` will be produced, which means that specialized implementations will be used
for the composite kernel as well.
Mixing specialized and "normal" kernels will also work, but will no longer use the
specialized implementation.

Specializing on 1d input allows to achieve lower allocation counts and faster evaluation,
especially combined with AD packages such as Zygote or Enzyme. Parameter fields are also
scalar, which saves allocations with repeated construction of the kernel, e.g. when kernel
parameters are being optimized.

### GPU-compatibility by default
The kernels in this package are implemented using broadcast, which allows them to work on
the GPU by default. We also export a `gpu` function, which converts any kernel to use
`Float32` parameters (where needed), and calling `kernelmatrix` will preserve `Float32` to
be most efficient on GPUs. For example, this is how we use a kernel on `CuArray`s:
```julia
using CUDA
x = CUDA.rand(100)
k = ScalarPeriodicKernel() |> gpu # ScalarPeriodicKernel{Float32}(1.0f0)
kernelmatrix(k, x) # returns 100×100 CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}
```
Omitting the `gpu` conversion will of course also work, but will be quite a bit slower.

## Supported kernels
### Base kernels
- [x] `ScalarSEKernel`
- [x] `ScalarLinearKernel`
- [x] `ScalarPeriodicKernel`
- [x] `ScalarMatern12Kernel` === `ScalarExponentialKernel`
- [x] `ScalarMatern32Kernel`
- [x] `ScalarMatern52Kernel`
- [ ] `ScalarMaternKernel

### Composite kernels
- [x] `ScalarKernelSum`, when doing `k1 + k2`, where `k1` and `k2` are `ScalarKernel`s
- [ ] `ScalarKernelProduct`
- [x] `ScalarTransformedKernel`, when doing `k ∘ t`, where `k` is a `ScalarKernel` and `t` is a `Transform`
- [x] `ScalarScaledKernel`, when doing `a * k`, where `k` is a `ScalarKernel` and `a` is a `Real`

### Transforms
- [x] `ScalarScaleTransform`
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
- [ ] `PolynomialKernel`
- [ ] `RationalKernel`
- [ ] `RationalQuadraticKernel`
- [ ] `GammaRationalKernel`
- [ ] `PiecewisePolynomialKernel`
- [ ] `NeuralNetworkKernel`
- [ ] `KernelTensorProduct`
- [ ] `NormalizedKernel`
- [ ] `GibbsKernel` -->
