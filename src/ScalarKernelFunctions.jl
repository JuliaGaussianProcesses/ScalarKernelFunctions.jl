module ScalarKernelFunctions

using Reexport
@reexport using KernelFunctions

import KernelFunctions: Kernel
import KernelFunctions: kernelmatrix, kernelmatrix!, kernelmatrix_diag, kernelmatrix_diag!
import KernelFunctions: Transform, IdentityTransform, with_lengthscale

export ScalarKernel, ScalarSEKernel, ScalarLinearKernel, ScalarPeriodicKernel
export ScalarKernelSum, ScalarScaledKernel, with_lengthscale
export TransformedScalarKernel, ScalarScaleTransform

export gpu

gpu(k::Kernel) = k
gpu(t::Transform) = t
gpu(x::Real) = Float32(x)

abstract type ScalarKernel <: Kernel end

function kernelmatrix(
    k::ScalarKernel,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real}
)
    return k.(x, y')
end

function kernelmatrix!(
    K::AbstractMatrix{<:Real},
    k::ScalarKernel,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real}
)
    K .= k.(x, y')
    return K
end

function kernelmatrix_diag(
    k::ScalarKernel,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real}
)
    l = min(length(x), length(y))
    _x = view(x, firstindex(x):firstindex(x)+l-1)
    _y = view(y, firstindex(y):firstindex(y)+l-1)
    return k.(_x, _y)
end

function kernelmatrix_diag!(
    K::AbstractVector{<:Real},
    k::ScalarKernel,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real}
)
    l = min(length(x), length(y))
    _x = view(x, firstindex(x):firstindex(x)+l-1)
    _y = view(y, firstindex(y):firstindex(y)+l-1)
    K .= k.(_x, _y)
    return K
end



struct ScalarSEKernel <: ScalarKernel end
(k::ScalarSEKernel)(x, y) = exp(-abs2(x - y) / 2)

struct ScalarLinearKernel <: ScalarKernel end
(k::ScalarLinearKernel)(x, y) = x * y

struct ScalarPeriodicKernel{T<:Real} <: ScalarKernel
    r::T
end
ScalarPeriodicKernel() = ScalarPeriodicKernel(1.)
(k::ScalarPeriodicKernel)(x, y) = exp(-abs2(sinpi(x - y) / k.r) / 2)
gpu(k::ScalarPeriodicKernel) = ScalarPeriodicKernel(gpu(k.r))



struct ScalarKernelSum{T1<:Kernel, T2<:Kernel} <: ScalarKernel
    k1::T1
    k2::T2
end

ScalarKernelSum(kernels::Tuple) = ScalarKernelSum(kernels...)

function ScalarKernelSum(k1::Kernel, k2::Kernel, k3::Kernel, ks...)
    return ScalarKernelSum(k1, ScalarKernelSum(k2, k3, ks...))
end

(k::ScalarKernelSum)(x, y) = k.k1(x, y) + k.k2(x, y)
gpu(k::ScalarKernelSum) = ScalarKernelSum(gpu(k.k1), gpu(k.k2))
Base.:+(k1::ScalarKernel, k2::ScalarKernel) = ScalarKernelSum(k1, k2)



struct ScalarScaledKernel{Tk<:Kernel,Tσ²<:Real} <: ScalarKernel
    kernel::Tk
    σ²::Tσ²
end

(k::ScalarScaledKernel)(x, y) = k.σ² * k.kernel(x, y)
gpu(k::ScalarScaledKernel) = ScalarScaledKernel(gpu(k.kernel), gpu(k.σ²))
Base.:*(w::Real, k::ScalarKernel) = ScalarScaledKernel(k, w)



struct TransformedScalarKernel{Tk<:Kernel,Tr<:Transform} <: ScalarKernel
    kernel::Tk
    transform::Tr
end

(k::TransformedScalarKernel)(x, y) = k.kernel(k.transform(x), k.transform(y))

function gpu(k::TransformedScalarKernel)
    return TransformedScalarKernel(gpu(k.kernel), gpu(k.transform))
end

Base.:∘(k::ScalarKernel, t::Transform) = TransformedScalarKernel(k, t)

function Base.:∘(k::TransformedScalarKernel, t::Transform)
    return TransformedScalarKernel(k.kernel, k.transform ∘ t)
end

Base.:∘(k::TransformedScalarKernel, ::IdentityTransform) = k



struct ScalarScaleTransform{T<:Real} <: Transform
    s::T
end

(t::ScalarScaleTransform)(x) = t.s * x
Base.:∘(t::ScalarScaleTransform, u::ScalarScaleTransform) = ScalarScaleTransform(t.s * u.s)
gpu(t::ScalarScaleTransform) = ScalarScaleTransform(gpu(t.s))
with_lengthscale(k::ScalarKernel, l::Real) = k ∘ ScalarScaleTransform(inv(l))

end # module ScalarKernelFunctions
