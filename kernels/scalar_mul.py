"""Multiply a tensor by a scalar.

This is something PyTorch 2.10 is bad at on Blackwell. We get like 3.3 TB/s
memory bandwidth, whereas the CuTe DSL kernel is 6.7 TB/s.
"""

import torch
import cutlass
import cutlass.cute as cute


@cute.jit
def next_power_of_2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@cute.kernel
def scalar_mul_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    gdim, _, _ = cute.arch.grid_dim()

    thread_idx = bidx * bdim + tidx
    num_threads = gdim * bdim

    b_val = gB[0]
    m, n = gA.shape[1], gA.shape[2]  # thread-domain
    for i in range(thread_idx, m * n, num_threads):
        ni = i % n
        mi = i // n
        a_val = gA[None, mi, ni].load()
        gC[None, mi, ni] = a_val * b_val


@cute.jit
def scalar_mul_(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    vectorize: cutlass.Constexpr[int] = 4,
    num_threads_per_block: cutlass.Constexpr[int] = 128,
    num_elems_per_thread: cutlass.Constexpr[int] = 16,
):
    m, n = mA.shape
    grid_x = next_power_of_2(cdiv(m * n, num_threads_per_block * num_elems_per_thread))

    gA = cute.tiled_divide(mA, (1, vectorize))
    gB = mB
    gC = cute.tiled_divide(mC, (1, vectorize))
    # print(mC.layout)
    # print(gA.layout)

    kernel = scalar_mul_kernel(gA, gB, gC)
    kernel.launch(
        grid=(grid_x, 1, 1),
        block=(num_threads_per_block, 1, 1),
    )


def scalar_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and a.is_contiguous() and len(a.shape) == 2
    assert b.is_cuda and b.is_contiguous() and b.numel() == 1

    out = torch.empty_like(a)

    cache_key = (a.dtype, b.dtype)
    if cache_key in scalar_mul._compile_cache:
        compiled_kernel = scalar_mul._compile_cache[cache_key]
    else:
        mA = cute.runtime.from_dlpack(
            a, assumed_align=16, enable_tvm_ffi=True
        ).mark_layout_dynamic()
        mB = cute.runtime.from_dlpack(b, assumed_align=16, enable_tvm_ffi=True)
        mC = cute.runtime.from_dlpack(
            out, assumed_align=16, enable_tvm_ffi=True
        ).mark_layout_dynamic()
        compiled_kernel = cute.compile(
            scalar_mul_,
            mA,
            mB,
            mC,
            options="--enable-tvm-ffi",
        )
        scalar_mul._compile_cache[cache_key] = compiled_kernel

    compiled_kernel(a, b, out)
    return out


scalar_mul._compile_cache = {}
