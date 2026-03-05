# type: ignore
"""Dense matmul (bf16, bf16) -> bf16.

This `(m, k) x (k, n) -> (m, n)` matmul assumes that the first input is
row-major, the second input is column-major, and the output is row-major.

This one uses Triton.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_dense_bf16_triton_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)

    # Swizzle for L2 locality
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        a = tl.load(a_ptrs, mask=k_offs[None, :] < K, other=0.0)
        b = tl.load(b_ptrs, mask=k_offs[:, None] < K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.bfloat16)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def matmul_dense_bf16_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and a.is_contiguous() and a.dtype == torch.bfloat16
    assert b.is_cuda and b.T.is_contiguous() and b.dtype == torch.bfloat16
    assert a.shape[1] == b.shape[0]

    M, K = a.shape
    N = b.shape[1]

    c = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
    GROUP_SIZE_M = 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    matmul_dense_bf16_triton_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    return c
