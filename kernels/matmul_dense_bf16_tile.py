"""Dense matmul (bf16, bf16) -> bf16.

This `(m, k) x (k, n) -> (m, n)` matmul assumes that the first input is
row-major, the second input is column-major, and the output is row-major.

This one uses cuTile.
"""

import torch

import cuda.tile as ct


@ct.kernel
def matmul_dense_bf16_tile_kernel(
    A,
    B,
    C,
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    # TODO: swizzle
    bidx, bidy = pid_m, pid_n

    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)

    for k in range(num_tiles_k):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk))
        b = ct.load(B, index=(k, bidy), shape=(tk, tn))
        accumulator = ct.mma(a, b, accumulator)

    ct.store(C, index=(pid_m, pid_n), tile=accumulator.astype(ct.bfloat16))


def matmul_dense_bf16_tile(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K, N = b.shape

    assert a.is_cuda and a.is_contiguous() and a.dtype == torch.bfloat16
    assert b.is_cuda and b.T.is_contiguous() and b.dtype == torch.bfloat16

    c = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)

    grid = (ct.cdiv(M, 128), ct.cdiv(N, 256), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        matmul_dense_bf16_tile_kernel,
        (a, b, c, 128, 256, 64),
    )
    return c
