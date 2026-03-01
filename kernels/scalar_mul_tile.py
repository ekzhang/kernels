"""cuTile version of `scalar_mul`."""

import torch
import cuda.tile as ct


@ct.kernel
def scalar_mul_tile_kernel(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(0,), shape=(1,))
    result = a_tile * b_tile
    ct.store(c, index=(pid,), tile=result)


def scalar_mul_tile(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and a.is_contiguous() and len(a.shape) == 2
    assert b.is_cuda and b.is_contiguous() and b.numel() == 1

    tile_size = 8192

    out = torch.empty_like(a)
    grid = (ct.cdiv(a.numel(), tile_size),)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        scalar_mul_tile_kernel,
        (a.ravel(), b.ravel(), out.ravel(), tile_size),
    )
    return out
