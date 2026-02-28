import torch
import cutlass
import cutlass.cute as cute


@cute.kernel
def rms_norm_kernel(
    gX: cute.Tensor,
    gW: cute.Tensor,
    gY: cute.tensor,
    threads_per_block: cutlass.Constexpr,
    eps: cutlass.Constexpr,
    version: cutlass.Constexpr[int],
):
    allocator = cutlass.utils.SmemAllocator()
    layout = cute.make_layout((threads_per_block))
    scalar_layout = cute.make_layout((1))

    sdata = allocator.allocate_tensor(
        cutlass.Float32, layout=layout, byte_alignment=16, swizzle=None
    )
    squared_reduce = allocator.allocate_tensor(cutlass.Float32, layout=scalar_layout)

    tidx, _, _ = cute.arch.thread_idx()
    widx = cute.arch.warp_idx()
    bidx, _, _ = cute.arch.block_idx()

    hidden_dim = gX.shape[0][1] * gX.shape[1][1]
    logical_size = gX.shape[1][1]

    block_sum = cute.full((), 0.0, dtype=cute.Float32)
    for i in range(tidx, logical_size, threads_per_block, unroll_full=True):
        x_ = cute.flatten(gX[None, (bidx, i)]).load()
        s = (x_ * x_).reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0)
        block_sum += s

    if cutlass.const_expr(version == 1):
        assert threads_per_block == 128
        sdata[tidx] = block_sum[0]
        cute.arch.sync_threads()

        if tidx < 64:
            sdata[tidx] += sdata[tidx + 64]
        cute.arch.sync_threads()

        if tidx < 32:
            sdata[tidx] += sdata[tidx + 32]
            res = cute.arch.warp_reduction_sum(sdata[tidx], threads_in_group=32)
            if tidx == 0:
                squared_reduce[0] = cute.math.rsqrt(
                    res / hidden_dim + eps, fastmath=True
                )

        cute.arch.sync_threads()

    elif cutlass.const_expr(version == 2):
        assert threads_per_block == 128
        warps_per_block = threads_per_block // 32

        # Initial warp reduction sum, followed by election.
        res = cute.arch.warp_reduction_sum(block_sum[0], threads_in_group=32)

        per_warp_sum = allocator.allocate_tensor(
            cute.Float32, layout=cute.make_layout((warps_per_block))
        )
        with cute.arch.elect_one():
            per_warp_sum[widx] = res
        cute.arch.sync_threads()

        # warp0 loads the per-warp sums into lanes 0..warps_per_block-1
        if widx == 0:
            val = 0.0
            if tidx < warps_per_block:
                val = per_warp_sum[tidx]
            total = cute.arch.warp_reduction_sum(val, threads_in_group=warps_per_block)
            if tidx == 0:
                squared_reduce[0] = cute.math.rsqrt(
                    total / hidden_dim + eps, fastmath=True
                )
        cute.arch.sync_threads()

    else:
        # Incorrect version, just for speed comparison.
        if tidx == 0:
            squared_reduce[0] = block_sum[0]

    rms = squared_reduce[0]
    for i in range(tidx, logical_size, threads_per_block, unroll_full=True):
        x_chunk = cute.flatten(gX[None, (bidx, i)])
        w_chunk = cute.flatten(gW[None, (i,)])
        gY[None, (bidx, i)] = (x_chunk.load() * rms) * w_chunk.load().reshape(
            (1, cute.size(w_chunk))
        )


@cute.jit
def rms_norm_(
    mX: cute.Tensor,
    mW: cute.Tensor,
    mY: cute.Tensor,
    eps: cutlass.Constexpr,
    version: cutlass.Constexpr[int],
):
    threads_per_block = 128

    # Vectorize memory load/store with zipped divide.
    gX = cute.zipped_divide(mX, (1, 8))
    gW = cute.zipped_divide(mW, (8,))
    gY = cute.zipped_divide(mY, (1, 8))

    num_tokens, logical_size = gX.shape[1]
    # assert gW.shape[1] == (logical_size,)
    # assert gY.shape[1] == (num_tokens, logical_size)

    rms_norm_kernel(gX, gW, gY, threads_per_block, eps, version).launch(
        grid=(num_tokens, 1, 1), block=(threads_per_block, 1, 1)
    )


def rms_norm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-5,
    *,
    version: int = 1,
) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous() and x.dtype == torch.float32
    assert w.is_cuda and w.is_contiguous() and w.dtype == torch.float32

    y = torch.empty_like(x)

    cache_key = (y.shape[1], eps, version)
    if cache_key in rms_norm._compile_cache:
        compiled_kernel = rms_norm._compile_cache[cache_key]
    else:
        mX = cute.runtime.from_dlpack(
            x, assumed_align=16, enable_tvm_ffi=True
        ).mark_compact_shape_dynamic(0)
        mW = cute.runtime.from_dlpack(w, assumed_align=16, enable_tvm_ffi=True)
        mY = cute.runtime.from_dlpack(
            y, assumed_align=16, enable_tvm_ffi=True
        ).mark_compact_shape_dynamic(0)
        compiled_kernel = cute.compile(
            rms_norm_,
            mX,
            mW,
            mY,
            eps=eps,
            version=version,
            options="--enable-tvm-ffi",
        )
        rms_norm._compile_cache[cache_key] = compiled_kernel

    compiled_kernel(x, w, y)
    return y


rms_norm._compile_cache = {}
