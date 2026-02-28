import modal

app = modal.App("kernels")
app.image = (
    modal.Image.debian_slim()
    .pip_install_from_pyproject("./pyproject.toml")
    .add_local_python_source("kernels")
)


def run_bench_rms_norm():
    import torch
    import torch.nn.functional as F
    from kernels.rms_norm import rms_norm
    from triton.testing import do_bench
    import quack

    M, N = 8192, 131072
    # M, N = 65536, 16384

    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    w = torch.randn(N, device="cuda", dtype=torch.float32)

    y = rms_norm(x, w, eps=1e-5)
    y_ref = F.rms_norm(x, (N,), weight=w, eps=1e-5)
    torch.testing.assert_close(y, y_ref)
    print("RMS Norm test passed!")

    elapsed_time_ms = do_bench(lambda: F.rms_norm(x, (N,), weight=w, eps=1e-5))
    mem_bw_GB_s = 2 * x.numel() * x.element_size() / (elapsed_time_ms * 1e-3) / 1e9
    print(f"PyTorch: {elapsed_time_ms:.2f} ms, {mem_bw_GB_s:.2f} GB/s")

    elapsed_time_ms = do_bench(lambda: rms_norm(x, w, eps=1e-5))
    mem_bw_GB_s = 2 * x.numel() * x.element_size() / (elapsed_time_ms * 1e-3) / 1e9
    print(f"CuTe DSL: {elapsed_time_ms:.2f} ms, {mem_bw_GB_s:.2f} GB/s")

    y_quack = quack.rmsnorm(x, w, eps=1e-5)
    torch.testing.assert_close(y_quack, y_ref)

    elapsed_time_ms = do_bench(lambda: quack.rmsnorm(x, w, eps=1e-5))
    mem_bw_GB_s = 2 * x.numel() * x.element_size() / (elapsed_time_ms * 1e-3) / 1e9
    print(f"Quack: {elapsed_time_ms:.2f} ms, {mem_bw_GB_s:.2f} GB/s")


def run_bench_scalar_mul():
    import torch
    from kernels.scalar_mul import scalar_mul
    from triton.testing import do_bench

    M, N = 8192, 131072

    a = torch.randn(M, N, device="cuda", dtype=torch.float32)
    b = torch.randn(1, device="cuda", dtype=torch.float32)

    c = scalar_mul(a, b)
    c_ref = a * b
    torch.testing.assert_close(c, c_ref)
    print("Scalar multiplication test passed!")

    elapsed_time_ms = do_bench(lambda: a * b)
    mem_bw_GB_s = 2 * a.numel() * a.element_size() / (elapsed_time_ms * 1e-3) / 1e9
    print(f"PyTorch: {elapsed_time_ms:.2f} ms, {mem_bw_GB_s:.2f} GB/s")

    elapsed_time_ms = do_bench(lambda: scalar_mul(a, b))
    mem_bw_GB_s = 2 * a.numel() * a.element_size() / (elapsed_time_ms * 1e-3) / 1e9
    print(f"CuTe DSL: {elapsed_time_ms:.2f} ms, {mem_bw_GB_s:.2f} GB/s")


@app.function(gpu="B200")
def run_bench(name: str):
    print("=" * 40)
    match name:
        case "rms_norm":
            run_bench_rms_norm()
        case "scalar_mul":
            run_bench_scalar_mul()
        case _:
            raise ValueError(f"Unknown benchmark: {name}")


@app.local_entrypoint()
def main(name: str = "all"):
    if name == "all":
        for name in ["rms_norm", "scalar_mul"]:
            run_bench.remote(name)
    else:
        run_bench.remote(name)
