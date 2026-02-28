import modal

app = modal.App("kernels")
app.image = (
    modal.Image.debian_slim()
    .pip_install_from_pyproject("./pyproject.toml")
    .add_local_python_source("kernels")
)


@app.function(gpu="B200")
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


@app.local_entrypoint()
def main(name: str):
    match name:
        case "rms_norm":
            run_bench_rms_norm.remote()
        case _:
            raise ValueError(f"Unknown benchmark: {name}")
