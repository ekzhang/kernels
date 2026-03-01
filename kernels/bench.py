import modal

app = modal.App("kernels")
app.image = (
    modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu24.04", add_python="3.13")
    .pip_install_from_pyproject("./pyproject.toml")
    .add_local_python_source("kernels")
)


def run_bench_rms_norm():
    import torch
    import torch.nn.functional as F
    from kernels.rms_norm import rms_norm
    from triton.testing import do_bench

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

    if torch.cuda.get_device_capability() >= (9, 0):
        import quack  # Needs at least Hopper to run the quack.rmsnorm().

        y_quack = quack.rmsnorm(x, w, eps=1e-5)
        torch.testing.assert_close(y_quack, y_ref)

        elapsed_time_ms = do_bench(lambda: quack.rmsnorm(x, w, eps=1e-5))
        mem_bw_GB_s = 2 * x.numel() * x.element_size() / (elapsed_time_ms * 1e-3) / 1e9
        print(f"Quack: {elapsed_time_ms:.2f} ms, {mem_bw_GB_s:.2f} GB/s")


def run_bench_scalar_mul():
    import torch
    from kernels.scalar_mul import scalar_mul
    from kernels.scalar_mul_tile import scalar_mul_tile
    from triton.testing import do_bench

    M, N = 8192, 131072

    a = torch.randn(M, N, device="cuda", dtype=torch.float32)
    b = torch.randn(1, device="cuda", dtype=torch.float32)

    c = scalar_mul(a, b)
    c_ref = a * b
    torch.testing.assert_close(c, c_ref)
    print("Scalar multiplication test passed!")

    c = scalar_mul_tile(a, b)
    torch.testing.assert_close(c, c_ref)
    print("Scalar multiplication (cuTile) test passed!")

    elapsed_time_ms = do_bench(lambda: a * b)
    mem_bw_GB_s = 2 * a.numel() * a.element_size() / (elapsed_time_ms * 1e-3) / 1e9
    print(f"PyTorch: {elapsed_time_ms:.2f} ms, {mem_bw_GB_s:.2f} GB/s")

    elapsed_time_ms = do_bench(lambda: scalar_mul(a, b))
    mem_bw_GB_s = 2 * a.numel() * a.element_size() / (elapsed_time_ms * 1e-3) / 1e9
    print(f"CuTe DSL: {elapsed_time_ms:.2f} ms, {mem_bw_GB_s:.2f} GB/s")

    if torch.cuda.get_device_capability() >= (10, 0):  # Blackwell
        elapsed_time_ms = do_bench(lambda: scalar_mul_tile(a, b))
        mem_bw_GB_s = 2 * a.numel() * a.element_size() / (elapsed_time_ms * 1e-3) / 1e9
        print(f"cuTile: {elapsed_time_ms:.2f} ms, {mem_bw_GB_s:.2f} GB/s")


def run_bench_matmul_dense_bf16():
    import torch
    from kernels.matmul_dense_bf16 import matmul_dense_bf16
    from kernels.matmul_dense_bf16_tile import matmul_dense_bf16_tile
    from triton.testing import do_bench

    if torch.cuda.get_device_capability() >= (10, 0):  # Blackwell tcgen05
        M, K, N = 8192, 8192, 8192

        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).T

        c = matmul_dense_bf16(a, b)
        c_ref = a @ b
        torch.testing.assert_close(c, c_ref)
        print("Matmul (dense BF16) test passed!")

        c = matmul_dense_bf16_tile(a, b)
        torch.testing.assert_close(c, c_ref)
        print("Matmul (dense BF16, cuTile) test passed!")

        elapsed_time_ms = do_bench(lambda: a @ b)
        tflops = 2 * M * K * N / (elapsed_time_ms * 1e-3) / 1e12
        print(f"PyTorch: {elapsed_time_ms:.2f} ms, {tflops:.2f} TFLOPS")

        elapsed_time_ms = do_bench(lambda: matmul_dense_bf16(a, b))
        tflops = 2 * M * K * N / (elapsed_time_ms * 1e-3) / 1e12
        print(f"CuTe DSL: {elapsed_time_ms:.2f} ms, {tflops:.2f} TFLOPS")

        elapsed_time_ms = do_bench(lambda: matmul_dense_bf16_tile(a, b))
        tflops = 2 * M * K * N / (elapsed_time_ms * 1e-3) / 1e12
        print(f"cuTile: {elapsed_time_ms:.2f} ms, {tflops:.2f} TFLOPS")


@app.cls()
class BenchmarkRunner:
    @modal.method()
    def run_bench(self, name: str):
        print("-" * 24 + f"[ {name} ]" + "-" * 24)

        match name:
            case "rms_norm":
                run_bench_rms_norm()
            case "scalar_mul":
                run_bench_scalar_mul()
            case "matmul_dense_bf16":
                run_bench_matmul_dense_bf16()
            case _:
                raise ValueError(f"Unknown benchmark: {name}")


@app.local_entrypoint()
def main(name: str = "all", gpu: str = "B200"):
    if gpu == "all":
        gpus = ("B200", "H200", "A100")
    else:
        gpus = (gpu,)

    for gpu in gpus:
        print(f"=> Running benchmark on {gpu}")
        Runner = BenchmarkRunner.with_options(gpu=gpu)
        if name == "all":
            for benchmark in ("rms_norm", "scalar_mul", "matmul_dense_bf16"):
                Runner().run_bench.remote(benchmark)
        else:
            Runner().run_bench.remote(name)
