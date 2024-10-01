import pytest
import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import (
    decorateIf,
    instantiate_parametrized_tests,
    parametrize,
)

import triton
from triton.ops import attention as attention_triton

torch.set_default_device("cuda")

ABS_TOL = 5e-3
REL_TOL = 1e-1


def benchmark_torch_function(iters, f, *args, **kwargs):
    f(*args, **kwargs)
    f(*args, **kwargs)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        f(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    # elapsed_time has a resolution of 0.5 microseconds:
    # but returns milliseconds, so we need to multiply it to increase resolution
    return start_event.elapsed_time(end_event) * 1000 / iters


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("n_heads", [6, 8, 12, 16])
@pytest.mark.parametrize("seq_len", [1, 128, 256, 512, 1024])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_op(
    batch_size: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
):
    torch.manual_seed(20)
    q = (
        torch.empty(
            (batch_size, n_heads, seq_len, head_dim), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.1, std=0.2)
        .requires_grad_()
    )
    k = (
        torch.empty(
            (batch_size, n_heads, seq_len, head_dim), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.4, std=0.2)
        .requires_grad_()
    )
    v = (
        torch.empty(
            (batch_size, n_heads, seq_len, head_dim), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.3, std=0.2)
        .requires_grad_()
    )
    softmax_scale = 0.2
    # reference implementation
    M = torch.tril(torch.ones((seq_len, seq_len), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * softmax_scale
    for z in range(batch_size):
        for h in range(n_heads):
            p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    # triton implementation
    tri_out = attention_triton(q, k, v, True, softmax_scale)
    assert torch.allclose(ref_out, tri_out, atol=ABS_TOL, rtol=REL_TOL)


try:
    import xformers.ops
    import xformers.ops.fmha as fmha

    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

batch_size = 1
n_heads = 32
seq_len = [256, 512]
head_dim = 128
configs = triton.testing.Benchmark(
    x_names=["seq_len"],
    x_vals=seq_len,
    line_arg="provider",
    line_vals=["triton"] + (["xformers"] if HAS_FLASH else []) + ["torch"],
    line_names=["Triton"] + (["Flash v3"] if HAS_FLASH else []) + ["Torch SDPA"],
    styles=[("red", "-"), ('blue', '-'), ("orange", "-")],
    ylabel="ms",
    plot_name=f"fused-attention-batch{batch_size}-n_head{n_heads}-head_dim{head_dim}-fwd",
    args={
        "n_heads": n_heads,
        "batch_size": batch_size,
        "head_dim": head_dim,
        "dtype": torch.float16,
        "mode": "fwd",
    },
)


@triton.testing.perf_report(configs)
def bench_flash_attention(
    batch_size: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
    mode: str,
    provider: str,
    dtype=torch.bfloat16,
    device="cuda",
):
    assert mode == "fwd"
    torch.random.manual_seed(0)
    warmup = 3
    iters = 10

    q = torch.randn(
        batch_size,
        seq_len,
        n_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=False,
    )
    k = torch.randn(
        batch_size,
        seq_len,
        n_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=False,
    )
    v = torch.randn(
        batch_size,
        seq_len,
        n_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=False,
    )
    softmax_scale = q.shape[-1] ** (-0.5)
    is_causal = True

    if provider == "triton":
        fn = lambda: attention_triton(q, k, v, is_causal, softmax_scale)  # noqa: E731
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=iters)
        return ms
    if provider == "torch":
        fn_torch_native = torch.compile(
            F.scaled_dot_product_attention,
            fullgraph=False,
            backend="inductor",
        )
        fn = lambda: fn_torch_native(  # noqa: E731
            q, k, v, is_causal=is_causal, scale=softmax_scale
        )
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=iters)
        return ms
    if provider == "xformers":
        flash3 = torch.compile(
            xformers.ops.fmha.flash3.FwOp,
            fullgraph=True,
            backend="inductor",
        )
        fn = lambda: fmha.memory_efficient_attention_forward(  # noqa: E731
            q,
            k,
            v,
            scale=softmax_scale,
            op=flash3,
        ) 
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=iters)
        return ms


bench_flash_attention.run(save_path="./output", print_data=True)