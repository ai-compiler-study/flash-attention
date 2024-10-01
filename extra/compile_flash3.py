import torch
import torch.nn.functional as F

import xformers.ops
import xformers.ops.fmha as fmha

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


dtype = torch.bfloat16
device = "cuda"
batch_size = 1
print(dtype)
torch.random.manual_seed(0)

head_dim = 256
seqlen_q = 512
seqlen_k = 512
n_heads = 6

q = torch.randn(
    batch_size,
    seqlen_q,
    n_heads,
    head_dim,
    device=device,
    dtype=dtype,
    requires_grad=False,
)
k = torch.randn(
    batch_size,
    seqlen_k,
    n_heads,
    head_dim,
    device=device,
    dtype=dtype,
    requires_grad=False,
)
v = torch.randn(
    batch_size,
    seqlen_k,
    n_heads,
    head_dim,
    device=device,
    dtype=dtype,
    requires_grad=False,
)

softmax_scale = q.shape[-1] ** (-0.5)
is_causal = True

fn = torch.compile(
    xformers.ops.fmha.flash3.FwOp,
    fullgraph=True,
    backend="inductor",
)
print("compilation done")

forward_time_flash3 = benchmark_torch_function(
    10,
    fmha.memory_efficient_attention_partial,
    q,
    k,
    v,
    scale=softmax_scale,
    op=fn,
)
print(f"avg xformers.ops.fmha.flash3.FwOp fwd time: {forward_time_flash3 / 1e3:.5f} ms")

fn_torch_native = torch.compile(
    F.scaled_dot_product_attention,
    fullgraph=False,
    backend="inductor",
)
forward_time_sdpa = benchmark_torch_function(
    10,
    fn_torch_native,
    q,
    k,
    v,
    is_causal=is_causal,
    scale=softmax_scale,
)
print(f"avg Torch sdpa fwd time: {forward_time_sdpa / 1e3:.5f} ms")

### Numerical correctness check
"""
q = q.permute(0, 2, 1, 3) # B, H, S, D
k = k.permute(0, 2, 1, 3) # B, H, S, D
v = v.permute(0, 2, 1, 3) # B, H, S, D
o_ref = fn_torch_native(q, k, v, is_causal=is_causal, scale=softmax_scale)
o, _ = fmha.memory_efficient_attention_partial(q, k, v, scale=softmax_scale, op=fn)
o = o.permute(0, 2, 1, 3) # B, S, H, D


print(f"Pytorch max diff: {(o - o_ref).abs().max().item()}")
print(f"Pytorch mean diff: {(o - o_ref).abs().mean().item()}")
assert torch.allclose(o_ref, o, atol=ABS_TOL, rtol=REL_TOL)
"""