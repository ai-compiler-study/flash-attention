from flash3 import hopper_mha

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.library import opcheck

import code

ABS_TOL = 5e-3
REL_TOL = 1e-1

def print_diffs(out, out_ref):
    out_1d = out.flatten()
    out_ref_1d = out_ref.flatten()
    for idx, (e_o, e_o_ref) in enumerate(zip(out_1d, out_ref_1d)):
        diff = e_o - e_o_ref
        abs_diff = abs(diff)
        abs_ref = abs(e_o_ref + 1e-5)
        relative_diff = abs_diff / abs_ref
        if abs_diff > ABS_TOL or relative_diff > REL_TOL:
            print(f"==== diff ==== {idx}, test: {e_o}, ref: {e_o_ref}")

dtype = torch.float16
device = "cuda"
batch_size = 1
seqlen = 128
dim = 128
head_dim = 64
compiled = False

n_heads = dim // head_dim
n_heads_kv = n_heads
q = torch.randn(
    batch_size,
    seqlen,
    n_heads,
    head_dim,
    device=device,
    dtype=dtype,
    requires_grad=False,
)
k = torch.randn(
    batch_size,
    seqlen,
    n_heads_kv,
    head_dim,
    device=device,
    dtype=dtype,
    requires_grad=False,
)
v = torch.randn(
    batch_size,
    seqlen,
    n_heads_kv,
    head_dim,
    device=device,
    dtype=dtype,
    requires_grad=False,
)

softmax_scale = q.shape[-1] ** -0.5  # Typically 1 / sqrt(head_dim)
ref_o = hopper_mha(q, k, v, softmax_scale, True)[0]

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    if compiled:
        fn = torch.compile(
            F.scaled_dot_product_attention,
            fullgraph=True,
            backend="inductor",
        )
    else:
        fn = F.scaled_dot_product_attention
    o = fn(q, k, v, is_causal=True, scale=softmax_scale)

# Slicing indices
batch_idx = 0
seq_indices = slice(0, 5)       # First 5 sequence positions
head_indices = 0                # First head
head_dim_indices = slice(0, 8)  # First 8 dimensions of the head

# Slice the tensors
ref_subset = ref_o[batch_idx, seq_indices, head_indices, head_dim_indices]
o_subset = o[batch_idx, seq_indices, head_indices, head_dim_indices]
code.interact(local=locals())
print(o_subset)
print(ref_subset)
print(torch.allclose(ref_o, o, atol=2e-06))
#print_diffs(o, ref_o)