import unittest

from flash3 import hopper_mha

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.library import opcheck
from torch.testing._internal.common_utils import (
    decorateIf,
    instantiate_parametrized_tests,
    parametrize,
)

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


class TestOps(unittest.TestCase):
    def test_hopper_mha_inductor_compatible(self):
        causal = False
        dtype = torch.float16
        device = "cuda"
        batch_size = 1
        seqlen = 512
        dim = 2048
        head_dim = 256

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

        softmax_scale = q.shape[-1] ** (-0.5)
        sample_input = (q, k, v, softmax_scale, causal)
        self.assertTrue(opcheck(torch.ops.hopper_flash3.flash_fwd, sample_input))

    @parametrize("dtype", [torch.float16])
    @parametrize("is_causal", [False, True])
    @parametrize("head_dim", [64, 128, 256])
    @parametrize("compiled", [True, False])
    @parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 1),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),

    ],
)
    def test_hopper_mha_output(
        self,
        seqlen_q: int,
        seqlen_k: int,
        head_dim: int,
        is_causal: bool,
        dtype: torch.dtype,
        compiled: bool,
    ):
        device = "cuda"
        batch_size = 1
        print(dtype)
        # set seed
        torch.random.manual_seed(0)

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
        o, lse = hopper_mha(q, k, v, softmax_scale, is_causal)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            if compiled:
                fn = torch.compile(
                    F.scaled_dot_product_attention,
                    fullgraph=False,
                    backend="inductor",
                )
            else:
                fn = F.scaled_dot_product_attention
            
            q = q.permute(0, 2, 1, 3) # B, H, S, D
            k = k.permute(0, 2, 1, 3) # B, H, S, D
            v = v.permute(0, 2, 1, 3) # B, H, S, D
            o_ref = fn(q, k, v, is_causal=is_causal, scale=softmax_scale)
            o = o.permute(0, 2, 1, 3) # B, S, H, D

        print(f"Pytorch max diff: {(o - o_ref).abs().max().item()}")
        print(f"Pytorch mean diff: {(o - o_ref).abs().mean().item()}")
        self.assertTrue(torch.allclose(o_ref, o, atol=ABS_TOL, rtol=REL_TOL))


if __name__ == "__main__":
    instantiate_parametrized_tests(TestOps)
    unittest.main()
