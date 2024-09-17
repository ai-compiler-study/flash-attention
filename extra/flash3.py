from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple

import torch

try:
    from flash_attn_interface import flashattn_hopper_cuda as _C_flashattention3
except ImportError:
    # We end up here is arch is not 90a
    _C_flashattention3 = None

if _C_flashattention3 is not None:
    # returns: out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p
    @torch.library.custom_op(
        "hopper_flash3::flash_fwd", mutates_args=(), device_types=["cuda"]
    )
    def mha_fwd(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        softmax_scale: Optional[float],
        is_causal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor,]:
        (
            out,
            q_padded,
            k_padded,
            v_padded,
            out_padded,
            softmax_lse,
            p,
        ) = _C_flashattention3.fwd(
            query, key, value, None, softmax_scale, None, None, None, is_causal
        )
        return out, softmax_lse
    
    @torch.library.register_fake("hopper_flash3::flash_fwd")
    def mha_fwd_fake(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        softmax_scale: Optional[float],
        is_causal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor,]:
        query_shape = query.shape
        out = query.new_empty(query_shape)
        # Query is (B, M, H, K) or (total_M, H, K)
        # LSE is (B, H, M) or (H, total_M)
        lse_shape = (
            (query_shape[0], query_shape[2], query_shape[1])
        )
        lse = query.new_empty(lse_shape, dtype=torch.float32)
        return out, lse

class HopperMHA(torch.autograd.Function):
    @staticmethod
    def forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        softmax_scale: float,
        is_causal: bool
        ):
        return torch.ops.hopper_flash3.flash_fwd(
                query,
                key,
                value,
                softmax_scale,
                is_causal,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass

hopper_mha = HopperMHA.apply