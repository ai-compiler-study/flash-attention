from flash3 import hopper_mha
from flash_attn_interface import flash_attn_func


import torch

fn = torch.compile(
    hopper_mha,
    fullgraph=True,
    backend="inductor",
    )
print("compilation of HopperMHA done")

flash_v3_fn = torch.compile(
    flash_attn_func,
    fullgraph=True,
    backend="inductor",
    )
print(f"compilation of {flash_v3_fn.__name__} done")