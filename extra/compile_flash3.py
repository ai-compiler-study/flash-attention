from flash3 import hopper_mha

import torch

fn = torch.compile(
    hopper_mha,
    fullgraph=True,
    backend="inductor",
    )
print("compilation done")