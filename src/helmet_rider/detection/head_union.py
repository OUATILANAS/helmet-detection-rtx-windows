"""Preserve learned helmet/bare-head evidence when learning physical heads."""

import torch
from torch import nn


class UnionHeadLogits(nn.Module):
    """Two logits -> one whose sigmoid is 1-(1-p_helmet)*(1-p_bare).

    This preserves both existing detection channels; averaging their weights
    can cancel a confident helmet against a confident bare-head negative.
    The union is an initialization rule, not a calibration claim.
    """

    def forward(self, logits):
        if logits.shape[1] != 2:
            raise ValueError("Physical-head union requires exactly two input logits")
        return torch.logsumexp(
            torch.cat((logits, logits[:, :1] + logits[:, 1:2]), dim=1),
            dim=1, keepdim=True,
        )


def preserve_head_union(model, source):
    """Install intact pretrained classification branches in a one-class head."""
    import copy

    target, original = model.model[-1], source.model[-1]
    if target.nc != 1 or original.nc != 2 or len(target.cv3) != len(original.cv3):
        raise ValueError("Unsupported detection-head contract")
    target.cv3 = nn.ModuleList([
        nn.Sequential(copy.deepcopy(branch), UnionHeadLogits())
        for branch in original.cv3
    ])
    model.names = {0: "physical_head"}
    return model
