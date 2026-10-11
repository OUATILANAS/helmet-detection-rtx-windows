import pytest
import torch

from helmet_rider.detection.head_union import UnionHeadLogits


def test_union_preserves_confident_evidence_in_either_channel():
    logits = torch.tensor([[[[8., -8., 2.]], [[-8., 8., 2.]]]])
    result = UnionHeadLogits()(logits).sigmoid()
    probs = logits.sigmoid()
    expected = 1 - (1 - probs[:, :1]) * (1 - probs[:, 1:2])
    assert torch.allclose(result, expected, atol=1e-6)
    assert torch.all(result >= probs.max(dim=1, keepdim=True).values)


def test_union_backward_is_finite_and_reaches_both_pretrained_channels():
    logits = torch.tensor([[[[100., -100., 1.]], [[-100., 100., -1.]]]], requires_grad=True)
    output = UnionHeadLogits()(logits)
    torch.nn.functional.binary_cross_entropy_with_logits(output, torch.ones_like(output)).backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 0, 2] != 0
    assert logits.grad[0, 1, 0, 2] != 0


def test_union_rejects_an_incompatible_class_contract():
    with pytest.raises(ValueError, match="two input logits"):
        UnionHeadLogits()(torch.zeros((1, 3, 2, 2)))
