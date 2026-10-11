"""Physical head/background verification, independent of helmet status."""

import hashlib
import json
from pathlib import Path

from helmet_rider.helmet.classifier import EfficientNetHelmetClassifier


class EfficientNetHeadValidator:
    def __init__(self, model_path, config, threads=2):
        path = Path(model_path)
        metadata = json.loads(path.with_suffix(".json").read_text())
        if not metadata.get("validated_background_discrimination"):
            raise ValueError("Head validator has not demonstrated background discrimination")
        if metadata.get("class_order") != ["head", "background"]:
            raise ValueError("Head validity checkpoint must declare head/background class order")
        if metadata.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Head validity checkpoint metadata SHA mismatch")
        # Share identical tensor loading and pixel preprocessing, but never use
        # its helmet decision method: these two logits mean head/background.
        self.network = EfficientNetHelmetClassifier(model_path, config, threads)

    def validate_batch(self, crops):
        network = self.network
        scores = [0.0] * len(crops)
        tensors = []
        indices = []
        for index, crop in enumerate(crops):
            tensor = network._preprocess(crop)
            if tensor is not None:
                indices.append(index)
                tensors.append(tensor)
        for start in range(0, len(tensors), 32):
            batch = network.torch.stack(tensors[start : start + 32]).to(network.device)
            with network.torch.inference_mode():
                output = network.torch.softmax(network.model(batch), dim=1)[:, 0].cpu().tolist()
            for index, score in zip(indices[start : start + 32], output, strict=True):
                scores[index] = score
        return scores
