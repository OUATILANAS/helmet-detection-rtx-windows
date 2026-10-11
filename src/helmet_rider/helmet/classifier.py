"""Helmet classifier interface, safe unknown fallback, and compact PyTorch backend."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from helmet_rider.config import ClassifierConfig
from helmet_rider.detection.detector import _resolve_device
from helmet_rider.detection.types import HelmetPrediction, HelmetStatus
from helmet_rider.helmet.tiny_model import build_tiny_helmet_net


class HelmetClassifier(Protocol):
    def classify(self, head_crop: np.ndarray) -> HelmetPrediction: ...


class UnknownHelmetClassifier:
    """Returns unknown while no trained model is installed."""

    def classify(self, head_crop: np.ndarray) -> HelmetPrediction:
        return HelmetPrediction()


class TorchHelmetClassifier:
    """Tiny 128px classifier; PyTorch is loaded only when weights exist."""

    def __init__(
        self, model_path: str, config: ClassifierConfig | None = None, threads: int = 1
    ) -> None:
        self.config = config or ClassifierConfig()
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Helmet classifier checkpoint not found: {path}")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to load this classifier checkpoint.") from exc
        self.torch = torch
        self.device = torch.device(_resolve_device(self.config.device).replace("0", "cuda"))
        torch.set_num_threads(max(1, int(threads)))
        self.model = build_tiny_helmet_net().to(self.device)
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:  # PyTorch before weights_only support
            checkpoint = torch.load(path, map_location=self.device)
        state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state)
        self.model.eval()

    def classify(self, head_crop: np.ndarray) -> HelmetPrediction:
        if head_crop is None or head_crop.size == 0:
            return HelmetPrediction()
        rgb = cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (self.config.input_size, self.config.input_size),
            interpolation=cv2.INTER_AREA,
        )
        tensor = self.torch.from_numpy(resized.copy()).permute(2, 0, 1).float()
        tensor = tensor.unsqueeze(0).to(self.device) / 255.0
        with self.torch.inference_mode():
            probabilities = self.torch.softmax(self.model(tensor) / self.config.temperature, dim=1)[0]
        helmet_probability = float(probabilities[0].item())
        no_helmet_probability = float(probabilities[1].item())
        confidence = max(helmet_probability, no_helmet_probability)
        if confidence < self.config.min_confidence:
            status = HelmetStatus.UNKNOWN
        elif helmet_probability >= self.config.helmet_positive_threshold:
            status = HelmetStatus.HELMET
        elif no_helmet_probability >= self.config.no_helmet_negative_threshold:
            status = HelmetStatus.NO_HELMET
        else:
            status = HelmetStatus.UNKNOWN
        return HelmetPrediction(
            status, confidence, helmet_probability, no_helmet_probability
        )


class EfficientNetHelmetClassifier:
    """EfficientNet-B0 head-crop fallback trained for helmet/no-helmet crops."""

    def __init__(
        self, model_path: str, config: ClassifierConfig | None = None, threads: int = 1
    ) -> None:
        self.config = config or ClassifierConfig()
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Helmet classifier checkpoint not found: {path}")
        try:
            import torch
            from torchvision import models
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch and torchvision are required for the EfficientNet classifier."
            ) from exc
        self.torch = torch
        self.device = torch.device(_resolve_device(self.config.device).replace("0", "cuda"))
        torch.set_num_threads(max(1, int(threads)))
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = torch.nn.Linear(1280, 2)
        # Checkpoints are tensor-only state dictionaries. Do not unpickle modules.
        state = torch.load(path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def classify(self, head_crop: np.ndarray) -> HelmetPrediction:
        return self.classify_batch([head_crop])[0]

    def classify_batch(self, head_crops: list[np.ndarray]) -> list[HelmetPrediction]:
        if not head_crops:
            return []
        predictions = [HelmetPrediction() for _ in head_crops]
        valid_tensors = []
        valid_indices = []
        for index, head_crop in enumerate(head_crops):
            tensor = self._preprocess(head_crop)
            if tensor is not None:
                valid_indices.append(index)
                valid_tensors.append(tensor)
        if not valid_tensors:
            return predictions
        batch = self.torch.stack(valid_tensors).to(self.device)
        with self.torch.inference_mode():
            probabilities = self.torch.softmax(self.model(batch) / self.config.temperature, dim=1).cpu().tolist()
        for index, (helmet_probability, no_helmet_probability) in zip(
            valid_indices, probabilities, strict=True
        ):
            confidence = max(helmet_probability, no_helmet_probability)
            if confidence < self.config.min_confidence:
                status = HelmetStatus.UNKNOWN
            elif helmet_probability >= self.config.helmet_positive_threshold:
                status = HelmetStatus.HELMET
            elif no_helmet_probability >= self.config.no_helmet_negative_threshold:
                status = HelmetStatus.NO_HELMET
            else:
                status = HelmetStatus.UNKNOWN
            predictions[index] = HelmetPrediction(
                status, confidence, helmet_probability, no_helmet_probability
            )
        return predictions

    def _preprocess(self, head_crop: np.ndarray):
        if head_crop is None or head_crop.size == 0:
            return None
        # Match the aspect-preserving, black-padded training transform.
        rgb = cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        size = self.config.input_size
        scale = size / max(width, height)
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        resized = cv2.resize(
            rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        left = (size - resized_width) // 2
        top = (size - resized_height) // 2
        padded[top : top + resized_height, left : left + resized_width] = resized
        tensor = self.torch.from_numpy(padded.copy()).permute(2, 0, 1).float() / 255.0
        return (tensor - self.mean) / self.std


class ONNXHelmetClassifier:
    """Direct ONNX Runtime classifier for the exported two-logit network."""

    def __init__(
        self, model_path: str, config: ClassifierConfig | None = None, threads: int = 1
    ) -> None:
        self.config = config or ClassifierConfig()
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional ONNX Runtime dependency to load classifier ONNX."
            ) from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        input_spec = self.session.get_inputs()[0]
        self.input_name = input_spec.name
        if any(isinstance(edge, int) and edge != self.config.input_size for edge in input_spec.shape[-2:]):
            raise ValueError("Classifier input_size does not match the ONNX model.")
        metadata_path = Path(model_path).with_suffix(".json")
        if metadata_path.is_file():
            import json
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("backend") != self.config.backend:
                raise ValueError("Classifier backend does not match exported preprocessing metadata.")

    def classify(self, head_crop: np.ndarray) -> HelmetPrediction:
        if head_crop is None or head_crop.size == 0:
            return HelmetPrediction()
        rgb = cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (self.config.input_size, self.config.input_size),
            interpolation=cv2.INTER_AREA,
        )
        if self.config.backend == "efficientnet_b0":
            height, width = rgb.shape[:2]
            size = self.config.input_size
            scale = size / max(width, height)
            rw, rh = max(1, int(width*scale)), max(1, int(height*scale))
            resized = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)
            padded = np.zeros((size, size, 3), dtype=np.uint8)
            left, top = (size-rw)//2, (size-rh)//2
            padded[top:top+rh, left:left+rw] = resized
            tensor = padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            tensor = (tensor - np.array([.485,.456,.406],dtype=np.float32)[None,:,None,None]) / np.array([.229,.224,.225],dtype=np.float32)[None,:,None,None]
        else:
            tensor = resized.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        logits = np.asarray(self.session.run(None, {self.input_name: tensor})[0])[0]
        logits = logits / self.config.temperature
        logits = logits - np.max(logits)
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        helmet_probability = float(probabilities[0])
        no_helmet_probability = float(probabilities[1])
        confidence = max(helmet_probability, no_helmet_probability)
        if confidence < self.config.min_confidence:
            status = HelmetStatus.UNKNOWN
        elif helmet_probability >= self.config.helmet_positive_threshold:
            status = HelmetStatus.HELMET
        elif no_helmet_probability >= self.config.no_helmet_negative_threshold:
            status = HelmetStatus.NO_HELMET
        else:
            status = HelmetStatus.UNKNOWN
        return HelmetPrediction(
            status, confidence, helmet_probability, no_helmet_probability
        )


def create_classifier(config: ClassifierConfig, threads: int = 1) -> HelmetClassifier:
    if not Path(config.model).is_file():
        return UnknownHelmetClassifier()
    if Path(config.model).suffix.lower() == ".onnx":
        return ONNXHelmetClassifier(config.model, config, threads)
    if config.backend == "efficientnet_b0":
        return EfficientNetHelmetClassifier(config.model, config, threads)
    return TorchHelmetClassifier(config.model, config, threads)
