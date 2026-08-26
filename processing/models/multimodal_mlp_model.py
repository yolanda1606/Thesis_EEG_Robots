"""Branched EEG-plus-face regression architecture only."""
from __future__ import annotations

import numpy as np
import torch


EEG_INPUT_DIM = 20
FACE_INPUT_DIM = 10
FUSED_EMBEDDING_DIM = 12


class BranchedMultimodalMLP(torch.nn.Module):
    """Fixed EEG/face branches followed by a linear rating-regression head."""

    def __init__(self) -> None:
        super().__init__()
        self.eeg_first_layer = torch.nn.Linear(EEG_INPUT_DIM, 16)
        self.eeg_second_layer = torch.nn.Linear(16, 8)
        self.face_first_layer = torch.nn.Linear(FACE_INPUT_DIM, 8)
        self.face_second_layer = torch.nn.Linear(8, 4)
        self.fusion_layer = torch.nn.Linear(FUSED_EMBEDDING_DIM, 8)
        self.output_layer = torch.nn.Linear(8, 1)
        self.relu = torch.nn.ReLU()
        self._initialize_deterministically()

    def _initialize_deterministically(self) -> None:
        """Match the project MLP's deterministic normal-weight convention."""
        generator = np.random.default_rng(42)
        with torch.no_grad():
            for layer in (
                self.eeg_first_layer, self.eeg_second_layer, self.face_first_layer,
                self.face_second_layer, self.fusion_layer, self.output_layer,
            ):
                layer.weight.copy_(torch.from_numpy(
                    generator.normal(0.0, 0.1, size=(layer.in_features, layer.out_features)).T
                ).float())
                layer.bias.zero_()

    def split_inputs(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split independently scaled EEG-first rows into their modality blocks."""
        feature_matrix = torch.as_tensor(features, dtype=torch.float32)
        if feature_matrix.ndim != 2:
            raise ValueError("features must be a two-dimensional array")
        expected = EEG_INPUT_DIM + FACE_INPUT_DIM
        if feature_matrix.shape[1] != expected:
            raise ValueError(f"Expected {expected} EEG+face input features, received {feature_matrix.shape[1]}.")
        return feature_matrix[:, :EEG_INPUT_DIM], feature_matrix[:, EEG_INPUT_DIM:]

    def encode_eeg(self, eeg_features: torch.Tensor) -> torch.Tensor:
        """Return the eight-feature EEG branch embedding."""
        return self.relu(self.eeg_second_layer(self.relu(self.eeg_first_layer(eeg_features))))

    def encode_face(self, face_features: torch.Tensor) -> torch.Tensor:
        """Return the four-feature face branch embedding."""
        return self.relu(self.face_second_layer(self.relu(self.face_first_layer(face_features))))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one unconstrained raw rating prediction per EEG-first input row."""
        eeg_features, face_features = self.split_inputs(features)
        fused = torch.cat((self.encode_eeg(eeg_features), self.encode_face(face_features)), dim=1)
        return self.output_layer(self.relu(self.fusion_layer(fused)))


def build_branched_multimodal_mlp() -> BranchedMultimodalMLP:
    """Build the fixed 20->16->8, 10->8->4, 12->8->1 regression MLP."""
    return BranchedMultimodalMLP()
