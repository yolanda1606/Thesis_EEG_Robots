"""Small deterministic 1D CNN architecture for saved EEG trial epochs."""
from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class CNNArchitecture:
    """The only permitted CNN variant settings for this controlled comparison."""
    kernel_size_1: int
    kernel_size_2: int
    dropout: float


CNN_ARCHITECTURES = {
    "baseline": CNNArchitecture(7, 5, 0.0),
    "baseline_dropout": CNNArchitecture(7, 5, 0.3),
    "larger_kernels": CNNArchitecture(15, 9, 0.0),
    "larger_kernels_dropout": CNNArchitecture(15, 9, 0.3),
}


class EEGCNNRegressor(torch.nn.Module):
    """Map ``(batch, channels, time)`` no-ICA epochs to one raw rating."""

    def __init__(self, input_channels: int, kernel_size_1: int = 7, kernel_size_2: int = 5,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if not isinstance(input_channels, int) or isinstance(input_channels, bool) or input_channels <= 0:
            raise ValueError("input_channels must be a positive integer")
        self.input_channels = input_channels
        for name, kernel_size in (("kernel_size_1", kernel_size_1), ("kernel_size_2", kernel_size_2)):
            if not isinstance(kernel_size, int) or kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        self.kernel_size_1, self.kernel_size_2, self.dropout_probability = kernel_size_1, kernel_size_2, dropout
        self.first_convolution = torch.nn.Conv1d(input_channels, 16, kernel_size=kernel_size_1, padding=kernel_size_1 // 2)
        self.max_pool = torch.nn.MaxPool1d(kernel_size=2)
        self.second_convolution = torch.nn.Conv1d(16, 32, kernel_size=kernel_size_2, padding=kernel_size_2 // 2)
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.hidden_layer = torch.nn.Linear(32, 16)
        self.dropout = torch.nn.Identity() if dropout == 0.0 else torch.nn.Dropout(p=dropout)
        self.output_layer = torch.nn.Linear(16, 1)
        self.relu = torch.nn.ReLU()
        self._initialize_deterministically()

    def _initialize_deterministically(self) -> None:
        """Use the project MLP's deterministic normal-weight convention."""
        generator = np.random.default_rng(42)
        with torch.no_grad():
            for layer in (self.first_convolution, self.second_convolution, self.hidden_layer, self.output_layer):
                layer.weight.copy_(torch.from_numpy(
                    generator.normal(0.0, 0.1, size=tuple(layer.weight.shape))
                ).float())
                layer.bias.zero_()

    def forward(self, epochs: torch.Tensor) -> torch.Tensor:
        """Return raw continuous predictions with shape ``(batch, 1)``."""
        epoch_tensor = torch.as_tensor(epochs, dtype=torch.float32)
        if epoch_tensor.ndim != 3:
            raise ValueError("epochs must have shape (batch, channels, time)")
        if epoch_tensor.shape[1] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, received {epoch_tensor.shape[1]}.")
        features = self.relu(self.first_convolution(epoch_tensor))
        features = self.max_pool(features)
        features = self.relu(self.second_convolution(features))
        features = self.global_pool(features).flatten(start_dim=1)
        return self.output_layer(self.dropout(self.relu(self.hidden_layer(features))))


def build_eeg_cnn(input_channels: int, kernel_size_1: int = 7, kernel_size_2: int = 5,
                  dropout: float = 0.0) -> EEGCNNRegressor:
    """Build the fixed Conv(16)->Conv(32)->16->1 EEG rating regressor."""
    return EEGCNNRegressor(input_channels, kernel_size_1, kernel_size_2, dropout)


def build_named_eeg_cnn(input_channels: int, architecture: str = "baseline") -> EEGCNNRegressor:
    """Build one named controlled CNN architecture."""
    try:
        configuration = CNN_ARCHITECTURES[architecture]
    except KeyError as error:
        raise ValueError(f"Unknown CNN architecture: {architecture}") from error
    return build_eeg_cnn(input_channels, **configuration.__dict__)


class LateFusionEEGCNNRegressor(torch.nn.Module):
    """Fuse a raw-epoch EEG embedding with ten normalized face predictors.

    The EEG branch deliberately matches :class:`EEGCNNRegressor`'s baseline
    layers.  Face scaling is performed by the caller using outer-training
    trials only; this model receives already-scaled face tensors.
    """

    face_input_dim = 10

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        if not isinstance(input_channels, int) or isinstance(input_channels, bool) or input_channels <= 0:
            raise ValueError("input_channels must be a positive integer")
        self.input_channels = input_channels
        self.first_convolution = torch.nn.Conv1d(input_channels, 16, kernel_size=7, padding=3)
        self.max_pool = torch.nn.MaxPool1d(kernel_size=2)
        self.second_convolution = torch.nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.eeg_hidden_layer = torch.nn.Linear(32, 16)
        self.face_first_layer = torch.nn.Linear(self.face_input_dim, 8)
        self.face_second_layer = torch.nn.Linear(8, 4)
        self.fusion_layer = torch.nn.Linear(20, 8)
        self.output_layer = torch.nn.Linear(8, 1)
        self.relu = torch.nn.ReLU()
        self._initialize_deterministically()

    def _initialize_deterministically(self) -> None:
        """Match the deterministic initialization convention of the EEG CNN."""
        generator = np.random.default_rng(42)
        with torch.no_grad():
            for layer in (
                self.first_convolution, self.second_convolution, self.eeg_hidden_layer,
                self.face_first_layer, self.face_second_layer, self.fusion_layer, self.output_layer,
            ):
                layer.weight.copy_(torch.from_numpy(
                    generator.normal(0.0, 0.1, size=tuple(layer.weight.shape))
                ).float())
                layer.bias.zero_()

    def encode_eeg(self, epochs: torch.Tensor) -> torch.Tensor:
        """Return the fixed 16-dimensional post-stimulus EEG embedding."""
        epoch_tensor = torch.as_tensor(epochs, dtype=torch.float32)
        if epoch_tensor.ndim != 3:
            raise ValueError("epochs must have shape (batch, channels, time)")
        if epoch_tensor.shape[1] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, received {epoch_tensor.shape[1]}.")
        features = self.relu(self.first_convolution(epoch_tensor))
        features = self.max_pool(features)
        features = self.relu(self.second_convolution(features))
        features = self.global_pool(features).flatten(start_dim=1)
        return self.relu(self.eeg_hidden_layer(features))

    def encode_face(self, face_features: torch.Tensor) -> torch.Tensor:
        """Return the fixed four-dimensional face embedding."""
        face_tensor = torch.as_tensor(face_features, dtype=torch.float32)
        if face_tensor.ndim != 2 or face_tensor.shape[1] != self.face_input_dim:
            received = face_tensor.shape[1] if face_tensor.ndim == 2 else "non-matrix"
            raise ValueError(f"face_features must have shape (batch, {self.face_input_dim}); received {received}.")
        return self.relu(self.face_second_layer(self.relu(self.face_first_layer(face_tensor))))

    def forward(self, epochs: torch.Tensor, face_features: torch.Tensor) -> torch.Tensor:
        """Return one unconstrained continuous rating for each aligned trial."""
        eeg_embedding, face_embedding = self.encode_eeg(epochs), self.encode_face(face_features)
        if eeg_embedding.shape[0] != face_embedding.shape[0]:
            raise ValueError("epochs and face_features must have the same batch size")
        fused = torch.cat((eeg_embedding, face_embedding), dim=1)
        return self.output_layer(self.relu(self.fusion_layer(fused)))


def build_late_fusion_eeg_cnn(input_channels: int) -> LateFusionEEGCNNRegressor:
    """Build the controlled 16-EEG plus 4-face late-fusion CNN regressor."""
    return LateFusionEEGCNNRegressor(input_channels)
