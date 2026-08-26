"""MLP definition, optimization primitives, and evaluation helpers."""
from __future__ import annotations

import copy
from collections.abc import Callable
import numpy as np
import torch

from processing.multimodal_image.modeling import train_classification


class TinyMLP(torch.nn.Module):
    """Two-hidden-layer MLP returning one raw logit or rating per input row."""
    def __init__(self, input_dim: int, hidden_dims: tuple[int, int] = (16, 8)) -> None:
        super().__init__()
        self.input_dim, self.hidden_dims = input_dim, hidden_dims
        first_hidden_dim, second_hidden_dim = hidden_dims
        self.first_layer = torch.nn.Linear(input_dim, first_hidden_dim)
        self.second_layer = torch.nn.Linear(first_hidden_dim, second_hidden_dim)
        self.output_layer = torch.nn.Linear(second_hidden_dim, 1)
        self.relu = torch.nn.ReLU()
        generator = np.random.default_rng(42)
        with torch.no_grad():
            self.first_layer.weight.copy_(torch.from_numpy(generator.normal(0., .1, (input_dim, first_hidden_dim)).T).float())
            self.first_layer.bias.zero_()
            self.second_layer.weight.copy_(torch.from_numpy(generator.normal(0., .1, (first_hidden_dim, second_hidden_dim)).T).float())
            self.second_layer.bias.zero_()
            self.output_layer.weight.copy_(torch.from_numpy(generator.normal(0., .1, (second_hidden_dim, 1)).T).float())
            self.output_layer.bias.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run a forward pass for shape ``(n_rows, input_dim)`` inputs."""
        feature_matrix = torch.as_tensor(features, dtype=torch.float32)
        if feature_matrix.ndim != 2:
            raise ValueError("features must be a two-dimensional array")
        if feature_matrix.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} input features, received {feature_matrix.shape[1]}.")
        return self.output_layer(self.relu(self.second_layer(self.relu(self.first_layer(feature_matrix)))))


def build_small_mlp(input_dim: int, hidden_dims: tuple[int, int] = (16, 8)) -> TinyMLP:
    """Build a deterministic ``input -> hidden -> hidden -> 1`` MLP."""
    if not isinstance(input_dim, int) or isinstance(input_dim, bool) or input_dim <= 0:
        raise ValueError("input_dim must be a positive integer")
    if len(hidden_dims) != 2 or any(not isinstance(width, int) or isinstance(width, bool) or width <= 0 for width in hidden_dims):
        raise ValueError("hidden_dims must contain two positive integers")
    return TinyMLP(input_dim=input_dim, hidden_dims=hidden_dims)


def build_binary_loss() -> torch.nn.BCEWithLogitsLoss:
    """Return unweighted BCE for raw logits."""
    return torch.nn.BCEWithLogitsLoss()


def build_regression_loss() -> torch.nn.MSELoss:
    """Return MSE for original-rating regression."""
    return torch.nn.MSELoss()


def build_balanced_binary_loss(y_train: torch.Tensor) -> torch.nn.BCEWithLogitsLoss:
    """Return training-label-only class-balanced BCE."""
    labels = y_train.reshape(-1)
    if set(labels.tolist()) != {0.0, 1.0}:
        raise ValueError("y_train must contain both binary labels 0 and 1")
    return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float((labels == 0).sum().item() / (labels == 1).sum().item())], dtype=torch.float32))


def build_optimizer(model: torch.nn.Module, learning_rate: float = .001) -> torch.optim.Adam:
    """Return Adam with the requested positive learning rate."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_one_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module,
                   x_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
    """Perform one parameter update and return its scalar loss."""
    if not isinstance(x_batch, torch.Tensor) or x_batch.dtype != torch.float32:
        raise TypeError("x_batch must be a torch.float32 tensor")
    if not isinstance(y_batch, torch.Tensor) or y_batch.dtype != torch.float32:
        raise TypeError("y_batch must be a torch.float32 tensor")
    model.train(); optimizer.zero_grad()
    logits = model(x_batch)
    targets = y_batch.unsqueeze(1) if y_batch.ndim == 1 else y_batch
    if targets.shape != logits.shape:
        raise ValueError(f"Expected target shape {tuple(logits.shape)}, received {tuple(targets.shape)}.")
    loss = loss_fn(logits, targets)
    loss.backward(); optimizer.step()
    return float(loss.item())


def train_one_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module,
                    x_train: torch.Tensor, y_train: torch.Tensor, batch_size: int = 16,
                    shuffle: bool = True, seed: int = 42) -> float:
    """Run one deterministic mini-batch epoch without a DataLoader."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if len(x_train) != len(y_train):
        raise ValueError("x_train and y_train must contain the same number of samples")
    if not len(x_train):
        raise ValueError("x_train and y_train must contain at least one sample")
    order = torch.randperm(len(x_train), generator=torch.Generator().manual_seed(seed)) if shuffle else torch.arange(len(x_train))
    losses = []
    for start in range(0, len(x_train), batch_size):
        indices = order[start:start + batch_size]
        losses.append(train_one_step(model, optimizer, loss_fn, x_train[indices], y_train[indices]))
    return float(np.mean(losses))


def predict_binary(model: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sigmoid probabilities and 0/1 labels."""
    model.eval()
    with torch.no_grad(): probabilities = torch.sigmoid(model(x)).reshape(-1)
    return probabilities, (probabilities >= .5).to(dtype=torch.int64)


def evaluate_binary(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """Return balanced and ordinary accuracy without updating the model."""
    _, labels = predict_binary(model, x); targets = y.reshape(-1)
    if targets.shape != labels.shape: raise ValueError(f"Expected target shape {tuple(labels.shape)}, received {tuple(targets.shape)}.")
    return {"balanced_accuracy": float(train_classification.balanced_accuracy_score(targets.numpy(), labels.numpy())), "accuracy": float(train_classification.accuracy_score(targets.numpy(), labels.numpy()))}


def prediction_diagnostics(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float | int]:
    """Return binary probability and confusion diagnostics without updates."""
    probabilities, labels = predict_binary(model, x); targets = y.reshape(-1)
    if targets.shape != labels.shape: raise ValueError(f"Expected target shape {tuple(labels.shape)}, received {tuple(targets.shape)}.")
    low, high = probabilities[targets == 0], probabilities[targets == 1]
    if not low.numel() or not high.numel(): raise ValueError("Prediction diagnostics require both LOW and HIGH labels")
    tn, fp, fn, tp = train_classification.confusion_matrix(targets.numpy(), labels.numpy(), labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "mean_p_high_true_low": float(low.mean()), "mean_p_high_true_high": float(high.mean()), "min_p_high": float(probabilities.min()), "median_p_high": float(torch.quantile(probabilities, .5)), "max_p_high": float(probabilities.max()), "predicted_high_fraction": float(labels.float().mean())}


def predict_regression(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return continuous per-row predictions."""
    model.eval()
    with torch.no_grad(): return model(x).reshape(-1)


def evaluate_regression(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """Return rating error and prediction-distribution summaries."""
    predictions, targets = predict_regression(model, x), y.reshape(-1)
    if predictions.shape != targets.shape: raise ValueError(f"Expected target shape {tuple(predictions.shape)}, received {tuple(targets.shape)}.")
    residuals = predictions - targets
    return {"rmse": float(torch.sqrt(torch.mean(residuals.square()))), "mae": float(torch.mean(torch.abs(residuals))), "mean_predicted_rating": float(predictions.mean()), "min_predicted_rating": float(predictions.min()), "median_predicted_rating": float(torch.quantile(predictions, .5)), "max_predicted_rating": float(predictions.max())}


def evaluate_rating_classification(model: torch.nn.Module, x: torch.Tensor, ratings: torch.Tensor) -> dict[str, float | int]:
    """Threshold ratings/predictions at four for classification diagnostics."""
    predictions, targets = predict_regression(model, x), ratings.reshape(-1)
    if predictions.shape != targets.shape: raise ValueError(f"Expected rating shape {tuple(predictions.shape)}, received {tuple(targets.shape)}.")
    truth, labels = (targets >= 4.).to(torch.int64), (predictions >= 4.).to(torch.int64)
    tn, fp, fn, tp = train_classification.confusion_matrix(truth.numpy(), labels.numpy(), labels=[0, 1]).ravel()
    return {"balanced_accuracy": float(train_classification.balanced_accuracy_score(truth.numpy(), labels.numpy())), "accuracy": float(train_classification.accuracy_score(truth.numpy(), labels.numpy())), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "predicted_high_fraction": float(labels.float().mean())}


def fit_with_early_stopping(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module,
                            x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor,
                            max_epochs: int = 200, batch_size: int = 16, patience: int = 20, seed: int = 42,
                            progress_callback: Callable[[int, float, float, bool], None] | None = None) -> dict[str, float | int]:
    """Fit with validation loss early stopping and in-memory best weights."""
    for name, value in (("max_epochs", max_epochs), ("batch_size", batch_size), ("patience", patience)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0: raise ValueError(f"{name} must be a positive integer")
    best_loss, best_epoch, stale, best_state, final_training_loss = float("inf"), 0, 0, None, float("nan")
    for epoch in range(1, max_epochs + 1):
        final_training_loss = train_one_epoch(model, optimizer, loss_fn, x_train, y_train, batch_size, True, seed + epoch - 1)
        model.eval()
        with torch.no_grad():
            logits = model(x_val); targets = y_val.unsqueeze(1) if y_val.ndim == 1 else y_val
            if targets.shape != logits.shape: raise ValueError(f"Expected validation target shape {tuple(logits.shape)}, received {tuple(targets.shape)}.")
            validation_loss = float(loss_fn(logits, targets))
        if validation_loss < best_loss:
            best_loss, best_epoch, best_state, stale = validation_loss, epoch, copy.deepcopy(model.state_dict()), 0
            improved = True
        else:
            stale += 1
            improved = False
        if progress_callback is not None:
            progress_callback(epoch, final_training_loss, validation_loss, improved)
        if not improved:
            if stale >= patience: break
    if best_state is None: raise RuntimeError("No validation loss was recorded")
    model.load_state_dict(best_state)
    return {"epochs_run": epoch, "best_epoch": best_epoch, "best_validation_loss": best_loss, "final_training_loss": final_training_loss}
