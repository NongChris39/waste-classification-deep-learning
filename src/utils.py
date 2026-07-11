# -*- coding: utf-8 -*-

"""
Utility functions for the waste classification project.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import (
    FIGURES_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    PREDICTIONS_DIR,
)


def create_project_directories() -> None:
    """Create all output directories required by the project."""

    directories = [
        MODELS_DIR,
        OUTPUTS_DIR,
        FIGURES_DIR,
        PREDICTIONS_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def save_training_history(
    history: tf.keras.callbacks.History,
    output_path: Path,
) -> None:
    """Save the Keras training history as a JSON file."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_dictionary = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with open(
        output_path,
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history_dictionary,
            file,
            indent=4,
        )

    print(
        f"Training history saved to: "
        f"{output_path.resolve()}"
    )


def load_training_history(
    history_path: Path,
) -> dict:
    """Load a training history from a JSON file."""

    if not history_path.exists():
        raise FileNotFoundError(
            f"Training history not found: "
            f"{history_path.resolve()}"
        )

    with open(
        history_path,
        mode="r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def plot_training_history(
    history,
    model_name: str,
) -> None:
    """Create and save accuracy and loss curves."""

    create_project_directories()

    if hasattr(history, "history"):
        history_data = history.history
    else:
        history_data = history

    required_keys = [
        "loss",
        "val_loss",
        "accuracy",
        "val_accuracy",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in history_data
    ]

    if missing_keys:
        raise KeyError(
            f"Missing history values: {missing_keys}"
        )

    epochs = range(
        1,
        len(history_data["loss"]) + 1,
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        history_data["accuracy"],
        label="Training",
    )

    plt.plot(
        epochs,
        history_data["val_accuracy"],
        label="Validation",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        f"Accuracy evolution - {model_name}"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    accuracy_path = (
        FIGURES_DIR
        / f"{model_name}_accuracy.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        history_data["loss"],
        label="Training",
    )

    plt.plot(
        epochs,
        history_data["val_loss"],
        label="Validation",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"Loss evolution - {model_name}"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    loss_path = (
        FIGURES_DIR
        / f"{model_name}_loss.png"
    )

    plt.savefig(
        loss_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Accuracy curve saved to: "
        f"{accuracy_path.resolve()}"
    )

    print(
        f"Loss curve saved to: "
        f"{loss_path.resolve()}"
    )


def get_true_labels(
    dataset: tf.data.Dataset,
) -> np.ndarray:
    """Extract all true labels from a TensorFlow dataset."""

    labels = []

    for _, batch_labels in dataset:
        labels.extend(
            batch_labels.numpy().tolist()
        )

    return np.asarray(labels)


def validate_image_path(
    image_path: Path,
) -> None:
    """Check that an image path exists and is a file."""

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: "
            f"{image_path.resolve()}"
        )

    if not image_path.is_file():
        raise ValueError(
            f"Path is not a file: "
            f"{image_path.resolve()}"
        )


if __name__ == "__main__":
    create_project_directories()

    print("Project directories are ready.")
    print(f"Models      : {MODELS_DIR.resolve()}")
    print(f"Outputs     : {OUTPUTS_DIR.resolve()}")
    print(f"Figures     : {FIGURES_DIR.resolve()}")
    print(f"Predictions : {PREDICTIONS_DIR.resolve()}")