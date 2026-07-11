# -*- coding: utf-8 -*-

"""
Model training and result storage in an experiment directory.
"""

from pathlib import Path

import tensorflow as tf

from config import PROJECT_ROOT

from experiment_manager import (
    chemin_relatif,
    plot_training_history,
    save_json,
    save_model_summary,
    save_training_history,
)


class JournalCompact(tf.keras.callbacks.Callback):
    """
    Displays only the main metrics at the end of each epoch.
    """

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

# =============================================================================
#         print(
#             f"Epoch {epoch + 1:02d} - "
#             f"accuracy: {logs.get('accuracy', 0):.4f} - "
#             f"val_accuracy: {logs.get('val_accuracy', 0):.4f} - "
#             f"loss: {logs.get('loss', 0):.4f} - "
#             f"val_loss: {logs.get('val_loss', 0):.4f}",
#             flush=True,
#         )
# 
# =============================================================================

def build_callbacks(
    experiment_directory,
    training_config,
):
    """Builds the training callbacks."""

    experiment_directory = Path(
        experiment_directory
    )

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=(
                experiment_directory
                / "best_model.keras"
            ),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=training_config.get(
                "early_stopping_patience",
                7,
            ),
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=training_config.get(
                "reduce_lr_factor",
                0.30,
            ),
            patience=training_config.get(
                "reduce_lr_patience",
                3,
            ),
            min_lr=training_config.get(
                "minimum_learning_rate",
                1e-7,
            ),
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=(
                experiment_directory
                / "training_log.csv"
            ),
            append=False,
        ),
        JournalCompact(),
    ]


def train_model(
    model,
    train_dataset,
    validation_dataset,
    experiment_directory,
    model_config,
    training_config,
):
    """Trains an already-built model."""

    experiment_directory = Path(
        experiment_directory
    )

    model_config_path = (
        experiment_directory
        / "model_config.json"
    )

    training_config_path = (
        experiment_directory
        / "training_config.json"
    )

    model_summary_path = (
        experiment_directory
        / "model_summary.txt"
    )

    save_json(
        model_config,
        model_config_path,
    )

    save_json(
        training_config,
        training_config_path,
    )

    save_model_summary(
        model,
        model_summary_path,
    )

    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)
    print(
        f"Maximum number of epochs : "
        f"{training_config.get('epochs', 30)}"
    )

    callbacks = build_callbacks(
        experiment_directory,
        training_config,
    )

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=training_config.get(
            "epochs",
            30,
        ),
        callbacks=callbacks,
        verbose=1,
    )

    final_model_path = (
        experiment_directory
        / "model.keras"
    )

    model.save(
        final_model_path,
        overwrite=True,
    )

    history_path = (
        experiment_directory
        / "history.json"
    )

    save_training_history(
        history,
        history_path,
    )

    plot_training_history(
        history,
        experiment_directory,
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(
        f"History     : "
        f"{chemin_relatif(history_path)}"
    )
    print(
        f"Final model   : "
        f"{chemin_relatif(final_model_path)}"
    )
    print(
        f"Best model de l'experience : "
        f"{chemin_relatif(experiment_directory / 'best_model.keras')}"
    )

    return model, history
