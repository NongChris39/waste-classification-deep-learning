# -*- coding: utf-8 -*-

"""
Evaluates an experiment model on the test dataset.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from tensorflow.keras.applications.resnet50 import preprocess_input

from experiment_manager import (
    chemin_relatif,
    save_json,
)

from model_builder import PretraitementResNet50


def charger_modele(model_path):
    """
    Loads both current models and legacy ResNet models containing Lambda(preprocess_input).
    """

    custom_objects = {
        "PretraitementResNet50": PretraitementResNet50,
        "ProjetPratique>PretraitementResNet50": PretraitementResNet50,
        "preprocess_input": preprocess_input,
    }

    try:
        return tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
        )

    except TypeError:
        return tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            safe_mode=False,
        )


def obtenir_labels_reels(dataset):
    """Extracts the true labels from a dataset."""

    labels = []

    for _, batch_labels in dataset:
        labels.extend(
            batch_labels.numpy().tolist()
        )

    return np.asarray(labels)


def obtenir_modele_experience(
    experiment_directory,
    use_best_model=True,
):
    """Returns the model to evaluate."""

    filename = (
        "best_model.keras"
        if use_best_model
        else "model.keras"
    )

    model_path = (
        Path(experiment_directory)
        / filename
    )

    if not model_path.exists():
        raise FileNotFoundError(
            "Model not found :\n"
            f"{chemin_relatif(model_path)}"
        )

    return model_path


def sauvegarder_matrice_confusion(
    true_labels,
    predicted_labels,
    class_names,
    experiment_directory,
):
    """Creates and saves the confusion matrix."""

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
    )

    figure, axis = plt.subplots(
        figsize=(9, 8)
    )

    display_matrix = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    display_matrix.plot(
        ax=axis,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    axis.set_title("Confusion matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")

    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = (
        Path(experiment_directory)
        / "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def evaluate_experiment(
    experiment_directory,
    test_dataset,
    class_names,
    use_best_model=True,
):
    """Evaluates an experiment on the test dataset."""

    experiment_directory = Path(
        experiment_directory
    )

    model_path = obtenir_modele_experience(
        experiment_directory,
        use_best_model,
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT EVALUATION")
    print("=" * 70)
    print(
        f"Evaluated model : "
        f"{chemin_relatif(model_path)}"
    )

    model = charger_modele(model_path)

    test_loss, test_accuracy = model.evaluate(
        test_dataset,
        verbose=0,
    )

    probabilities = model.predict(
        test_dataset,
        verbose=0,
    )

    predicted_labels = np.argmax(
        probabilities,
        axis=1,
    )

    true_labels = obtenir_labels_reels(
        test_dataset
    )

    report_text = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    report_dictionary = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    print("\nTest results")
    print("-" * 40)
    print(f"Test loss     : {test_loss:.4f}")
    print(f"Test accuracy : {test_accuracy:.4f}")

    print("\nClassification report")
    print("-" * 70)
    print(report_text)

    confusion_matrix_path = (
        sauvegarder_matrice_confusion(
            true_labels,
            predicted_labels,
            class_names,
            experiment_directory,
        )
    )

    predictions_path = (
        experiment_directory
        / "predictions.csv"
    )

    pd.DataFrame(
        {
            "true_label_index": true_labels,
            "predicted_label_index": predicted_labels,
            "true_class": [
                class_names[index]
                for index in true_labels
            ],
            "predicted_class": [
                class_names[index]
                for index in predicted_labels
            ],
            "correct_prediction": (
                true_labels == predicted_labels
            ),
            "confidence": np.max(
                probabilities,
                axis=1,
            ),
        }
    ).to_csv(
        predictions_path,
        index=False,
    )

    evaluation_results = {
        "model_path": str(
            chemin_relatif(model_path)
        ),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "number_of_test_images": int(
            len(true_labels)
        ),
        "class_names": list(class_names),
        "classification_report": report_dictionary,
    }

    evaluation_path = (
        experiment_directory
        / "evaluation.json"
    )

    save_json(
        evaluation_results,
        evaluation_path,
        afficher=False,
    )

    print("\nGenerated files")
    print("-" * 40)
    print(
        f"Evaluation          : "
        f"{chemin_relatif(evaluation_path)}"
    )
    print(
        f"Predictions         : "
        f"{chemin_relatif(predictions_path)}"
    )
    print(
        f"Confusion matrix: "
        f"{chemin_relatif(confusion_matrix_path)}"
    )

    return evaluation_results