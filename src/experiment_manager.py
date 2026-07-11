# -*- coding: utf-8 -*-

"""
Manages experiment directories and best-model tracking.

Experiments are saved in :
    experiments/cnn_YYYYMMDD_HHMMSS/
    experiments/resnet_YYYYMMDD_HHMMSS/

Best models are stored separately in :
    best_models/cnn/
    best_models/resnet/
"""

import json
import os
import re
import shutil
import stat
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from config import PROJECT_ROOT


EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
BEST_MODELS_DIR = PROJECT_ROOT / "best_models"


def chemin_relatif(path):
    """Returns a path relative to the project root."""

    return Path(path).resolve().relative_to(
        PROJECT_ROOT.resolve()
    )


def nettoyer_nom(name):
    """Converts a name into a safe directory name."""

    cleaned_name = str(name).strip().lower()

    cleaned_name = re.sub(
        pattern=r"[^a-zA-Z0-9_-]+",
        repl="_",
        string=cleaned_name,
    )

    return cleaned_name.strip("_") or "experience"


def normaliser_nom_modele(model_name):
    """Normalizes the model family name."""

    normalized_name = nettoyer_nom(
        model_name
    )

    if normalized_name == "cnn":
        return "cnn"

    if normalized_name in [
        "resnet",
        "resnet50",
    ]:
        return "resnet"

    return normalized_name


def create_experiment_directory(model_name):
    """Creates a unique experiment directory."""

    EXPERIMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    model_family = normaliser_nom_modele(
        model_name
    )

    base_name = (
        f"{model_family}_{timestamp}"
    )

    experiment_directory = (
        EXPERIMENTS_DIR / base_name
    )

    counter = 1

    while experiment_directory.exists():
        experiment_directory = (
            EXPERIMENTS_DIR
            / f"{base_name}_{counter}"
        )
        counter += 1

    experiment_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT CREATED")
    print("=" * 70)
    print(f"Name : {experiment_directory.name}")
    print(
        f"Path : "
        f"{chemin_relatif(experiment_directory)}"
    )

    return experiment_directory


def convertir_json(value):
    """Converts values into JSON-compatible objects."""

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, tuple):
        return [
            convertir_json(item)
            for item in value
        ]

    if isinstance(value, list):
        return [
            convertir_json(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            key: convertir_json(item)
            for key, item in value.items()
        }

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def save_json(data, output_path, afficher=True):
    """Saves a dictionary as a JSON file."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            convertir_json(data),
            file,
            indent=4,
        )

    if afficher:
        print(
            f"File saved : "
            f"{chemin_relatif(output_path)}"
        )


def save_model_summary(
    model,
    output_path,
):
    """Saves the model summary."""

    output_path = Path(output_path)

    summary_lines = []

    model.summary(
        print_fn=lambda line: summary_lines.append(line)
    )

    with open(
        output_path,
        mode="w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(summary_lines))

    print(
        f"File saved : "
        f"{chemin_relatif(output_path)}"
    )


def save_training_history(
    history,
    output_path,
):
    """Saves the training history."""

    history_dictionary = {
        key: [
            float(value)
            for value in values
        ]
        for key, values in history.history.items()
    }

    save_json(
        data=history_dictionary,
        output_path=output_path,
    )


def plot_training_history(
    history,
    experiment_directory,
):
    """Creates accuracy and loss curves."""

    experiment_directory = Path(
        experiment_directory
    )

    history_data = history.history

    epochs = range(
        1,
        len(history_data["loss"]) + 1,
    )

    accuracy_path = (
        experiment_directory
        / "accuracy.png"
    )

    loss_path = (
        experiment_directory
        / "loss.png"
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
        "Accuracy evolution"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
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
        "Loss evolution"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        loss_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"Accuracy curve : "
        f"{chemin_relatif(accuracy_path)}"
    )
    print(
        f"Loss curve     : "
        f"{chemin_relatif(loss_path)}"
    )


def save_experiment_metadata(
    experiment_directory,
    model_name,
    class_names,
):
    """Saves experiment metadata."""

    experiment_directory = Path(
        experiment_directory
    )

    metadata = {
        "model_name": model_name,
        "model_family": normaliser_nom_modele(
            model_name
        ),
        "experiment_name": (
            experiment_directory.name
        ),
        "experiment_directory": str(
            chemin_relatif(
                experiment_directory
            )
        ),
        "created_at": (
            datetime.now().isoformat()
        ),
        "class_names": list(class_names),
    }

    save_json(
        data=metadata,
        output_path=(
            experiment_directory
            / "experiment_metadata.json"
        ),
        afficher=False,
    )


def supprimer_lecture_seule(
    function,
    path,
    exception,
):
    """Removes the read-only attribute and retries."""

    try:
        os.chmod(
            path,
            stat.S_IRUSR
            | stat.S_IWUSR
            | stat.S_IXUSR,
        )

        function(path)

    except Exception:
        if isinstance(exception, BaseException):
            raise exception

        if (
            isinstance(exception, tuple)
            and len(exception) >= 2
            and isinstance(
                exception[1],
                BaseException,
            )
        ):
            raise exception[1]

        raise


def supprimer_dossier(directory):
    """Deletes a directory, including read-only content."""

    directory = Path(directory)

    if not directory.exists():
        return

    try:
        shutil.rmtree(
            directory,
            onexc=supprimer_lecture_seule,
        )

    except TypeError:
        shutil.rmtree(
            directory,
            onerror=supprimer_lecture_seule,
        )


def copier_si_existe(
    source,
    destination,
):
    """Copies a file if it exists."""

    source = Path(source)
    destination = Path(destination)

    if not source.exists():
        return False

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        destination,
    )

    return True


def obtenir_meilleure_validation(history):
    """Returns the best validation accuracy and its epoch."""

    values = history.history.get(
        "val_accuracy"
    )

    if not values:
        raise KeyError(
            "val_accuracy is missing from the training history."
        )

    best_value = max(values)
    best_epoch = values.index(best_value) + 1

    return float(best_value), int(best_epoch)


def lire_meilleur_precedent(
    best_model_directory,
):
    """Reads the previous best-model information."""

    info_path = (
        Path(best_model_directory)
        / "best_model_info.json"
    )

    if not info_path.exists():
        return None

    with open(
        info_path,
        mode="r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def update_best_model(
    experiment_directory,
    model_name,
    history,
    evaluation_results=None,
):
    """
    Compares the current experiment with the previous best model from the same family.
    """

    experiment_directory = Path(
        experiment_directory
    )

    model_family = normaliser_nom_modele(
        model_name
    )

    best_model_directory = (
        BEST_MODELS_DIR / model_family
    )

    current_score, best_epoch = (
        obtenir_meilleure_validation(history)
    )

    previous_info = lire_meilleur_precedent(
        best_model_directory
    )

    previous_score = None

    if previous_info is not None:
        previous_score = previous_info.get(
            "best_validation_accuracy"
        )

    is_new_best = (
        previous_score is None
        or current_score > float(previous_score)
    )

    print("\n" + "=" * 70)
    print("BEST MODEL COMPARISON")
    print("=" * 70)
    print(
        f"Best validation accuracy "
        f"de l'experience : {current_score:.4f}"
    )

    if previous_score is None:
        print(
            "No previous best model was found."
        )
    else:
        print(
            f"Previous best validation accuracy : "
            f"{float(previous_score):.4f}"
        )

    if not is_new_best:
        print(
            "The previous best model is kept."
        )

        return {
            "is_new_best": False,
            "current_validation_accuracy": (
                current_score
            ),
            "previous_validation_accuracy": (
                previous_score
            ),
        }

    if best_model_directory.exists():
        supprimer_dossier(
            best_model_directory
        )

    best_model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    files_to_copy = [
        "best_model.keras",
        "model.keras",
        "model_config.json",
        "training_config.json",
        "history.json",
        "training_log.csv",
        "model_summary.txt",
        "accuracy.png",
        "loss.png",
        "evaluation.json",
        "predictions.csv",
        "confusion_matrix.png",
        "experiment_metadata.json",
    ]

    for filename in files_to_copy:
        copier_si_existe(
            experiment_directory / filename,
            best_model_directory / filename,
        )

    best_model_info = {
        "model_family": model_family,
        "source_experiment": (
            experiment_directory.name
        ),
        "selection_metric": "val_accuracy",
        "best_validation_accuracy": (
            current_score
        ),
        "best_epoch": best_epoch,
        "test_accuracy": (
            None
            if evaluation_results is None
            else evaluation_results.get(
                "test_accuracy"
            )
        ),
        "test_loss": (
            None
            if evaluation_results is None
            else evaluation_results.get(
                "test_loss"
            )
        ),
        "updated_at": (
            datetime.now().isoformat()
        ),
    }

    save_json(
        data=best_model_info,
        output_path=(
            best_model_directory
            / "best_model_info.json"
        ),
        afficher=False,
    )

    print(
        "The model from this experiment becomes "
        "the new best model."
    )
    print(
        f"Best model : "
        f"{chemin_relatif(best_model_directory)}"
    )

    return {
        "is_new_best": True,
        "current_validation_accuracy": (
            current_score
        ),
        "previous_validation_accuracy": (
            previous_score
        ),
        "best_model_directory": str(
            chemin_relatif(
                best_model_directory
            )
        ),
    }
