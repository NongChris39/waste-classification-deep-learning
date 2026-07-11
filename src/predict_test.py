# -*- coding: utf-8 -*-

"""
Predicts a test image using an experiment model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import preprocess_input

from config import (
    IMAGE_SIZE,
    TEST_DIR,
)

from experiment_manager import chemin_relatif
from model_builder import PretraitementResNet50


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def charger_modele(model_path):
    """Loads a current model or a legacy ResNet50 model."""

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


def obtenir_dossier_classe(
    class_name,
    class_names,
):
    """Returns the test directory for a class."""

    normalized_class = class_name.lower().strip()

    if normalized_class not in class_names:
        raise ValueError(
            f"Unknown class : '{class_name}'."
        )

    class_directory = (
        TEST_DIR / normalized_class
    )

    if not class_directory.exists():
        raise FileNotFoundError(
            "Class directory not found :\n"
            f"{chemin_relatif(class_directory)}"
        )

    return class_directory


def list_available_images(
    class_name,
    class_names,
):
    """Returns the available images."""

    class_directory = obtenir_dossier_classe(
        class_name,
        class_names,
    )

    return sorted(
        [
            path
            for path in class_directory.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ]
    )


def display_available_images(
    class_name,
    class_names,
    maximum=20,
):
    """Displays the names of available images."""

    images = list_available_images(
        class_name,
        class_names,
    )

    print("\n" + "=" * 70)
    print(
        f"AVAILABLE IMAGES - {class_name}"
    )
    print("=" * 70)

    for image_path in images[:maximum]:
        print(f"- {image_path.name}")

    if len(images) > maximum:
        print(
            f"... and "
            f"{len(images) - maximum} more image(s)."
        )


def build_test_image_path(
    class_name,
    image_name,
    class_names,
):
    """Builds the path to the selected image."""

    image_path = (
        obtenir_dossier_classe(
            class_name,
            class_names,
        )
        / image_name
    )

    if not image_path.exists():
        raise FileNotFoundError(
            "Image not found :\n"
            f"{chemin_relatif(image_path)}"
        )

    return image_path


def display_selected_image(
    image_path,
):
    """Displays the image before prediction."""

    image = tf.keras.utils.load_img(
        image_path,
        color_mode="rgb",
    )

    image_array = (
        tf.keras.utils.img_to_array(image)
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(
        image_array.astype("uint8")
    )
    plt.axis("off")
    plt.title(
        f"Image to predict\n"
        f"True class : {image_path.parent.name}\n"
        f"File : {image_path.name}"
    )
    plt.tight_layout()
    plt.show()


def obtenir_modele_experience(
    experiment_directory,
    use_best_model=True,
):
    """Returns the experiment model."""

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


def predict_experiment_image(
    experiment_directory,
    image_path,
    class_names,
    use_best_model=True,
):
    """Predicts the class of an image."""

    model_path = obtenir_modele_experience(
        experiment_directory,
        use_best_model,
    )

    model = charger_modele(model_path)

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb",
    )

    image_array = (
        tf.keras.utils.img_to_array(image)
    )

    image_batch = np.expand_dims(
        image_array,
        axis=0,
    )

    probabilities = model.predict(
        image_batch,
        verbose=0,
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    true_class = image_path.parent.name

    print("\n" + "=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)
    print(
        f"Image         : "
        f"{chemin_relatif(image_path)}"
    )
    print(
        f"Model        : "
        f"{chemin_relatif(model_path)}"
    )
    print(f"True class : {true_class}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence     : {confidence:.2%}")

    print("\nClass probabilities")

    for class_name, probability in zip(
        class_names,
        probabilities,
    ):
        print(
            f"{class_name:<12}: "
            f"{float(probability):.2%}"
        )

    plt.figure(figsize=(8, 6))
    plt.imshow(
        image_array.astype("uint8")
    )
    plt.axis("off")
    plt.title(
        f"True class : {true_class}\n"
        f"Predicted class : {predicted_class}\n"
        f"Confidence : {confidence:.2%}"
    )
    plt.tight_layout()

    output_path = (
        Path(experiment_directory)
        / f"prediction_{image_path.stem}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    # plt.show()
    plt.close()

    return {
        "true_class": true_class,
        "predicted_class": predicted_class,
        "predicted_index": predicted_index,
        "confidence": confidence,
        "image_path": str(
            chemin_relatif(image_path)
        ),
        "model_path": str(
            chemin_relatif(model_path)
        ),
    }