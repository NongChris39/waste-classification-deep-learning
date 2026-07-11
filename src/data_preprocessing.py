# -*- coding: utf-8 -*-

"""
Dataset loading and preprocessing.
Classs are detected automatically from directory names,
and images are resized to the configured dimensions.

This module creates TensorFlow datasets for:
- training;
- validation;
- testing.
"""

import tensorflow as tf

from config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
    VALIDATION_DIR,
)


AUTOTUNE = tf.data.AUTOTUNE #Lets TensorFlow automatically tune input pipeline performance.


def verifier_dossier(directory, nom_dataset):
    """Checks that a dataset directory exists."""

    if not directory.exists():
        raise FileNotFoundError(
            f"The dataset directory for {nom_dataset} "
            f"was not found :\n{directory}"
        )


def charger_dataset(
    directory,
    shuffle,
):
    """Loads an image dataset from a directory."""

    return tf.keras.utils.image_dataset_from_directory(
        directory=directory,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED if shuffle else None,
        verbose=False,
    )


def optimiser_dataset(
    dataset,
    cache=False,
):
    """Optimizes dataset input performance."""

    if cache:
        dataset = dataset.cache()

    return dataset.prefetch(
        buffer_size=AUTOTUNE #
    )


def afficher_informations_dataset(
    class_names,
    train_dataset,
    validation_dataset,
    test_dataset,
):
    """Displays only the main dataset information."""

    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print(
        f"Image size : "
        f"{IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}"
    )
    print(
        f"Batch size : "
        f"{BATCH_SIZE}"
    )
    print(
        f"Number of classes : "
        f"{len(class_names)}"
    )

    print("\nDetected classes :")

    for index, class_name in enumerate(
        class_names,
        start=1,
    ):
        print(f"- {index}. {class_name}")

    print("\nNumber of batches :")
    print(
        f"- Training : "
        f"{tf.data.experimental.cardinality(train_dataset).numpy()}"
    )
    print(
        f"- Validation   : "
        f"{tf.data.experimental.cardinality(validation_dataset).numpy()}"
    )
    print(
        f"- Test         : "
        f"{tf.data.experimental.cardinality(test_dataset).numpy()}"
    )


def get_datasets():
    """
    Loads and returns the training, validation, and test datasets.
    """

    verifier_dossier(
        TRAIN_DIR,
        "training data",
    )

    verifier_dossier(
        VALIDATION_DIR,
        "validation data",
    )

    verifier_dossier(
        TEST_DIR,
        "test data",
    )

    train_dataset = charger_dataset(
        directory=TRAIN_DIR,
        shuffle=True,
    )

    validation_dataset = charger_dataset(
        directory=VALIDATION_DIR,
        shuffle=False,
    )

    test_dataset = charger_dataset(
        directory=TEST_DIR,
        shuffle=False,
    )

    class_names = list(
        train_dataset.class_names
    )

    validation_class_names = list(
        validation_dataset.class_names
    )

    test_class_names = list(
        test_dataset.class_names
    )

    if (
        class_names != validation_class_names
        or class_names != test_class_names
    ):
        raise ValueError(
            "Les classes detectees ne sont pas "
            "identiques dans les jeux train, "
            "validation et test."
        )

    train_dataset = optimiser_dataset(
        train_dataset,
        cache=False,
    )

    validation_dataset = optimiser_dataset(
        validation_dataset,
        cache=True,
    )

    test_dataset = optimiser_dataset(
        test_dataset,
        cache=True,
    )

    afficher_informations_dataset(
        class_names=class_names,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names,
    )
