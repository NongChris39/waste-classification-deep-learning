# -*- coding: utf-8 -*-

"""
Splits the dataset into training, validation, and test sets.
"""

import os
import shutil
import stat
from pathlib import Path

from sklearn.model_selection import train_test_split

from config import (
    CLASS_NAMES,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    SEED,
    TEST_DIR,
    TEST_SPLIT_PERCENTAGE,
    TRAIN_DIR,
    TRAIN_SPLIT_PERCENTAGE,
    VALIDATION_DIR,
    VALIDATION_SPLIT_PERCENTAGE,
)


SOURCE_DATA_DIR = (
    RAW_DATA_DIR
    / "Garbage classification"
    / "Garbage classification"
)

# Supported image extensions.
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def chemin_relatif(path):
    """Returns a path relative to the project root."""

    return Path(path).resolve().relative_to(
        PROJECT_ROOT.resolve()
    )


def normaliser_pourcentages():
    """
    Validates the dataset split percentages.

    If their sum differs from 100, they are normalized proportionally.
    """

    train_percentage = float(
        TRAIN_SPLIT_PERCENTAGE
    )
    validation_percentage = float(
        VALIDATION_SPLIT_PERCENTAGE
    )
    test_percentage = float(
        TEST_SPLIT_PERCENTAGE
    )

    percentages = [
        train_percentage,
        validation_percentage,
        test_percentage,
    ]
    
    # Validate the split percentages defined in config.py.
    
    if any(value < 0 for value in percentages):
        raise ValueError(
            "Les pourcentages de decoupage "
            "ne peuvent pas etre negatifs."
        )

    total = sum(percentages)

    if total <= 0:
        raise ValueError(
            "La somme des pourcentages doit "
            "etre superieure à zero."
        )

    if abs(total - 100.0) > 1e-9:
        train_percentage = (
            train_percentage / total * 100.0
        )
        validation_percentage = (
            validation_percentage
            / total
            * 100.0
        )
        test_percentage = (
            test_percentage / total * 100.0
        )

        print(
            "The sum of the dataset split percentages "
            f"was {total:.2f} %."
        )
        print(
            "Les pourcentages ont été ajustés "
            "proportionnellement :"
        )
        print(
            f"- Training : "
            f"{train_percentage:.2f} %"
        )
        print(
            f"- Validation   : "
            f"{validation_percentage:.2f} %"
        )
        print(
            f"- Test         : "
            f"{test_percentage:.2f} %"
        )

    return (
        train_percentage,
        validation_percentage,
        test_percentage,
    )


def supprimer_lecture_seule(
    fonction,
    chemin,
    exception,
):
    """
    Removes the read-only attribute and retries the deletion operation.
    """

    try:
        os.chmod(
            chemin,
            stat.S_IRUSR
            | stat.S_IWUSR
            | stat.S_IXUSR,
        )

        fonction(chemin)

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


def reinitialiser_dossier(directory):
    """Deletes and recreates a directory."""

    directory = Path(directory)

    if directory.exists():
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

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def obtenir_images(directory):
    """Returns all valid images."""

    return sorted(
        [
            path
            for path in Path(directory).rglob("*")
            if path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ]
    )


def copier_images(
    images,
    destination_directory,
):
    """Copies images into a directory."""

    destination_directory = Path(
        destination_directory
    )

    destination_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, image_path in enumerate(images):
        destination_path = (
            destination_directory
            / image_path.name
        )

        if destination_path.exists():
            destination_path = (
                destination_directory
                / (
                    f"{image_path.stem}_{index}"
                    f"{image_path.suffix.lower()}"
                )
            )

        shutil.copy2(
            image_path,
            destination_path,
        )


def decouper_classe(
    class_name,
    train_percentage,
    validation_percentage,
    test_percentage,
):
    """Splits one class according to the requested percentages."""

    source_directory = (
        SOURCE_DATA_DIR / class_name
    )

    if not source_directory.exists():
        raise FileNotFoundError(
            "Class directory not found :\n"
            f"{chemin_relatif(source_directory)}"
        )

    images = obtenir_images(
        source_directory
    )

    if not images:
        raise ValueError(
            f"No valid image was found for class "
            f"'{class_name}'."
        )

    temporary_percentage = (
        validation_percentage
        + test_percentage
    )

    if temporary_percentage == 0:
        train_images = images
        validation_images = []
        test_images = []

    elif train_percentage == 0:
        train_images = []

        if validation_percentage == 0:
            validation_images = []
            test_images = images

        elif test_percentage == 0:
            validation_images = images
            test_images = []

        else:
            validation_ratio = (
                validation_percentage
                / temporary_percentage
            )

            validation_images, test_images = (
                train_test_split(
                    images,
                    train_size=validation_ratio,
                    random_state=SEED,
                    shuffle=True,
                )
            )

    else:
        train_ratio = (
            train_percentage / 100.0
        )

        train_images, temporary_images = (
            train_test_split(
                images,
                train_size=train_ratio,
                random_state=SEED,
                shuffle=True,
            )
        )

        if validation_percentage == 0:
            validation_images = []
            test_images = temporary_images

        elif test_percentage == 0:
            validation_images = temporary_images
            test_images = []

        else:
            validation_ratio = (
                validation_percentage
                / temporary_percentage
            )

            validation_images, test_images = (
                train_test_split(
                    temporary_images,
                    train_size=validation_ratio,
                    random_state=SEED,
                    shuffle=True,
                )
            )

    copier_images(
        train_images,
        TRAIN_DIR / class_name,
    )

    copier_images(
        validation_images,
        VALIDATION_DIR / class_name,
    )

    copier_images(
        test_images,
        TEST_DIR / class_name,
    )

    return {
        "train": len(train_images),
        "validation": len(validation_images),
        "test": len(test_images),
        "total": len(images),
    }


def afficher_resume(
    results,
    train_percentage,
    validation_percentage,
    test_percentage,
):
    """Displays the complete split summary."""

    print("\n" + "=" * 72)
    print("DATASET SPLIT SUMMARY")
    print("=" * 72)

    print(
        f"{'Class':<15}"
        f"{'Train':>10}"
        f"{'Validation':>15}"
        f"{'Test':>10}"
        f"{'Total':>10}"
    )

    print("-" * 72)

    total_train = 0
    total_validation = 0
    total_test = 0
    total_images = 0

    for class_name, counts in results.items():
        print(
            f"{class_name:<15}"
            f"{counts['train']:>10}"
            f"{counts['validation']:>15}"
            f"{counts['test']:>10}"
            f"{counts['total']:>10}"
        )

        total_train += counts["train"]
        total_validation += counts["validation"]
        total_test += counts["test"]
        total_images += counts["total"]

    print("-" * 72)

    print(
        f"{'TOTAL':<15}"
        f"{total_train:>10}"
        f"{total_validation:>15}"
        f"{total_test:>10}"
        f"{total_images:>10}"
    )

    print("\nRequested percentages :")
    print(
        f"- Training : "
        f"{train_percentage:.2f} %"
    )
    print(
        f"- Validation   : "
        f"{validation_percentage:.2f} %"
    )
    print(
        f"- Test         : "
        f"{test_percentage:.2f} %"
    )

    print("\nActual percentages after integer rounding :")
    print(
        f"- Training : "
        f"{100 * total_train / total_images:.2f} %"
    )
    print(
        f"- Validation   : "
        f"{100 * total_validation / total_images:.2f} %"
    )
    print(
        f"- Test         : "
        f"{100 * total_test / total_images:.2f} %"
    )


def split_dataset():
    """Recreates the training, validation, and test sets."""

    if not SOURCE_DATA_DIR.exists():
        raise FileNotFoundError(
            "The main source directory was not found:\n"
            f"{chemin_relatif(SOURCE_DATA_DIR)}"
        )

    (
        train_percentage,
        validation_percentage,
        test_percentage,
    ) = normaliser_pourcentages()

    print("=" * 72)
    print("DATASET SPLIT")
    print("=" * 72)
    print(
        f"Source     : "
        f"{chemin_relatif(SOURCE_DATA_DIR)}"
    )
    print(
        f"Train      : "
        f"{chemin_relatif(TRAIN_DIR)}"
    )
    print(
        f"Validation : "
        f"{chemin_relatif(VALIDATION_DIR)}"
    )
    print(
        f"Test       : "
        f"{chemin_relatif(TEST_DIR)}"
    )

    reinitialiser_dossier(TRAIN_DIR)
    reinitialiser_dossier(VALIDATION_DIR)
    reinitialiser_dossier(TEST_DIR)

    results = {}

    for class_name in CLASS_NAMES:
        results[class_name] = decouper_classe(
            class_name=class_name,
            train_percentage=train_percentage,
            validation_percentage=(
                validation_percentage
            ),
            test_percentage=test_percentage,
        )

    afficher_resume(
        results=results,
        train_percentage=train_percentage,
        validation_percentage=(
            validation_percentage
        ),
        test_percentage=test_percentage,
    )
