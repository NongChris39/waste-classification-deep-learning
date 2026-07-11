# -*- coding: utf-8 -*-

"""
Telechargement du dataset Garbage Classification.

The dataset is downloaded from Kaggle and copied to data/raw.
"""

import os
import shutil
import stat
from pathlib import Path

import kagglehub

from config import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
)


DATASET_IDENTIFIER = (
    "asdasdasasdas/garbage-classification"
)


def chemin_relatif(path):
    """Returns a path relative to the project root."""

    return Path(path).resolve().relative_to(
        PROJECT_ROOT.resolve()
    )


def supprimer_lecture_seule(
    fonction,
    chemin,
    exception,
):
    """
    Removes the read-only attribute and retries the deletion operation. 
    Required because previous dataset folders are removed before a new download.
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


def supprimer_fichier(file_path):
    """Deletes a file, including a read-only file."""

    file_path = Path(file_path)

    if not file_path.exists():
        return

    try:
        file_path.unlink()

    except PermissionError:
        os.chmod(
            file_path,
            stat.S_IRUSR
            | stat.S_IWUSR,
        )

        file_path.unlink()


def copier_element(
    source,
    destination,
):
    """Copies a downloaded file or directory."""

    source = Path(source)
    destination = Path(destination)

    if source.is_dir():
        if destination.exists():
            supprimer_dossier(
                destination
            )

        shutil.copytree(
            source,
            destination,
        )

    else:
        if destination.exists():
            supprimer_fichier(
                destination
            )

        shutil.copy2(
            source,
            destination,
        )


def download_dataset():
    """Downloads and copies the dataset into data/raw."""

    print("Downloading dataset...")

    downloaded_path = Path(
        kagglehub.dataset_download(
            DATASET_IDENTIFIER
        )
    )

    if not downloaded_path.exists():
        raise FileNotFoundError(
            "The downloaded directory was not found."
        )

    downloaded_items = list(
        downloaded_path.iterdir()
    )

    if not downloaded_items:
        raise ValueError(
            "The downloaded directory is empty."
        )

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for source_item in downloaded_items:
        copier_element(
            source=source_item,
            destination=(
                RAW_DATA_DIR
                / source_item.name
            ),
        )

    print(
        f"Dataset saved to : "
        f"{chemin_relatif(RAW_DATA_DIR)}"
    )

    return RAW_DATA_DIR
