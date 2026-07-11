# -*- coding: utf-8 -*-

"""
General project configuration.
"""

from pathlib import Path


# Project root directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Main project directories.
DATA_DIR = PROJECT_ROOT / "data" # Contains the training, validation, and test image folders.
RAW_DATA_DIR = DATA_DIR / "raw" # Contains the downloaded raw dataset.
TRAIN_DIR = DATA_DIR / "train"
VALIDATION_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

#MODELS_DIR = PROJECT_ROOT / "models" # Contains saved models.
#OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
BEST_MODELS_DIR = PROJECT_ROOT / "best_models"

# Image and training parameters.
IMAGE_SIZE = (224, 224) # Image resize dimensions used for training.
BATCH_SIZE = 64 # Batch size.
SEED = 42
EPOCHS = 20 # Default number of epochs.
LEARNING_RATE = 1e-4

# Dataset classes corresponding to the downloaded dataset subdirectories.
CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

# Dataset split percentages.
# They are automatically normalized in split_dataset.py if their sum differs from 100.
TRAIN_SPLIT_PERCENTAGE = 70.0
VALIDATION_SPLIT_PERCENTAGE = 15.0
TEST_SPLIT_PERCENTAGE = 15.0
