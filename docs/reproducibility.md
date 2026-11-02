# Reproducibility and verification guide

This guide describes **what can be reproduced from the public repository**,
how to run the existing pipeline, and what evidence is still missing for an
independent verification of the published model scores.

## Local setup

Use a recent supported Python version in an isolated environment. From the
repository root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the environment before installing dependencies (see the
[main README](../README.md#installation) for Windows, macOS and Linux examples).

Downloading the dataset requires network access and access to the dataset
through `kagglehub`; neither the dataset nor trained model files are stored
in Git.

## Execute the existing workflow

Open `notebooks/main_pipeline_project.ipynb` and run its cells in order.
The notebook orchestrates the existing modules under `src/`:

1. `download_dataset.py` downloads the Garbage Classification dataset
   into `data/raw/`.
2. `split_dataset.py` builds `data/train/`, `data/validation/` and
   `data/test/`.
3. `data_preprocessing.py` loads and resizes images for TensorFlow.
4. `model_builder.py` and `train.py` define and train the CNN or ResNet50.
5. `evaluate.py` produces evaluation metrics and plots; the notebook also
   provides test-image prediction and best-model comparison.

The dataset split and training/evaluation steps generate files locally.
Re-running training can take substantial time and may require suitable
hardware. There is no committed one-command script that reconstructs the
published final experiment from a saved checkpoint.

## Dataset split and determinism

The current configuration in `src/config.py` specifies:

| Setting | Value |
| --- | ---: |
| Seed | 42 |
| Training allocation | 70% |
| Validation allocation | 15% |
| Test allocation | 15% |
| Image size | 224 × 224 |
| Batch size | 64 |

`split_dataset.py` processes each class separately, using scikit-learn
`train_test_split(..., random_state=SEED, shuffle=True)`. When the three
percentages do not total 100, it normalizes them proportionally. The splits
are generated per class, and rounding may affect the final percentages.

The executed project notebook records the following split for the dataset
used in the academic project:

| Split | Recorded images |
| --- | ---: |
| Training | 1,766 |
| Validation | 378 |
| Test | 383 |
| **Total** | **2,527** |

These counts describe the recorded dataset and should not be assumed to
match a different upstream dataset revision.

## Model selection and evaluation

The training pipeline selects checkpoints based on **validation accuracy**,
not test accuracy. Evaluation reports may include test loss/accuracy,
per-class precision/recall/F1, predictions and a confusion matrix.

The [main README](../README.md#results-at-a-glance) reports rounded scores
from the final academic presentation: 86% test accuracy for ResNet50 and
68% for the custom CNN. These are **reported results, not a new evaluation
performed by this documentation change**.

Published figures under `results/cnn/` and `results/resnet/` illustrate
training behavior and confusion matrices. Full trained checkpoints,
per-run configuration metadata, exact predictions and evaluation JSON
are currently absent from the public repository. As a result, the exact
reported results are **not independently reproducible from the repository
alone**.

## Checks that do not require the dataset

To check that the tracked Python source files parse:

```bash
python -m compileall -q src
```

This is a **syntax check only**. It does not execute training, validate
model quality, verify dependency compatibility, or reproduce the
published accuracy numbers.

## What to record for a future reproducible experiment

For any newly published run, preserve the exact dataset version or hash,
the split configuration and random seed, Python and package versions,
training hyperparameters, model-selection criteria, relevant
hardware/runtime information, the trained checkpoint, and the matching
evaluation report. Keep the held-out test set separate from hyperparameter
and checkpoint selection.
